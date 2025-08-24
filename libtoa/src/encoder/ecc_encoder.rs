use alloc::vec::Vec;

use crate::{
    ErrorCorrection, Result, Write,
    reed_solomon::{code_255_191, code_255_223, code_255_239, get_generator_poly},
};

struct CircularBuffer {
    buffer: Vec<u8>,
    start: usize,
    len: usize,
}

impl CircularBuffer {
    fn with_capacity(capacity: usize) -> Self {
        // Use power-of-2 for efficient modulo via bit masking.
        let capacity = capacity.next_power_of_two().max(4096);

        Self {
            buffer: vec![0u8; capacity],
            start: 0,
            len: 0,
        }
    }

    #[inline]
    fn available_data(&self) -> usize {
        self.len
    }

    /// Get contiguous slices without copying.
    fn as_slices(&self) -> (&[u8], &[u8]) {
        if self.len == 0 {
            return (&[], &[]);
        }

        let end = (self.start + self.len) & (self.buffer.len() - 1);

        if end > self.start {
            // Data is contiguous.
            (&self.buffer[self.start..self.start + self.len], &[])
        } else if end == 0 {
            // Data fills to the end.
            (&self.buffer[self.start..], &[])
        } else {
            // Data wraps around.
            (&self.buffer[self.start..], &self.buffer[..end])
        }
    }

    fn copy_to(&self, out: &mut [u8]) -> usize {
        let copy_len = out.len().min(self.len);
        if copy_len == 0 {
            return 0;
        }

        let (first, second) = self.as_slices();

        if copy_len <= first.len() {
            out[..copy_len].copy_from_slice(&first[..copy_len]);
        } else {
            let first_len = first.len();
            out[..first_len].copy_from_slice(first);
            let remaining = copy_len - first_len;
            if remaining > 0 && !second.is_empty() {
                out[first_len..copy_len].copy_from_slice(&second[..remaining]);
            }
        }

        copy_len
    }

    fn append(&mut self, data: &[u8]) {
        if data.is_empty() {
            return;
        }

        // Grow if needed (double size).
        if self.len + data.len() > self.buffer.len() {
            let new_capacity = (self.len + data.len()).next_power_of_two();
            let mut new_buffer = vec![0u8; new_capacity];

            let copied = self.copy_to(&mut new_buffer[..self.len]);
            debug_assert_eq!(copied, self.len);

            self.buffer = new_buffer;
            self.start = 0;
        }

        let write_pos = (self.start + self.len) & (self.buffer.len() - 1);
        let available_at_end = self.buffer.len() - write_pos;

        if data.len() <= available_at_end {
            self.buffer[write_pos..write_pos + data.len()].copy_from_slice(data);
        } else {
            self.buffer[write_pos..].copy_from_slice(&data[..available_at_end]);
            self.buffer[..data.len() - available_at_end].copy_from_slice(&data[available_at_end..]);
        }

        self.len += data.len();
    }

    #[inline]
    fn consume(&mut self, count: usize) {
        let consume_count = count.min(self.len);
        self.start = (self.start + consume_count) & (self.buffer.len() - 1);
        self.len -= consume_count;
    }

    fn fill_batch_from_buffer<const BATCH: usize, const DATA_LEN: usize>(
        &self,
        batch_codewords: &mut [[u8; DATA_LEN]; BATCH],
        batch_data_size: usize,
    ) -> bool {
        let (first_slice, second_slice) = self.as_slices();

        if first_slice.len() >= batch_data_size {
            // Data is contiguous, copy directly from first slice.
            for (i, codeword) in batch_codewords.iter_mut().enumerate() {
                let start = i * DATA_LEN;
                codeword.copy_from_slice(&first_slice[start..start + DATA_LEN]);
            }
            return true;
        } else if first_slice.len() + second_slice.len() >= batch_data_size {
            // Data wraps around, handle both slices.
            let mut processed_bytes = 0;
            let mut codeword_idx = 0;

            // Fill codewords from first slice.
            while processed_bytes + DATA_LEN <= first_slice.len() && codeword_idx < BATCH {
                batch_codewords[codeword_idx]
                    .copy_from_slice(&first_slice[processed_bytes..processed_bytes + DATA_LEN]);
                processed_bytes += DATA_LEN;
                codeword_idx += 1;
            }

            // Handle partial codeword at slice boundary if needed.
            if processed_bytes < first_slice.len() && codeword_idx < BATCH {
                let remaining_in_first = first_slice.len() - processed_bytes;
                let needed_from_second = DATA_LEN - remaining_in_first;

                if needed_from_second <= second_slice.len() {
                    batch_codewords[codeword_idx][..remaining_in_first]
                        .copy_from_slice(&first_slice[processed_bytes..]);
                    batch_codewords[codeword_idx][remaining_in_first..DATA_LEN]
                        .copy_from_slice(&second_slice[..needed_from_second]);
                    codeword_idx += 1;
                    processed_bytes = needed_from_second;
                }
            } else {
                processed_bytes = 0;
            }

            // Fill remaining codewords from second slice.
            while processed_bytes + DATA_LEN <= second_slice.len() && codeword_idx < BATCH {
                batch_codewords[codeword_idx]
                    .copy_from_slice(&second_slice[processed_bytes..processed_bytes + DATA_LEN]);
                processed_bytes += DATA_LEN;
                codeword_idx += 1;
            }

            return true;
        }

        false
    }
}

type EncodeFunction<W> = fn(&mut ECCEncoder<W>, &[u8]) -> Result<usize>;

const BATCH_SIZE_AVX512: usize = 64;
const BATCH_SIZE_AVX2: usize = 32;

fn encode_none<W: Write>(encoder: &mut ECCEncoder<W>, data: &[u8]) -> Result<usize> {
    encoder.inner.write_all(data)?;
    Ok(data.len())
}

fn encode_standard<W: Write>(encoder: &mut ECCEncoder<W>, data: &[u8]) -> Result<usize> {
    encoder.encode_with_rs::<_, 239, 16>(data, code_255_239::encode)
}

fn encode_paranoid<W: Write>(encoder: &mut ECCEncoder<W>, data: &[u8]) -> Result<usize> {
    encoder.encode_with_rs::<_, 223, 32>(data, code_255_223::encode)
}

fn encode_extreme<W: Write>(encoder: &mut ECCEncoder<W>, data: &[u8]) -> Result<usize> {
    encoder.encode_with_rs::<_, 191, 64>(data, code_255_191::encode)
}

#[cfg(target_arch = "x86_64")]
fn encode_standard_avx512<W: Write>(encoder: &mut ECCEncoder<W>, data: &[u8]) -> Result<usize> {
    unsafe { encoder.encode_batch_avx512::<239, 16>(data) }
}

#[cfg(target_arch = "x86_64")]
fn encode_paranoid_avx512<W: Write>(encoder: &mut ECCEncoder<W>, data: &[u8]) -> Result<usize> {
    unsafe { encoder.encode_batch_avx512::<223, 32>(data) }
}

#[cfg(target_arch = "x86_64")]
fn encode_extreme_avx512<W: Write>(encoder: &mut ECCEncoder<W>, data: &[u8]) -> Result<usize> {
    unsafe { encoder.encode_batch_avx512::<191, 64>(data) }
}

#[cfg(target_arch = "x86_64")]
fn encode_standard_avx2<W: Write>(encoder: &mut ECCEncoder<W>, data: &[u8]) -> Result<usize> {
    unsafe { encoder.encode_batch_avx2::<239, 16>(data) }
}

#[cfg(target_arch = "x86_64")]
fn encode_paranoid_avx2<W: Write>(encoder: &mut ECCEncoder<W>, data: &[u8]) -> Result<usize> {
    unsafe { encoder.encode_batch_avx2::<223, 32>(data) }
}

#[cfg(target_arch = "x86_64")]
fn encode_extreme_avx2<W: Write>(encoder: &mut ECCEncoder<W>, data: &[u8]) -> Result<usize> {
    unsafe { encoder.encode_batch_avx2::<191, 64>(data) }
}

fn detect_simd_capability() -> (bool, usize) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("gfni") {
            return (true, BATCH_SIZE_AVX512);
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("gfni") {
            return (true, BATCH_SIZE_AVX2);
        }
    }

    (false, 1)
}

/// Error Correction Code Writer that applies Reed-Solomon encoding to compressed data.
pub struct ECCEncoder<W> {
    inner: W,
    encode_fn: EncodeFunction<W>,
    encode_fn_simd: Option<EncodeFunction<W>>,
    buffer: CircularBuffer,
    uses_buffer: bool,
    batch_size: usize,
}

impl<W: Write> ECCEncoder<W> {
    /// Create a new ECCWriter with the specified error correction level.
    pub fn new(inner: W, error_correction: ErrorCorrection) -> Self {
        let (use_simd, simd_batch_size) = detect_simd_capability();

        let (encode_fn, uses_buffer, codeword_data_len) = match error_correction {
            ErrorCorrection::None => (encode_none as EncodeFunction<W>, false, 0),
            ErrorCorrection::Standard => (encode_standard as EncodeFunction<W>, true, 239),
            ErrorCorrection::Paranoid => (encode_paranoid as EncodeFunction<W>, true, 223),
            ErrorCorrection::Extreme => (encode_extreme as EncodeFunction<W>, true, 191),
        };

        let encode_fn_simd = match (use_simd, simd_batch_size, error_correction) {
            #[cfg(target_arch = "x86_64")]
            (true, 64, ErrorCorrection::Standard) => {
                Some(encode_standard_avx512 as EncodeFunction<W>)
            }
            #[cfg(target_arch = "x86_64")]
            (true, 64, ErrorCorrection::Paranoid) => {
                Some(encode_paranoid_avx512 as EncodeFunction<W>)
            }
            #[cfg(target_arch = "x86_64")]
            (true, 64, ErrorCorrection::Extreme) => {
                Some(encode_extreme_avx512 as EncodeFunction<W>)
            }
            #[cfg(target_arch = "x86_64")]
            (true, 32, ErrorCorrection::Standard) => {
                Some(encode_standard_avx2 as EncodeFunction<W>)
            }
            #[cfg(target_arch = "x86_64")]
            (true, 32, ErrorCorrection::Paranoid) => {
                Some(encode_paranoid_avx2 as EncodeFunction<W>)
            }
            #[cfg(target_arch = "x86_64")]
            (true, 32, ErrorCorrection::Extreme) => Some(encode_extreme_avx2 as EncodeFunction<W>),
            _ => None,
        };

        let batch_size = if encode_fn_simd.is_some() {
            simd_batch_size * codeword_data_len
        } else {
            codeword_data_len
        };

        let buffer = if uses_buffer {
            CircularBuffer::with_capacity(batch_size * 2)
        } else {
            CircularBuffer::with_capacity(0)
        };

        Self {
            inner,
            encode_fn,
            encode_fn_simd,
            buffer,
            uses_buffer,
            batch_size,
        }
    }

    fn encode_and_write_data(&mut self, data: &[u8]) -> Result<usize> {
        if let Some(simd_fn) = self.encode_fn_simd {
            simd_fn(self, data)
        } else {
            (self.encode_fn)(self, data)
        }
    }

    #[inline(always)]
    fn encode_with_rs<F, const DATA_LEN: usize, const PARITY_LEN: usize>(
        &mut self,
        data: &[u8],
        encode_rs_fn: F,
    ) -> Result<usize>
    where
        F: Fn(&[u8; DATA_LEN]) -> [u8; PARITY_LEN],
    {
        let mut input_processed = 0;

        while self.buffer.available_data() >= DATA_LEN
            || (data.is_empty() && self.buffer.available_data() > 0)
        {
            let mut codeword_data = [0u8; DATA_LEN];
            let copied = self.buffer.copy_to(&mut codeword_data);

            // Zero-padding is already in place from initialization.
            let parity = encode_rs_fn(&codeword_data);
            self.inner.write_all(&codeword_data)?;
            self.inner.write_all(&parity)?;

            self.buffer.consume(copied.min(DATA_LEN));

            if copied < DATA_LEN {
                break;
            }
        }

        if data.is_empty() {
            return Ok(0);
        }

        let mut pos = 0;
        while pos + DATA_LEN <= data.len() {
            let mut codeword_data = [0u8; DATA_LEN];
            codeword_data.copy_from_slice(&data[pos..pos + DATA_LEN]);

            let parity = encode_rs_fn(&codeword_data);
            self.inner.write_all(&codeword_data)?;
            self.inner.write_all(&parity)?;

            pos += DATA_LEN;
            input_processed += DATA_LEN;
        }

        Ok(input_processed)
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f,gfni")]
    unsafe fn encode_batch_avx512<const DATA_LEN: usize, const PARITY_LEN: usize>(
        &mut self,
        data: &[u8],
    ) -> Result<usize> {
        const BATCH: usize = 64;
        let batch_data_size = BATCH * DATA_LEN;
        let mut input_processed = 0;

        if self.buffer.available_data() >= batch_data_size {
            let mut batch_codewords = [[0u8; DATA_LEN]; BATCH];

            if self
                .buffer
                .fill_batch_from_buffer::<BATCH, DATA_LEN>(&mut batch_codewords, batch_data_size)
            {
                unsafe {
                    encode_simd_batch_avx512::<_, BATCH, DATA_LEN, PARITY_LEN>(
                        &mut self.inner,
                        &batch_codewords,
                    )?;
                }

                self.buffer.consume(batch_data_size);
                return Ok(0);
            }
        }

        // Try to process aligned data directly without copying.
        let (left, aligned, _right) = unsafe { data.align_to::<[u8; DATA_LEN]>() };

        if left.is_empty() && aligned.len() >= BATCH {
            // Data is perfectly aligned and we have enough for at least one batch!
            let batches_possible = aligned.len() / BATCH;

            for batch_idx in 0..batches_possible {
                let batch_start = batch_idx * BATCH;
                let batch_slice = &aligned[batch_start..batch_start + BATCH];

                // Safe transmute: we know the slice has exactly BATCH elements of [u8; DATA_LEN]
                let batch_codewords: &[[u8; DATA_LEN]; BATCH] =
                    unsafe { &*(batch_slice.as_ptr() as *const [[u8; DATA_LEN]; BATCH]) };

                unsafe {
                    encode_simd_batch_avx512::<_, BATCH, DATA_LEN, PARITY_LEN>(
                        &mut self.inner,
                        batch_codewords,
                    )?;
                }

                input_processed += batch_data_size;
            }

            return Ok(input_processed);
        }

        // Fallback to copying approach for misaligned or insufficient data.
        let mut pos = 0;
        while pos + batch_data_size <= data.len() {
            let mut batch_codewords = [[0u8; DATA_LEN]; BATCH];

            for (i, codeword) in batch_codewords.iter_mut().enumerate() {
                let start = pos + i * DATA_LEN;
                let end = start + DATA_LEN;
                codeword.copy_from_slice(&data[start..end]);
            }

            unsafe {
                encode_simd_batch_avx512::<_, BATCH, DATA_LEN, PARITY_LEN>(
                    &mut self.inner,
                    &batch_codewords,
                )?;
            }

            pos += batch_data_size;
            input_processed += batch_data_size;
        }

        Ok(input_processed)
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,gfni")]
    unsafe fn encode_batch_avx2<const DATA_LEN: usize, const PARITY_LEN: usize>(
        &mut self,
        data: &[u8],
    ) -> Result<usize> {
        const BATCH: usize = 32;
        let batch_data_size = BATCH * DATA_LEN;
        let mut input_processed = 0;

        if self.buffer.available_data() >= batch_data_size {
            let mut batch_codewords = [[0u8; DATA_LEN]; BATCH];

            if self
                .buffer
                .fill_batch_from_buffer::<BATCH, DATA_LEN>(&mut batch_codewords, batch_data_size)
            {
                unsafe {
                    encode_simd_batch_avx2::<_, BATCH, DATA_LEN, PARITY_LEN>(
                        &mut self.inner,
                        &batch_codewords,
                    )?;
                }

                self.buffer.consume(batch_data_size);
                return Ok(0);
            }
        }

        // Try to process aligned data directly without copying.
        let (left, aligned, _right) = unsafe { data.align_to::<[u8; DATA_LEN]>() };

        if left.is_empty() && aligned.len() >= BATCH {
            // Data is perfectly aligned and we have enough for at least one batch!
            let batches_possible = aligned.len() / BATCH;

            for batch_idx in 0..batches_possible {
                let batch_start = batch_idx * BATCH;
                let batch_slice = &aligned[batch_start..batch_start + BATCH];

                // Safe transmute: we know the slice has exactly BATCH elements of [u8; DATA_LEN]
                let batch_codewords: &[[u8; DATA_LEN]; BATCH] =
                    unsafe { &*(batch_slice.as_ptr() as *const [[u8; DATA_LEN]; BATCH]) };

                unsafe {
                    encode_simd_batch_avx2::<_, BATCH, DATA_LEN, PARITY_LEN>(
                        &mut self.inner,
                        batch_codewords,
                    )?;
                }

                input_processed += batch_data_size;
            }

            return Ok(input_processed);
        }

        // Fallback to copying approach for misaligned or insufficient data.
        let mut pos = 0;
        while pos + batch_data_size <= data.len() {
            let mut batch_codewords = [[0u8; DATA_LEN]; BATCH];

            for (i, codeword) in batch_codewords.iter_mut().enumerate() {
                let start = pos + i * DATA_LEN;
                let end = start + DATA_LEN;
                codeword.copy_from_slice(&data[start..end]);
            }

            unsafe {
                encode_simd_batch_avx2::<_, BATCH, DATA_LEN, PARITY_LEN>(
                    &mut self.inner,
                    &batch_codewords,
                )?;
            }

            pos += batch_data_size;
            input_processed += batch_data_size;
        }

        Ok(input_processed)
    }

    /// Finish writing and flush any remaining data.
    pub fn finish(mut self) -> Result<W> {
        if self.buffer.available_data() > 0 {
            // Process remaining buffered data.
            (self.encode_fn)(&mut self, &[])?;
        }

        Ok(self.inner)
    }
}

impl<W: Write> Write for ECCEncoder<W> {
    fn write(&mut self, buf: &[u8]) -> Result<usize> {
        if !self.uses_buffer {
            return self.inner.write(buf);
        }

        if buf.is_empty() {
            return Ok(0);
        }

        let mut buf_pos = 0;

        // If we have buffered data, try to fill it to a processable amount.
        if self.buffer.available_data() > 0 {
            let needed = self.batch_size - self.buffer.available_data();
            let available = buf.len().min(needed);

            self.buffer.append(&buf[..available]);
            buf_pos += available;

            // Try to process the buffer if it's full enough, else return.
            match self.buffer.available_data() >= self.batch_size {
                true => {
                    self.encode_and_write_data(&[])?;
                }
                false => {
                    // Not enough data to encode yet.
                    return Ok(available);
                }
            }
        }

        let remaining = buf.len() - buf_pos;

        if remaining >= self.batch_size {
            // We have enough for a complete batch, process directly.
            let processed = self.encode_and_write_data(&buf[buf_pos..])?;
            if buf_pos + processed < buf.len() {
                self.buffer.append(&buf[buf_pos + processed..]);
            }
        } else {
            // Not enough for a complete batch, store in buffer.
            if remaining > 0 {
                self.buffer.append(&buf[buf_pos..]);
            }
        }

        Ok(buf.len())
    }

    fn flush(&mut self) -> Result<()> {
        self.inner.flush()
    }
}

/// Transpose codewords for SIMD processing
/// From: [codeword0][codeword1][codeword2]...
/// To:   [byte0_all][byte1_all][byte2_all]...
#[cfg(target_arch = "x86_64")]
fn transpose_for_simd<const BATCH: usize, const LEN: usize>(
    batched_data: &[[u8; LEN]; BATCH],
) -> [[u8; BATCH]; LEN] {
    let mut transposed = [[0u8; BATCH]; LEN];

    for (codeword_idx, codeword) in batched_data.iter().enumerate() {
        for (byte_idx, &byte) in codeword.iter().enumerate() {
            transposed[byte_idx][codeword_idx] = byte;
        }
    }

    transposed
}

/// Transpose back after SIMD processing
/// From:   [byte0_all][byte1_all][byte2_all]...
/// To:     [codeword0][codeword1][codeword2]...
#[cfg(target_arch = "x86_64")]
fn transpose_from_simd<const BATCH: usize, const DATA_LEN: usize, const PARITY_LEN: usize>(
    transposed_data: &[[u8; BATCH]; DATA_LEN],
    transposed_parity: &[[u8; BATCH]; PARITY_LEN],
) -> ([[u8; DATA_LEN]; BATCH], [[u8; PARITY_LEN]; BATCH]) {
    let mut data_codewords = [[0u8; DATA_LEN]; BATCH];
    let mut parity_codewords = [[0u8; PARITY_LEN]; BATCH];

    for (byte_idx, byte_batch) in transposed_data.iter().enumerate() {
        for (codeword_idx, &byte) in byte_batch.iter().enumerate() {
            data_codewords[codeword_idx][byte_idx] = byte;
        }
    }

    for (byte_idx, byte_batch) in transposed_parity.iter().enumerate() {
        for (codeword_idx, &byte) in byte_batch.iter().enumerate() {
            parity_codewords[codeword_idx][byte_idx] = byte;
        }
    }

    (data_codewords, parity_codewords)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,gfni")]
unsafe fn encode_simd_batch_avx512<
    R: Write,
    const BATCH: usize,
    const DATA_LEN: usize,
    const PARITY_LEN: usize,
>(
    mut writer: R,
    batch_codewords: &[[u8; DATA_LEN]; BATCH],
) -> Result<()> {
    use core::arch::x86_64::*;

    let transposed_data = transpose_for_simd::<BATCH, DATA_LEN>(batch_codewords);
    let gen_poly = get_generator_poly::<PARITY_LEN>();

    // LFSR-based encoding with SIMD.
    let mut remainder = [[0u8; BATCH]; PARITY_LEN];

    // Process each data byte position (from highest to lowest).
    for data_bytes in transposed_data.iter().rev() {
        let data_vec = unsafe { _mm512_loadu_si512(data_bytes.as_ptr() as *const __m512i) };

        // XOR with feedback from the highest remainder position.
        let feedback_vec =
            unsafe { _mm512_loadu_si512(remainder[PARITY_LEN - 1].as_ptr() as *const __m512i) };
        let feedback = _mm512_xor_si512(data_vec, feedback_vec);

        // Shift remainder right.
        for i in (1..PARITY_LEN).rev() {
            remainder[i] = remainder[i - 1];
        }
        remainder[0] = [0u8; BATCH];

        // Apply generator polynomial multiplication with GFNI.
        for (i, &g_coeff) in gen_poly[..PARITY_LEN].iter().enumerate() {
            if g_coeff != 0 {
                let g_vec = _mm512_set1_epi8(g_coeff as i8);
                let product = _mm512_gf2p8mul_epi8(feedback, g_vec);

                let current =
                    unsafe { _mm512_loadu_si512(remainder[i].as_ptr() as *const __m512i) };
                let result = _mm512_xor_si512(current, product);
                unsafe { _mm512_storeu_si512(remainder[i].as_mut_ptr() as *mut __m512i, result) };
            }
        }
    }

    let (data_codewords, parity_codewords) =
        transpose_from_simd::<BATCH, DATA_LEN, PARITY_LEN>(&transposed_data, &remainder);

    for i in 0..BATCH {
        writer.write_all(&data_codewords[i])?;
        writer.write_all(&parity_codewords[i])?;
    }

    Ok(())
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,gfni")]
unsafe fn encode_simd_batch_avx2<
    R: Write,
    const BATCH: usize,
    const DATA_LEN: usize,
    const PARITY_LEN: usize,
>(
    mut writer: R,
    batch_codewords: &[[u8; DATA_LEN]; BATCH],
) -> Result<()> {
    use core::arch::x86_64::*;

    let transposed_data = transpose_for_simd::<BATCH, DATA_LEN>(batch_codewords);
    let gen_poly = get_generator_poly::<PARITY_LEN>();

    // LFSR-based encoding with SIMD.
    let mut remainder = [[0u8; BATCH]; PARITY_LEN];

    // Process each data byte position (from highest to lowest).
    for data_bytes in transposed_data.iter().rev() {
        let data_vec = unsafe { _mm256_loadu_si256(data_bytes.as_ptr() as *const __m256i) };

        // XOR with feedback from the highest remainder position.
        let feedback_vec =
            unsafe { _mm256_loadu_si256(remainder[PARITY_LEN - 1].as_ptr() as *const __m256i) };
        let feedback = _mm256_xor_si256(data_vec, feedback_vec);

        // Shift remainder right.
        for i in (1..PARITY_LEN).rev() {
            remainder[i] = remainder[i - 1];
        }
        remainder[0] = [0u8; BATCH];

        // Apply generator polynomial multiplication with GFNI.
        for (i, &g_coeff) in gen_poly[..PARITY_LEN].iter().enumerate() {
            if g_coeff != 0 {
                let g_vec = _mm256_set1_epi8(g_coeff as i8);
                let product = _mm256_gf2p8mul_epi8(feedback, g_vec);

                let current =
                    unsafe { _mm256_loadu_si256(remainder[i].as_ptr() as *const __m256i) };
                let result = _mm256_xor_si256(current, product);
                unsafe { _mm256_storeu_si256(remainder[i].as_mut_ptr() as *mut __m256i, result) };
            }
        }
    }

    let (data_codewords, parity_codewords) =
        transpose_from_simd::<BATCH, DATA_LEN, PARITY_LEN>(&transposed_data, &remainder);

    for i in 0..BATCH {
        writer.write_all(&data_codewords[i])?;
        writer.write_all(&parity_codewords[i])?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use super::*;
    use crate::{
        reed_solomon::{code_255_191, code_255_223, code_255_239},
        tests::Lcg,
    };

    #[test]
    fn test_ec_encoder_none_passthrough() {
        let mut output = Vec::new();
        let mut ec_encoder = ECCEncoder::new(&mut output, ErrorCorrection::None);

        let test_data = b"Hello, World!";
        ec_encoder.write_all(test_data).unwrap();

        let _final_output = ec_encoder.finish().unwrap();

        assert_eq!(output, test_data);
    }

    #[test]
    fn test_ec_encoder_standard_encoding() {
        let mut output = Vec::new();
        let mut ec_encoder = ECCEncoder::new(&mut output, ErrorCorrection::Standard);

        let test_data = b"Hello, Reed-Solomon encoding!";
        ec_encoder.write_all(test_data).unwrap();

        let _final_output = ec_encoder.finish().unwrap();

        assert_eq!(output.len(), 255);

        assert_eq!(&output[..test_data.len()], test_data);

        for (i, &x) in output[test_data.len()..239].iter().enumerate() {
            assert_eq!(x, 0, "Padding should be zero at position {i}");
        }

        assert_eq!(output[239..].len(), 16);
    }

    #[test]
    fn test_ec_encoder_multiple_codewords() {
        let mut output = Vec::new();
        let mut ec_encoder = ECCEncoder::new(&mut output, ErrorCorrection::Standard);

        let mut test_data = Vec::new();
        test_data.extend_from_slice(b"A".repeat(300).as_slice());

        ec_encoder.write_all(&test_data).unwrap();
        let _final_output = ec_encoder.finish().unwrap();

        // We should have 2 codewords: 2 * 255 = 510 bytes.
        assert_eq!(output.len(), 2 * 255);

        assert_eq!(&output[..239], &test_data[0..239]);

        let second_codeword_data_size = 300 - 239;

        assert_eq!(
            &output[255..255 + second_codeword_data_size],
            &test_data[239..]
        );

        for (i, &x) in output[(255 + second_codeword_data_size)..(255 + 239)]
            .iter()
            .enumerate()
        {
            assert_eq!(x, 0, "Padding should be zero at position {i}");
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_direct_simd_vs_scalar_functions() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("gfni") {
            println!("GFNI not supported");
            return;
        }
        println!("GFNI supported");

        fn test_simd_batch<const BATCH: usize, const DATA_LEN: usize, const PARITY_LEN: usize>(
            rng: &mut Lcg,
            encode_fn: fn(&[u8; DATA_LEN]) -> [u8; PARITY_LEN],
            simd_fn: impl FnOnce(&mut Vec<u8>, &[[u8; DATA_LEN]; BATCH]) -> Result<()>,
            test_name: &str,
        ) {
            let mut batch_data = [[0u8; DATA_LEN]; BATCH];
            for codeword in &mut batch_data {
                rng.fill_buffer(codeword);
            }

            let mut scalar_output = Vec::new();
            let mut simd_output = Vec::new();

            for codeword in &batch_data {
                let parity = encode_fn(codeword);
                scalar_output.extend_from_slice(codeword);
                scalar_output.extend_from_slice(&parity);
            }

            simd_fn(&mut simd_output, &batch_data).unwrap();

            assert_eq!(
                blake3::hash(&scalar_output),
                blake3::hash(&simd_output),
                "{test_name}"
            );
        }

        let mut rng = Lcg::new(0x123456789ABCDEF0);

        test_simd_batch::<32, 239, 16>(
            &mut rng,
            code_255_239::encode,
            |out, data| unsafe { encode_simd_batch_avx2::<_, 32, 239, 16>(out, data) },
            "Direct AVX2 vs scalar mismatch for Standard error correction",
        );

        test_simd_batch::<32, 223, 32>(
            &mut rng,
            code_255_223::encode,
            |out, data| unsafe { encode_simd_batch_avx2::<_, 32, 223, 32>(out, data) },
            "Direct AVX2 vs scalar mismatch for Paranoid error correction",
        );

        test_simd_batch::<32, 191, 64>(
            &mut rng,
            code_255_191::encode,
            |out, data| unsafe { encode_simd_batch_avx2::<_, 32, 191, 64>(out, data) },
            "Direct AVX2 vs scalar mismatch for Extreme error correction",
        );

        if is_x86_feature_detected!("avx512f") {
            test_simd_batch::<64, 239, 16>(
                &mut rng,
                code_255_239::encode,
                |out, data| unsafe { encode_simd_batch_avx512::<_, 64, 239, 16>(out, data) },
                "Direct AVX512 vs scalar mismatch for Standard error correction",
            );

            test_simd_batch::<64, 223, 32>(
                &mut rng,
                code_255_223::encode,
                |out, data| unsafe { encode_simd_batch_avx512::<_, 64, 223, 32>(out, data) },
                "Direct AVX512 vs scalar mismatch for Paranoid error correction",
            );

            test_simd_batch::<64, 191, 64>(
                &mut rng,
                code_255_191::encode,
                |out, data| unsafe { encode_simd_batch_avx512::<_, 64, 191, 64>(out, data) },
                "Direct AVX512 vs scalar mismatch for Extreme error correction",
            );
        }
    }
}
