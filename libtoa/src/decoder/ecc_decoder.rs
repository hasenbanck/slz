use alloc::vec::Vec;

use crate::{
    ErrorCorrection, Read, Result, error_invalid_data,
    reed_solomon::{code_255_191, code_255_223, code_255_239, gf_alpha_pow},
};

/// Statistics for decoder performance monitoring
#[derive(Default, Debug)]
pub(crate) struct DecoderStats {
    pub blocks_processed: usize,
    pub blocks_corrected: usize,
}

/// Circular buffer for efficient batch processing
struct CircularBuffer {
    buffer: Vec<u8>,
    start: usize,
    len: usize,
}

impl CircularBuffer {
    fn with_capacity(capacity: usize) -> Self {
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

    fn append(&mut self, data: &[u8]) {
        if data.is_empty() {
            return;
        }

        // Grow if needed
        if self.len + data.len() > self.buffer.len() {
            let new_capacity = (self.len + data.len()).next_power_of_two();
            let mut new_buffer = vec![0u8; new_capacity];

            self.copy_to(&mut new_buffer[..self.len]);
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

    fn copy_to(&self, out: &mut [u8]) -> usize {
        let copy_len = out.len().min(self.len);
        if copy_len == 0 {
            return 0;
        }

        let end = (self.start + copy_len) & (self.buffer.len() - 1);

        if end > self.start || end == 0 {
            // Data is contiguous
            out[..copy_len].copy_from_slice(&self.buffer[self.start..self.start + copy_len]);
        } else {
            // Data wraps around
            let first_part = self.buffer.len() - self.start;
            out[..first_part].copy_from_slice(&self.buffer[self.start..]);
            out[first_part..copy_len].copy_from_slice(&self.buffer[..end]);
        }

        copy_len
    }

    #[inline]
    fn consume(&mut self, count: usize) {
        let consume_count = count.min(self.len);
        self.start = (self.start + consume_count) & (self.buffer.len() - 1);
        self.len -= consume_count;
    }

    fn extract_codewords<const BATCH: usize>(&mut self) -> (usize, [[u8; 255]; BATCH]) {
        let mut codewords = [[0u8; 255]; BATCH];
        let mut count = 0;

        while count < BATCH && self.available_data() >= 255 {
            self.copy_to(&mut codewords[count]);
            self.consume(255);
            count += 1;
        }

        (count, codewords)
    }
}

type DecodeFunction<R> = fn(&mut ECCDecoder<R>, &mut [u8]) -> Result<usize>;

fn decode_none<R: Read>(decoder: &mut ECCDecoder<R>, buf: &mut [u8]) -> Result<usize> {
    decoder.inner.read(buf)
}

fn decode_standard<R: Read>(decoder: &mut ECCDecoder<R>, buf: &mut [u8]) -> Result<usize> {
    decoder.decode_with_rs::<_, 239>(buf, code_255_239::decode)
}

fn decode_paranoid<R: Read>(decoder: &mut ECCDecoder<R>, buf: &mut [u8]) -> Result<usize> {
    decoder.decode_with_rs::<_, 223>(buf, code_255_223::decode)
}

fn decode_extreme<R: Read>(decoder: &mut ECCDecoder<R>, buf: &mut [u8]) -> Result<usize> {
    decoder.decode_with_rs::<_, 191>(buf, code_255_191::decode)
}

#[cfg(target_arch = "x86_64")]
fn decode_standard_avx512<R: Read>(decoder: &mut ECCDecoder<R>, buf: &mut [u8]) -> Result<usize> {
    unsafe { decoder.decode_batch_avx512::<239, 16>(buf) }
}

#[cfg(target_arch = "x86_64")]
fn decode_paranoid_avx512<R: Read>(decoder: &mut ECCDecoder<R>, buf: &mut [u8]) -> Result<usize> {
    unsafe { decoder.decode_batch_avx512::<223, 32>(buf) }
}

#[cfg(target_arch = "x86_64")]
fn decode_extreme_avx512<R: Read>(decoder: &mut ECCDecoder<R>, buf: &mut [u8]) -> Result<usize> {
    unsafe { decoder.decode_batch_avx512::<191, 64>(buf) }
}

#[cfg(target_arch = "x86_64")]
fn decode_standard_avx2<R: Read>(decoder: &mut ECCDecoder<R>, buf: &mut [u8]) -> Result<usize> {
    unsafe { decoder.decode_batch_avx2::<239, 16>(buf) }
}

#[cfg(target_arch = "x86_64")]
fn decode_paranoid_avx2<R: Read>(decoder: &mut ECCDecoder<R>, buf: &mut [u8]) -> Result<usize> {
    unsafe { decoder.decode_batch_avx2::<223, 32>(buf) }
}

#[cfg(target_arch = "x86_64")]
fn decode_extreme_avx2<R: Read>(decoder: &mut ECCDecoder<R>, buf: &mut [u8]) -> Result<usize> {
    unsafe { decoder.decode_batch_avx2::<191, 64>(buf) }
}

fn detect_simd_capability() -> (bool, usize) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("gfni") {
            return (true, 64);
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("gfni") {
            return (true, 32);
        }
    }

    (false, 1)
}

/// Error Correction Decoder that applies Reed-Solomon decoding to compressed data.
pub struct ECCDecoder<R> {
    inner: R,
    decode_fn: DecodeFunction<R>,
    decode_fn_simd: Option<DecodeFunction<R>>,
    input_buffer: CircularBuffer,
    output_buffer: CircularBuffer,
    validate_rs: bool,
    uses_buffer: bool,
    stats: DecoderStats,
}

impl<R: Read> ECCDecoder<R> {
    /// Create a new ECCDecoder with the specified error correction level.
    pub fn new(inner: R, error_correction: ErrorCorrection, validate_rs: bool) -> Self {
        let (use_simd, simd_batch_size) = detect_simd_capability();

        let (decode_fn, uses_buffer, data_len) = match error_correction {
            ErrorCorrection::None => (decode_none as DecodeFunction<R>, false, 0),
            ErrorCorrection::Standard => (decode_standard as DecodeFunction<R>, true, 239),
            ErrorCorrection::Paranoid => (decode_paranoid as DecodeFunction<R>, true, 223),
            ErrorCorrection::Extreme => (decode_extreme as DecodeFunction<R>, true, 191),
        };

        let decode_fn_simd = match (use_simd, simd_batch_size, error_correction) {
            #[cfg(target_arch = "x86_64")]
            (true, 64, ErrorCorrection::Standard) => {
                Some(decode_standard_avx512 as DecodeFunction<R>)
            }
            #[cfg(target_arch = "x86_64")]
            (true, 64, ErrorCorrection::Paranoid) => {
                Some(decode_paranoid_avx512 as DecodeFunction<R>)
            }
            #[cfg(target_arch = "x86_64")]
            (true, 64, ErrorCorrection::Extreme) => {
                Some(decode_extreme_avx512 as DecodeFunction<R>)
            }
            #[cfg(target_arch = "x86_64")]
            (true, 32, ErrorCorrection::Standard) => {
                Some(decode_standard_avx2 as DecodeFunction<R>)
            }
            #[cfg(target_arch = "x86_64")]
            (true, 32, ErrorCorrection::Paranoid) => {
                Some(decode_paranoid_avx2 as DecodeFunction<R>)
            }
            #[cfg(target_arch = "x86_64")]
            (true, 32, ErrorCorrection::Extreme) => Some(decode_extreme_avx2 as DecodeFunction<R>),
            _ => None,
        };

        let batch_size = if decode_fn_simd.is_some() {
            simd_batch_size
        } else {
            1
        };

        let input_buffer_size = if uses_buffer {
            255 * batch_size * 2 // Buffer for input codewords
        } else {
            0
        };

        let output_buffer_size = if uses_buffer {
            data_len * batch_size * 2 // Buffer for decoded data
        } else {
            0
        };

        Self {
            inner,
            decode_fn,
            decode_fn_simd,
            input_buffer: CircularBuffer::with_capacity(input_buffer_size),
            output_buffer: CircularBuffer::with_capacity(output_buffer_size),
            validate_rs,
            uses_buffer,
            stats: DecoderStats::default(),
        }
    }

    /// Generic Reed-Solomon decoding with configurable data length and decoder function.
    fn decode_with_rs<F, const DATA_LEN: usize>(
        &mut self,
        buf: &mut [u8],
        decode_rs_fn: F,
    ) -> Result<usize>
    where
        F: Fn(&mut [u8; 255]) -> Result<bool>,
    {
        let mut total_read = 0;

        // First, drain any buffered output
        if self.output_buffer.available_data() > 0 {
            let copied = self.output_buffer.copy_to(buf);
            self.output_buffer.consume(copied);
            total_read += copied;

            if total_read >= buf.len() {
                return Ok(total_read);
            }
        }

        // Process single codewords (fallback for non-batch processing)
        while total_read < buf.len() {
            // Read a complete codeword
            let mut codeword = [0u8; 255];
            let mut bytes_read = 0;

            while bytes_read < 255 {
                match self.inner.read(&mut codeword[bytes_read..]) {
                    Ok(0) => break, // EOF
                    Ok(n) => bytes_read += n,
                    Err(e) => return Err(e),
                }
            }

            if bytes_read == 0 {
                break; // EOF
            }

            if bytes_read < 255 {
                // Pad with zeros for partial read
                for i in bytes_read..255 {
                    codeword[i] = 0;
                }
            }

            if self.validate_rs {
                let corrected = decode_rs_fn(&mut codeword).map_err(|_| {
                    error_invalid_data("error correction couldn't correct a faulty block")
                })?;

                if corrected {
                    self.stats.blocks_corrected += 1;
                    eprint!("Error correction corrected a faulty block");
                }
            }

            // Copy decoded data to output
            let available = DATA_LEN.min(buf.len() - total_read);
            buf[total_read..total_read + available].copy_from_slice(&codeword[..available]);
            total_read += available;
            self.stats.blocks_processed += 1;

            if available < DATA_LEN {
                // Buffer remaining data
                self.output_buffer.append(&codeword[available..DATA_LEN]);
                break;
            }
        }

        Ok(total_read)
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f,gfni")]
    unsafe fn decode_batch_avx512<const DATA_LEN: usize, const PARITY_LEN: usize>(
        &mut self,
        buf: &mut [u8],
    ) -> Result<usize> {
        const BATCH: usize = 64;
        let mut total_read = 0;

        // First, drain any buffered output
        if self.output_buffer.available_data() > 0 {
            let copied = self.output_buffer.copy_to(buf);
            self.output_buffer.consume(copied);
            total_read += copied;

            if total_read >= buf.len() {
                return Ok(total_read);
            }
        }

        while total_read < buf.len() {
            // Fill input buffer with codewords
            while self.input_buffer.available_data() < 255 * BATCH {
                let mut temp_buf = [0u8; 4096];
                match self.inner.read(&mut temp_buf) {
                    Ok(0) => break, // EOF
                    Ok(n) => self.input_buffer.append(&temp_buf[..n]),
                    Err(e) => return Err(e),
                }
            }

            // Extract complete codewords from buffer
            let (batch_count, mut codeword_batch) = self.input_buffer.extract_codewords::<BATCH>();

            if batch_count == 0 {
                break; // No complete codewords available
            }

            // Process batch with SIMD
            let syndrome_results =
                unsafe { calculate_syndrome_batch_avx512::<BATCH, PARITY_LEN>(&codeword_batch) };

            // Process each codeword based on syndrome results
            for i in 0..batch_count {
                if !syndrome_results[i] && self.validate_rs {
                    // Errors detected - use scalar correction
                    let corrected = match PARITY_LEN {
                        16 => code_255_239::decode(&mut codeword_batch[i]),
                        32 => code_255_223::decode(&mut codeword_batch[i]),
                        64 => code_255_191::decode(&mut codeword_batch[i]),
                        _ => return Err(error_invalid_data("Invalid parity length")),
                    }
                    .map_err(|_| {
                        error_invalid_data("error correction couldn't correct a faulty block")
                    })?;

                    if corrected {
                        self.stats.blocks_corrected += 1;
                    }
                }

                // Add decoded data to output buffer
                self.output_buffer.append(&codeword_batch[i][..DATA_LEN]);
                self.stats.blocks_processed += 1;
            }

            // Copy from output buffer to user buffer
            let available = self
                .output_buffer
                .available_data()
                .min(buf.len() - total_read);
            let copied = self
                .output_buffer
                .copy_to(&mut buf[total_read..total_read + available]);
            self.output_buffer.consume(copied);
            total_read += copied;
        }

        Ok(total_read)
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,gfni")]
    unsafe fn decode_batch_avx2<const DATA_LEN: usize, const PARITY_LEN: usize>(
        &mut self,
        buf: &mut [u8],
    ) -> Result<usize> {
        const BATCH: usize = 32;
        let mut total_read = 0;

        // First, drain any buffered output
        if self.output_buffer.available_data() > 0 {
            let copied = self.output_buffer.copy_to(buf);
            self.output_buffer.consume(copied);
            total_read += copied;

            if total_read >= buf.len() {
                return Ok(total_read);
            }
        }

        while total_read < buf.len() {
            // Fill input buffer with codewords
            while self.input_buffer.available_data() < 255 * BATCH {
                let mut temp_buf = [0u8; 4096];
                match self.inner.read(&mut temp_buf) {
                    Ok(0) => break, // EOF
                    Ok(n) => self.input_buffer.append(&temp_buf[..n]),
                    Err(e) => return Err(e),
                }
            }

            // Extract complete codewords from buffer
            let (batch_count, mut codeword_batch) = self.input_buffer.extract_codewords::<BATCH>();

            if batch_count == 0 {
                break; // No complete codewords available
            }

            // Process batch with SIMD
            let syndrome_results =
                unsafe { calculate_syndrome_batch_avx2::<BATCH, PARITY_LEN>(&codeword_batch) };

            // Process each codeword based on syndrome results
            for i in 0..batch_count {
                if !syndrome_results[i] && self.validate_rs {
                    // Errors detected - use scalar correction
                    let corrected = match PARITY_LEN {
                        16 => code_255_239::decode(&mut codeword_batch[i]),
                        32 => code_255_223::decode(&mut codeword_batch[i]),
                        64 => code_255_191::decode(&mut codeword_batch[i]),
                        _ => return Err(error_invalid_data("Invalid parity length")),
                    }
                    .map_err(|_| {
                        error_invalid_data("error correction couldn't correct a faulty block")
                    })?;

                    if corrected {
                        self.stats.blocks_corrected += 1;
                    }
                }

                // Add decoded data to output buffer
                self.output_buffer.append(&codeword_batch[i][..DATA_LEN]);
                self.stats.blocks_processed += 1;
            }

            // Copy from output buffer to user buffer
            let available = self
                .output_buffer
                .available_data()
                .min(buf.len() - total_read);
            let copied = self
                .output_buffer
                .copy_to(&mut buf[total_read..total_read + available]);
            self.output_buffer.consume(copied);
            total_read += copied;
        }

        Ok(total_read)
    }

    fn decode_and_read_data(&mut self, buf: &mut [u8]) -> Result<usize> {
        if let Some(simd_fn) = self.decode_fn_simd {
            simd_fn(self, buf)
        } else {
            (self.decode_fn)(self, buf)
        }
    }

    /// Get the inner reader.
    pub(crate) fn into_inner(self) -> R {
        self.inner
    }
}

/// Transpose codewords for SIMD processing
#[cfg(target_arch = "x86_64")]
fn transpose_for_simd<const BATCH: usize>(codewords: &[[u8; 255]; BATCH]) -> [[u8; BATCH]; 255] {
    let mut transposed = [[0u8; BATCH]; 255];

    for (codeword_idx, codeword) in codewords.iter().enumerate() {
        for (byte_idx, &byte) in codeword.iter().enumerate() {
            transposed[byte_idx][codeword_idx] = byte;
        }
    }

    transposed
}

/// SIMD Reed-Solomon syndrome calculation for batch of 64 codewords using AVX512+GFNI
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,gfni")]
unsafe fn calculate_syndrome_batch_avx512<const BATCH: usize, const PARITY_LEN: usize>(
    batch_codewords: &[[u8; 255]; BATCH],
) -> [bool; BATCH] {
    use core::arch::x86_64::*;

    // Transpose data for SIMD processing
    let transposed = transpose_for_simd::<BATCH>(batch_codewords);

    // Calculate syndromes in parallel for all codewords
    let mut syndromes = [[0u8; BATCH]; PARITY_LEN];

    for syndrome_idx in 0..PARITY_LEN {
        let alpha_power = gf_alpha_pow((syndrome_idx + 1) as isize);
        let alpha_vec = _mm512_set1_epi8(alpha_power as i8);

        // Initialize syndrome with highest coefficient
        let mut syndrome_vec =
            unsafe { _mm512_loadu_si512(transposed[254].as_ptr() as *const __m512i) };

        // Horner's method: process each coefficient from high to low
        for byte_idx in (0..254).rev() {
            let data_vec =
                unsafe { _mm512_loadu_si512(transposed[byte_idx].as_ptr() as *const __m512i) };

            // syndrome = syndrome * α + coeff[byte_idx]
            syndrome_vec = _mm512_gf2p8mul_epi8(syndrome_vec, alpha_vec);
            syndrome_vec = _mm512_xor_si512(syndrome_vec, data_vec);
        }

        // Store syndrome results
        unsafe {
            _mm512_storeu_si512(
                syndromes[syndrome_idx].as_mut_ptr() as *mut __m512i,
                syndrome_vec,
            )
        };
    }

    // Check if syndromes are all zero (no errors)
    let mut no_errors = [true; BATCH];
    for i in 0..BATCH.min(batch_codewords.len()) {
        for syndrome_idx in 0..PARITY_LEN {
            if syndromes[syndrome_idx][i] != 0 {
                no_errors[i] = false;
                break;
            }
        }
    }

    no_errors
}

/// SIMD Reed-Solomon syndrome calculation for batch of 32 codewords using AVX2+GFNI
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,gfni")]
unsafe fn calculate_syndrome_batch_avx2<const BATCH: usize, const PARITY_LEN: usize>(
    batch_codewords: &[[u8; 255]; BATCH],
) -> [bool; BATCH] {
    use core::arch::x86_64::*;

    // Transpose data for SIMD processing
    let transposed = transpose_for_simd::<BATCH>(batch_codewords);

    // Calculate syndromes in parallel for all codewords
    let mut syndromes = [[0u8; BATCH]; PARITY_LEN];

    for syndrome_idx in 0..PARITY_LEN {
        let alpha_power = gf_alpha_pow((syndrome_idx + 1) as isize);
        let alpha_vec = _mm256_set1_epi8(alpha_power as i8);

        // Initialize syndrome with highest coefficient
        let mut syndrome_vec =
            unsafe { _mm256_loadu_si256(transposed[254].as_ptr() as *const __m256i) };

        // Horner's method: process each coefficient from high to low
        for byte_idx in (0..254).rev() {
            let data_vec =
                unsafe { _mm256_loadu_si256(transposed[byte_idx].as_ptr() as *const __m256i) };

            // syndrome = syndrome * α + coeff[byte_idx]
            syndrome_vec = _mm256_gf2p8mul_epi8(syndrome_vec, alpha_vec);
            syndrome_vec = _mm256_xor_si256(syndrome_vec, data_vec);
        }

        // Store syndrome results
        unsafe {
            _mm256_storeu_si256(
                syndromes[syndrome_idx].as_mut_ptr() as *mut __m256i,
                syndrome_vec,
            )
        };
    }

    // Check if syndromes are all zero (no errors)
    let mut no_errors = [true; BATCH];
    for i in 0..BATCH.min(batch_codewords.len()) {
        for syndrome_idx in 0..PARITY_LEN {
            if syndromes[syndrome_idx][i] != 0 {
                no_errors[i] = false;
                break;
            }
        }
    }

    no_errors
}

impl<R: Read> Read for ECCDecoder<R> {
    fn read(&mut self, buf: &mut [u8]) -> Result<usize> {
        if !self.uses_buffer {
            return self.inner.read(buf);
        }

        self.decode_and_read_data(buf)
    }
}
