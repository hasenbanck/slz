//! Stack only Reed–Solomon implementation.
//!
//! ## Specification
//!
//! - Field: GF(2^8) = GF(256)
//! - Primitive polynomial: x^8 + x^4 + x^3 + x + 1
//! - Generator: α = 3
//! - Codes:
//!     - (n=255, k=239, t=8)
//!     - (n=255, k=223, t=16)
//!     - (n=255, k=191, t=32)
//!     - (n=64, k=40, t=12)
//!     - (n=32, k=10, t=11)
//!
//! ## License
//!
//! The code in this file is in the public domain or can be licensed under the Apache 2 License.

/// Building blocks for all codes.
mod primitives {
    use crate::error_invalid_data;

    // GF(256) parameters
    const PRIMITIVE_POLY: u16 = 0x11B; // (x^8 + x^4 + x^3 + x + 1)
    const GF_EXP_LEN: usize = 512;
    const GF_LOG_LEN: usize = 256;

    pub(crate) static GF_TABLES: GfTables = build_gf_tables();

    pub(crate) struct GfTables {
        pub(crate) exp: [u8; GF_EXP_LEN],
        pub(crate) log: [u8; GF_LOG_LEN],
    }

    const fn build_gf_tables() -> GfTables {
        let mut exp = [0u8; GF_EXP_LEN];
        let mut log = [0u8; GF_LOG_LEN];

        // α = 3, build α^0..α^254
        let mut i = 0usize;
        let mut x: u16 = 1;
        while i < 255 {
            exp[i] = (x & 0xFF) as u8;
            log[(x & 0xFF) as usize] = i as u8;
            let x_times_2 = x << 1;
            if (x_times_2 & 0x100) != 0 {
                x = (x_times_2 ^ PRIMITIVE_POLY) ^ x;
            } else {
                x = x_times_2 ^ x;
            }
            i += 1;
        }

        // Duplicate exp for simple indexing without modulo.
        let mut j = 255usize;
        while j < GF_EXP_LEN {
            exp[j] = exp[j - 255];
            j += 1;
        }

        // Ensure log[0] = 0 (we won't use it for multiplicative ops).
        log[0] = 0;

        GfTables { exp, log }
    }

    #[inline]
    const fn gf_mul(a: u8, b: u8) -> u8 {
        if a == 0 || b == 0 {
            0
        } else {
            let idx = GF_TABLES.log[a as usize] as usize + GF_TABLES.log[b as usize] as usize;
            GF_TABLES.exp[idx]
        }
    }

    #[inline]
    const fn gf_div(a: u8, b: u8) -> u8 {
        debug_assert!(b != 0);
        if a == 0 {
            0
        } else {
            let la = GF_TABLES.log[a as usize] as usize;
            let lb = GF_TABLES.log[b as usize] as usize;
            let idx = la + 255 - lb;
            GF_TABLES.exp[idx]
        }
    }

    #[inline]
    const fn gf_alpha_pow(power: isize) -> u8 {
        let mut p = power % 255;
        if p < 0 {
            p += 255;
        }
        GF_TABLES.exp[p as usize]
    }

    #[inline]
    const fn gf_pow_primitive_const(power: usize) -> u8 {
        // const-friendly primitive power (α^power)
        // power % 255 and then index exp
        let idx = power % 255;
        GF_TABLES.exp[idx]
    }

    /// Multiply polynomial `g` by (x - root) where root = α^i.
    const fn mul_by_x_minus_root_const<const PARITY_LEN_PLUS_ONE: usize>(
        g: [u8; PARITY_LEN_PLUS_ONE],
        g_len: usize,
        root: u8,
    ) -> ([u8; PARITY_LEN_PLUS_ONE], usize) {
        let mut res = [0u8; PARITY_LEN_PLUS_ONE];

        // res = g * (x + root) = g*x + g*root
        let mut i = 0;
        while i < g_len {
            // Contribution from g*x (g[i] becomes coefficient of x^(i+1)).
            res[i + 1] ^= g[i];
            // Contribution from g*root (g[i]*root is coefficient of x^i).
            res[i] ^= gf_mul(g[i], root);
            i += 1;
        }

        let mut new_len = g_len + 1;
        if new_len > PARITY_LEN_PLUS_ONE {
            new_len = PARITY_LEN_PLUS_ONE;
        }
        while new_len > 1 && res[new_len - 1] == 0 {
            new_len -= 1;
        }
        (res, new_len)
    }

    pub(crate) const fn gen_poly_const<
        const PARITY_LEN: usize,
        const PARITY_LEN_PLUS_ONE: usize,
    >() -> [u8; PARITY_LEN_PLUS_ONE] {
        let mut g = [0u8; PARITY_LEN_PLUS_ONE];
        g[0] = 1;
        let mut g_len = 1usize;
        let mut i = 1usize;
        while i <= PARITY_LEN {
            let root = gf_pow_primitive_const(i);
            let (next, next_len) = mul_by_x_minus_root_const(g, g_len, root);
            g = next;
            g_len = next_len;
            i += 1;
        }

        g
    }

    #[inline]
    fn poly_eval(poly: &[u8], len: usize, x: u8) -> u8 {
        if len == 0 {
            return 0;
        }
        let mut acc = poly[len - 1];
        for i in (0..len - 1).rev() {
            acc = gf_mul(acc, x) ^ poly[i];
        }
        acc
    }

    /// Multiply a (len_a) and b (len_b) into `out`, set out_len.
    /// out capacity must be >= len_a + len_b - 1 (we use MAX_POLY).
    fn poly_mul_into(
        a: &[u8],
        len_a: usize,
        b: &[u8],
        len_b: usize,
        out: &mut [u8],
        out_len: &mut usize,
    ) {
        if len_a == 0 || len_b == 0 {
            out[0] = 0;
            *out_len = 1;
            return;
        }
        let res_len = len_a + len_b - 1;

        out[..res_len].fill(0);

        for i in 0..len_a {
            let ai = a[i];
            if ai == 0 {
                continue;
            }
            for j in 0..len_b {
                let bj = b[j];
                if bj == 0 {
                    continue;
                }
                out[i + j] ^= gf_mul(ai, bj);
            }
        }
        // trim
        let mut l = res_len;
        while l > 1 && out[l - 1] == 0 {
            l -= 1;
        }
        *out_len = l;
    }

    fn derivative_char2(poly: &[u8], len: usize, out: &mut [u8], out_len: &mut usize) {
        if len <= 1 {
            out[0] = 0;
            *out_len = 1;
            return;
        }
        let mut l = len - 1;
        for (i, out_item) in out.iter_mut().enumerate().take(l) {
            let idx = i + 1;
            if (idx & 1) == 1 {
                *out_item = poly[idx];
            } else {
                *out_item = 0;
            }
        }
        while l > 1 && out[l - 1] == 0 {
            l -= 1;
        }
        *out_len = l;
    }

    /// Finds the error locator polynomial `Λ(x)` for a given syndrome sequence `S(x)`.
    ///
    /// This is an optimized implementation of the Berlekamp-Massey algorithm.
    ///
    /// # Arguments
    /// * `synd`: The syndrome polynomial coefficients `[s_0, s_1, ..., s_{2t-1}]`.
    /// * `lambda_out`: An output buffer to store the resulting locator polynomial `Λ(x)`.
    /// * `lambda_len`: An output parameter to store the length of `Λ(x)`.
    fn berlekamp_massey<const PARITY_LEN_PLUS_ONE: usize>(
        synd: &[u8],
        lambda_out: &mut [u8],
        lambda_len: &mut usize,
    ) {
        // Λ(x), the error locator polynomial. Initialized to 1.
        let mut lambda = [0u8; PARITY_LEN_PLUS_ONE];
        lambda[0] = 1;
        let mut current_lambda_len = 1;

        // B(x), the previous Λ(x) from the last time `l` was updated. Initialized to 1.
        let mut b_poly = [0u8; PARITY_LEN_PLUS_ONE];
        b_poly[0] = 1;
        let mut b_len = 1;

        // `l` is the degree of the current locator polynomial (number of known errors).
        let mut l = 0;
        // `m` is the number of steps since `l` was last updated.
        let mut m = 1;
        // `d_prime` is the discrepancy from the last time `l` was updated.
        let mut d_prime = 1;

        // Iterate through all syndromes. `n` is the current time step.
        for n in 0..synd.len() {
            // 1. Calculate the discrepancy `d`.
            // This is the error in the next term of the sequence predicted by the current LFSR (lambda).
            // d = s_n + λ_1*s_{n-1} + ... + λ_l*s_{n-l}
            let mut d = synd[n];
            for i in 1..=l {
                if lambda[i] != 0 {
                    d ^= gf_mul(lambda[i], synd[n - i]);
                }
            }

            if d == 0 {
                // 2a. Discrepancy is zero. Our current `lambda` is still correct.
                // We just increase the shift `m`.
                m += 1;
            } else {
                // 2b. Discrepancy is non-zero. We must update `lambda`.
                let t_poly = lambda; // Save a copy of the current lambda(x).
                let t_len = current_lambda_len;

                let scale_factor = gf_div(d, d_prime);

                // Update lambda: Λ(x) = Λ(x) - (d/d') * x^m * B(x)
                let required_len = b_len + m;
                if current_lambda_len < required_len {
                    current_lambda_len = required_len;
                }
                for i in 0..b_len {
                    if b_poly[i] != 0 {
                        lambda[i + m] ^= gf_mul(scale_factor, b_poly[i]);
                    }
                }

                // Check if we need to update the LFSR length `l`.
                if 2 * l <= n {
                    l = n + 1 - l;
                    d_prime = d;
                    b_poly = t_poly; // The old lambda becomes the new B(x).
                    b_len = t_len; // Use the length of the old lambda.
                    m = 1;
                } else {
                    // The length `l` doesn't change.
                    m += 1;
                }
            }
        }

        // The final length of the locator polynomial is `l + 1`.
        *lambda_len = l + 1;
        lambda_out[..*lambda_len].copy_from_slice(&lambda[..*lambda_len]);

        // Trim any trailing zeros that might have been introduced, although `l+1` should be correct.
        while *lambda_len > 1 && lambda_out[*lambda_len - 1] == 0 {
            *lambda_len -= 1;
        }
    }

    fn calculate_syndromes<const CODEWORD_SIZE: usize, const PARITY_LEN: usize>(
        c: &mut [u8; CODEWORD_SIZE],
        syndrome: &mut [u8; PARITY_LEN],
    ) -> bool {
        let mut all_zero = true;

        let mut x = 1u8;
        let alpha = gf_alpha_pow(1); // which is 2
        for syndrome_item in syndrome.iter_mut().take(PARITY_LEN) {
            x = gf_mul(x, alpha); // x = α^(i+1)
            let s = poly_eval(c, CODEWORD_SIZE, x);
            *syndrome_item = s;
            if s != 0 {
                all_zero = false;
            }
        }

        all_zero
    }

    /// Calculates the error evaluator polynomial, Omega(x).
    /// Omega(x) = (Syndromes(x) * Lambda(x)) mod x^{PARITY_LEN}
    fn calculate_error_evaluator_poly<
        const PARITY_LEN: usize,
        const PARITY_LEN_PLUS_ONE: usize,
        const MAX_POLY: usize,
    >(
        syndromes: &[u8; PARITY_LEN],
        lambda: &[u8; PARITY_LEN_PLUS_ONE],
        lambda_len: usize,
    ) -> ([u8; PARITY_LEN], usize) {
        let mut prod = [0u8; MAX_POLY];
        let mut prod_len = 0;

        // Multiply S(x) and Λ(x)
        poly_mul_into(
            syndromes,
            PARITY_LEN,
            lambda,
            lambda_len,
            &mut prod,
            &mut prod_len,
        );

        // The result is truncated to the degree of PARITY_LEN - 1.
        // Equivalent to `mod x^{PARITY_LEN}`.
        let omega_len = prod_len.min(PARITY_LEN);
        let mut omega = [0u8; PARITY_LEN];
        omega[..omega_len].copy_from_slice(&prod[..omega_len]);

        (omega, omega_len)
    }

    /// Calculates the error magnitudes using Forney's algorithm.
    /// Magnitude_j = - Omega(X_j⁻¹) / Lambda'(X_j⁻¹)
    fn calculate_error_magnitudes<
        const PARITY_LEN: usize,
        const PARITY_LEN_PLUS_ONE: usize,
        const HALF_PARITY_LEN: usize,
    >(
        err_pos: &[usize; HALF_PARITY_LEN],
        err_count: usize,
        omega: &[u8; PARITY_LEN],
        omega_len: usize,
        lambda: &[u8; PARITY_LEN_PLUS_ONE],
        lambda_len: usize,
    ) -> crate::Result<[u8; HALF_PARITY_LEN]> {
        // We need the formal derivative of Lambda, Λ'(x).
        let mut lambda_deriv = [0u8; PARITY_LEN_PLUS_ONE];
        let mut lambda_deriv_len = 0;
        derivative_char2(lambda, lambda_len, &mut lambda_deriv, &mut lambda_deriv_len);

        let mut magnitudes = [0u8; HALF_PARITY_LEN];

        for i in 0..err_count {
            let pos = err_pos[i];
            let x_inv = gf_alpha_pow(-(pos as isize));

            // Evaluate Omega at the inverse error location.
            let omega_val = poly_eval(omega, omega_len, x_inv);

            // Evaluate the derivative of Lambda at the inverse error location.
            let denom = poly_eval(&lambda_deriv, lambda_deriv_len, x_inv);
            if denom == 0 {
                // This indicates a decoder failure, possibly due to too many errors.
                return Err(error_invalid_data(
                    "zero derivative at error position (cannot invert)",
                ));
            }

            // In GF(2^n), negation is the identity op, so -A = A.
            magnitudes[i] = gf_div(omega_val, denom);
        }

        Ok(magnitudes)
    }

    fn find_error_locations<
        const PARITY_LEN: usize,
        const PARITY_LEN_PLUS_ONE: usize,
        const HALF_PARITY_LEN: usize,
        const CODEWORD_SIZE: usize,
    >(
        lambda: &[u8; PARITY_LEN_PLUS_ONE],
        lambda_len: usize,
        num_errors: usize,
    ) -> crate::Result<([usize; HALF_PARITY_LEN], usize)> {
        let mut err_pos = [0usize; HALF_PARITY_LEN];
        let mut err_count = 0usize;

        for pos in 0..CODEWORD_SIZE {
            let x_inv = gf_alpha_pow(-(pos as isize));
            let val = poly_eval(lambda, lambda_len, x_inv);
            if val == 0 {
                if err_count < PARITY_LEN {
                    err_pos[err_count] = pos;
                    err_count += 1;
                } else {
                    return Err(error_invalid_data("too many roots found"));
                }
            }
        }

        if err_count == 0 {
            return Err(error_invalid_data("no error positions found"));
        }

        if err_count != num_errors {
            return Err(error_invalid_data(
                "mismatch between locator degree and roots found",
            ));
        }

        if err_count > PARITY_LEN / 2 {
            return Err(error_invalid_data("too many errors to correct"));
        }

        Ok((err_pos, err_count))
    }

    fn find_error_locator_poly<const PARITY_LEN: usize, const PARITY_LEN_PLUS_ONE: usize>(
        syndromes: &mut [u8; PARITY_LEN],
    ) -> crate::Result<([u8; PARITY_LEN_PLUS_ONE], usize, usize)> {
        let mut lambda = [0u8; PARITY_LEN_PLUS_ONE];
        let mut lambda_len = 0;
        berlekamp_massey::<PARITY_LEN_PLUS_ONE>(syndromes, &mut lambda, &mut lambda_len);

        if lambda_len == 0 {
            return Err(error_invalid_data("no error locator found"));
        }

        let num_errors = lambda_len - 1;

        if num_errors == 0 {
            return Err(error_invalid_data("no errors located"));
        }

        if num_errors > PARITY_LEN {
            return Err(error_invalid_data(
                "too many errors (locator degree > parity)",
            ));
        }

        Ok((lambda, lambda_len, num_errors))
    }

    /// Verifies that the corrected codeword is valid by checking if all syndromes are zero.
    fn verify_correction<const PARITY_LEN: usize, const CODEWORD_SIZE: usize>(
        codeword_poly: &[u8; CODEWORD_SIZE],
    ) -> bool {
        for i in 0..PARITY_LEN {
            let x = gf_alpha_pow((i + 1) as isize);
            if poly_eval(codeword_poly, CODEWORD_SIZE, x) != 0 {
                // A non-zero syndrome means correction failed.
                return false;
            }
        }
        // All syndromes are zero.
        true
    }

    /// This implementation uses a LFSR-based method.
    #[inline(always)]
    pub(crate) fn encode<
        const DATA_LEN: usize,
        const PARITY_LEN: usize,
        const PARITY_LEN_PLUS_ONE: usize,
    >(
        g: &'static [u8; PARITY_LEN_PLUS_ONE],
        data: &[u8; DATA_LEN],
    ) -> [u8; PARITY_LEN] {
        // The generator polynomial g(x) has degree PARITY_LEN. g_len is PARITY_LEN_PLUS_ONE.
        // We only need the coefficients g_0, g_1, ..., g_{PARITY_LEN-1}.
        // The leading coefficient g_{PARITY_LEN} is 1 and is implicit.
        let mut remainder = [0u8; PARITY_LEN];

        // Process data from the highest degree to lowest.
        for &byte in data.iter().rev() {
            let feedback = byte ^ remainder[PARITY_LEN - 1];
            remainder.rotate_right(1);
            remainder[0] = 0;

            // If feedback is non-zero, subtract feedback_term * g(x)
            if feedback != 0 {
                for (r, &gcoef) in remainder.iter_mut().zip(&g[..PARITY_LEN]) {
                    *r ^= gf_mul(gcoef, feedback);
                }
            }
        }

        remainder
    }

    /// The decoder is a classic Peterson–Gorenstein–Zierler decoder.
    #[inline(always)]
    pub(crate) fn decode<
        const DATA_LEN: usize,
        const PARITY_LEN: usize,
        const PARITY_LEN_PLUS_ONE: usize,
        const HALF_PARITY_LEN: usize,
        const CODEWORD_SIZE: usize,
        const MAX_POLY: usize,
    >(
        codeword: &mut [u8; CODEWORD_SIZE],
    ) -> crate::Result<bool> {
        // The received codeword polynomial is C(x) = D(x) * x^32 + P(x).
        // Our arrays store coefficients from the lowest degree to highest, so we arrange it as:
        // c = [p_0, p_1, ..., p_31, d_0, d_1, ..., d_31]
        let mut c = [0u8; CODEWORD_SIZE];
        c[..PARITY_LEN].copy_from_slice(&codeword[DATA_LEN..]); // P(x) coeffs
        c[PARITY_LEN..].copy_from_slice(&codeword[..DATA_LEN]); // D(x) coeffs

        // Step 1: Calculate Syndromes.
        let mut syndromes = [0u8; PARITY_LEN];
        if calculate_syndromes(&mut c, &mut syndromes) {
            // No errors found, the data is already correct.
            return Ok(false);
        }

        // Step 2: Find error locator polynomial Λ(x).
        let (lambda, lambda_len, num_errors) =
            find_error_locator_poly::<PARITY_LEN, PARITY_LEN_PLUS_ONE>(&mut syndromes)?;

        // Step 3: Find error locations by finding the roots of Λ(x) (Chien Search).
        let (err_pos, err_count) = find_error_locations::<
            PARITY_LEN,
            PARITY_LEN_PLUS_ONE,
            HALF_PARITY_LEN,
            CODEWORD_SIZE,
        >(&lambda, lambda_len, num_errors)?;

        // Step 4: Find error evaluator polynomial Ω(x).
        let (omega, omega_len) =
            calculate_error_evaluator_poly::<PARITY_LEN, PARITY_LEN_PLUS_ONE, MAX_POLY>(
                &syndromes, &lambda, lambda_len,
            );

        // Step 5: Find the error magnitudes using Forney's algorithm.
        let err_magnitudes =
            calculate_error_magnitudes::<PARITY_LEN, PARITY_LEN_PLUS_ONE, HALF_PARITY_LEN>(
                &err_pos, err_count, &omega, omega_len, &lambda, lambda_len,
            )?;

        // Step 6: Correct the errors in the codeword polynomial.
        for i in 0..err_count {
            c[err_pos[i]] ^= err_magnitudes[i];
        }

        // Step 7: Verify that the correction was successful.
        if !verify_correction::<PARITY_LEN, CODEWORD_SIZE>(&c) {
            return Err(error_invalid_data(
                "correction failed; syndromes are still non-zero",
            ));
        }

        // Step 8: Write corrected data back into the original buffer.
        codeword[..DATA_LEN].copy_from_slice(&c[PARITY_LEN..]);

        Ok(true)
    }

    #[cfg(test)]
    mod gfni_compatibility_tests {
        use super::gf_mul;

        #[cfg(target_arch = "x86_64")]
        #[test]
        fn test_gfni_vs_table_multiplication() {
            if !is_x86_feature_detected!("gfni") || !is_x86_feature_detected!("sse4.1") {
                println!("GFNI not supported");
                return;
            }
            println!("GFNI supported");

            let test_cases = [
                (0x53, 0xCA), // Two random non-zero values
                (0x01, 0x42), // Multiplicative identity
                (0xFF, 0xFF), // Max values
                (0x02, 0x04), // Simple powers of 2
            ];

            for (a, b) in test_cases {
                let table_result = gf_mul(a, b);
                let gfni_result = gf_mul_gfni_single(a, b);

                if table_result != gfni_result {
                    println!("  MISMATCH: Different field representations!");
                }
            }
        }

        #[cfg(target_arch = "x86_64")]
        fn gf_mul_gfni_single(a: u8, b: u8) -> u8 {
            unsafe {
                use core::arch::x86_64::*;
                let a_vec = _mm_set1_epi8(a as i8);
                let b_vec = _mm_set1_epi8(b as i8);
                let result = _mm_gf2p8mul_epi8(a_vec, b_vec);
                _mm_extract_epi8::<0>(result) as u8
            }
        }
    }
}

/// Implements RS(255,239)
pub(crate) mod code_255_239 {
    /// The size of the data payload.
    const DATA_LEN: usize = 239;

    /// The size of the parity bytes.
    const PARITY_LEN: usize = 16;

    /// The size of the parity bytes plus one.
    const PARITY_LEN_PLUS_ONE: usize = PARITY_LEN + 1;

    /// The half size of the parity bytes.
    const HALF_PARITY_LEN: usize = PARITY_LEN / 2;

    /// Overall codeword size.
    const CODEWORD_SIZE: usize = DATA_LEN + PARITY_LEN;

    /// Safe upper bound for intermediate polynomials.
    const MAX_POLY: usize = CODEWORD_SIZE;

    pub(super) static GEN_POLY: [u8; PARITY_LEN_PLUS_ONE] =
        super::primitives::gen_poly_const::<PARITY_LEN, _>();

    /// Encode 10-byte data with RS(255,239) protection.
    pub(crate) fn encode(data: &[u8; DATA_LEN]) -> [u8; PARITY_LEN] {
        super::primitives::encode(&GEN_POLY, data)
    }

    /// Decode codeword in-place (data || parity) for RS(255,239).
    ///
    /// Returns false if the data was not corrupted. False if the data was corrected but could be
    /// corrected. Returns an error if the data was corrupted and could not be corrected.
    pub(crate) fn decode(codeword: &mut [u8; CODEWORD_SIZE]) -> crate::Result<bool> {
        super::primitives::decode::<
            DATA_LEN,
            PARITY_LEN,
            PARITY_LEN_PLUS_ONE,
            HALF_PARITY_LEN,
            CODEWORD_SIZE,
            MAX_POLY,
        >(codeword)
    }

    #[cfg(test)]
    mod tests {
        use hex_literal::hex;

        use super::*;

        fn test_vector(data: [u8; DATA_LEN], expected_parity: [u8; PARITY_LEN]) {
            let parity = encode(&data);

            assert_eq!(parity, expected_parity);

            let mut cw = [0u8; CODEWORD_SIZE];
            cw[..DATA_LEN].copy_from_slice(&data);
            cw[DATA_LEN..].copy_from_slice(&parity);

            decode(&mut cw).expect("should decode with no errors");

            assert_eq!(&cw[..DATA_LEN], &data);
        }

        #[test]
        fn test_rs_255_239_specification_test_vector_1() {
            let data = hex!(
                "0000000000000000000000000000000000000000
                 0000000000000000000000000000000000000000
                 0000000000000000000000000000000000000000
                 0000000000000000000000000000000000000000
                 0000000000000000000000000000000000000000
                 0000000000000000000000000000000000000000
                 0000000000000000000000000000000000000000
                 0000000000000000000000000000000000000000
                 0000000000000000000000000000000000000000
                 0000000000000000000000000000000000000000
                 0000000000000000000000000000000000000000
                 00000000000000000000000000000000000000"
            );
            let expected_parity = hex!("00000000000000000000000000000000");
            test_vector(data, expected_parity);
        }

        #[test]
        fn test_rs_255_239_specification_test_vector_2() {
            let data = hex!(
                "ffffffffffffffffffffffffffffffffffffffff
                 ffffffffffffffffffffffffffffffffffffffff
                 ffffffffffffffffffffffffffffffffffffffff
                 ffffffffffffffffffffffffffffffffffffffff
                 ffffffffffffffffffffffffffffffffffffffff
                 ffffffffffffffffffffffffffffffffffffffff
                 ffffffffffffffffffffffffffffffffffffffff
                 ffffffffffffffffffffffffffffffffffffffff
                 ffffffffffffffffffffffffffffffffffffffff
                 ffffffffffffffffffffffffffffffffffffffff
                 ffffffffffffffffffffffffffffffffffffffff
                 ffffffffffffffffffffffffffffffffffffff"
            );
            let expected_parity = hex!("ffffffffffffffffffffffffffffffff");
            test_vector(data, expected_parity);
        }

        #[test]
        fn test_rs_255_239_specification_test_vector_3() {
            let data = hex!(
                "000102030405060708090a0b0c0d0e0f10111213
                 1415161718191a1b1c1d1e1f2021222324252627
                 28292a2b2c2d2e2f303132333435363738393a3b
                 3c3d3e3f404142434445464748494a4b4c4d4e4f
                 505152535455565758595a5b5c5d5e5f60616263
                 6465666768696a6b6c6d6e6f7071727374757677
                 78797a7b7c7d7e7f808182838485868788898a8b
                 8c8d8e8f909192939495969798999a9b9c9d9e9f
                 a0a1a2a3a4a5a6a7a8a9aaabacadaeafb0b1b2b3
                 b4b5b6b7b8b9babbbcbdbebfc0c1c2c3c4c5c6c7
                 c8c9cacbcccdcecfd0d1d2d3d4d5d6d7d8d9dadb
                 dcdddedfe0e1e2e3e4e5e6e7e8e9eaebecedee"
            );
            let expected_parity = hex!("07ffcc5e9bfb1c0838aee03603b502aa");
            test_vector(data, expected_parity);
        }
    }
}

/// Implements RS(255,223)
pub(crate) mod code_255_223 {
    /// The size of the data payload.
    const DATA_LEN: usize = 223;

    /// The size of the parity bytes.
    const PARITY_LEN: usize = 32;

    /// The size of the parity bytes plus one.
    const PARITY_LEN_PLUS_ONE: usize = PARITY_LEN + 1;

    /// The half size of the parity bytes.
    const HALF_PARITY_LEN: usize = PARITY_LEN / 2;

    /// Overall codeword size.
    const CODEWORD_SIZE: usize = DATA_LEN + PARITY_LEN;

    /// Safe upper bound for intermediate polynomials.
    const MAX_POLY: usize = CODEWORD_SIZE;

    pub(super) static GEN_POLY: [u8; PARITY_LEN_PLUS_ONE] =
        super::primitives::gen_poly_const::<PARITY_LEN, _>();

    /// Encode 10-byte data with RS(255,223) protection.
    pub(crate) fn encode(data: &[u8; DATA_LEN]) -> [u8; PARITY_LEN] {
        super::primitives::encode(&GEN_POLY, data)
    }

    /// Decode codeword in-place (data || parity) for RS(255,223).
    ///
    /// Returns false if the data was not corrupted. False if the data was corrected but could be
    /// corrected. Returns an error if the data was corrupted and could not be corrected.
    pub(crate) fn decode(codeword: &mut [u8; CODEWORD_SIZE]) -> crate::Result<bool> {
        super::primitives::decode::<
            DATA_LEN,
            PARITY_LEN,
            PARITY_LEN_PLUS_ONE,
            HALF_PARITY_LEN,
            CODEWORD_SIZE,
            MAX_POLY,
        >(codeword)
    }

    #[cfg(test)]
    mod tests {
        use hex_literal::hex;

        use super::*;

        fn test_vector(data: [u8; DATA_LEN], expected_parity: [u8; PARITY_LEN]) {
            let parity = encode(&data);

            assert_eq!(parity, expected_parity);

            let mut cw = [0u8; CODEWORD_SIZE];
            cw[..DATA_LEN].copy_from_slice(&data);
            cw[DATA_LEN..].copy_from_slice(&parity);

            decode(&mut cw).expect("should decode with no errors");

            assert_eq!(&cw[..DATA_LEN], &data);
        }

        #[test]
        fn test_rs_255_223_specification_test_vector_1() {
            let data = hex!(
                "0000000000000000000000000000000000000000
                 0000000000000000000000000000000000000000
                 0000000000000000000000000000000000000000
                 0000000000000000000000000000000000000000
                 0000000000000000000000000000000000000000
                 0000000000000000000000000000000000000000
                 0000000000000000000000000000000000000000
                 0000000000000000000000000000000000000000
                 0000000000000000000000000000000000000000
                 0000000000000000000000000000000000000000
                 0000000000000000000000000000000000000000
                 000000"
            );
            let expected_parity = hex!(
                "0000000000000000000000000000000000000000
                 000000000000000000000000"
            );
            test_vector(data, expected_parity);
        }

        #[test]
        fn test_rs_255_223_specification_test_vector_2() {
            let data = hex!(
                "ffffffffffffffffffffffffffffffffffffffff
                 ffffffffffffffffffffffffffffffffffffffff
                 ffffffffffffffffffffffffffffffffffffffff
                 ffffffffffffffffffffffffffffffffffffffff
                 ffffffffffffffffffffffffffffffffffffffff
                 ffffffffffffffffffffffffffffffffffffffff
                 ffffffffffffffffffffffffffffffffffffffff
                 ffffffffffffffffffffffffffffffffffffffff
                 ffffffffffffffffffffffffffffffffffffffff
                 ffffffffffffffffffffffffffffffffffffffff
                 ffffffffffffffffffffffffffffffffffffffff
                 ffffff"
            );
            let expected_parity = hex!(
                "ffffffffffffffffffffffffffffffffffffffff
                 ffffffffffffffffffffffff"
            );
            test_vector(data, expected_parity);
        }

        #[test]
        fn test_rs_255_223_specification_test_vector_3() {
            let data = hex!(
                "000102030405060708090a0b0c0d0e0f10111213
                 1415161718191a1b1c1d1e1f2021222324252627
                 28292a2b2c2d2e2f303132333435363738393a3b
                 3c3d3e3f404142434445464748494a4b4c4d4e4f
                 505152535455565758595a5b5c5d5e5f60616263
                 6465666768696a6b6c6d6e6f7071727374757677
                 78797a7b7c7d7e7f808182838485868788898a8b
                 8c8d8e8f909192939495969798999a9b9c9d9e9f
                 a0a1a2a3a4a5a6a7a8a9aaabacadaeafb0b1b2b3
                 b4b5b6b7b8b9babbbcbdbebfc0c1c2c3c4c5c6c7
                 c8c9cacbcccdcecfd0d1d2d3d4d5d6d7d8d9dadb
                 dcddde"
            );
            let expected_parity = hex!(
                "93cca4cfe7c914d65c083eb57a634cd5a86f77ed
                 f97b87cf4c05be2478e175d4"
            );
            test_vector(data, expected_parity);
        }
    }
}

/// Implements RS(255,191)
pub(crate) mod code_255_191 {

    /// The size of the data payload.
    const DATA_LEN: usize = 191;

    /// The size of the parity bytes.
    const PARITY_LEN: usize = 64;

    /// The size of the parity bytes plus one.
    const PARITY_LEN_PLUS_ONE: usize = PARITY_LEN + 1;

    /// The half size of the parity bytes.
    const HALF_PARITY_LEN: usize = PARITY_LEN / 2;

    /// Overall codeword size.
    const CODEWORD_SIZE: usize = DATA_LEN + PARITY_LEN;

    /// Safe upper bound for intermediate polynomials.
    const MAX_POLY: usize = CODEWORD_SIZE;

    pub(super) static GEN_POLY: [u8; PARITY_LEN_PLUS_ONE] =
        super::primitives::gen_poly_const::<PARITY_LEN, _>();

    /// Encode 10-byte data with RS(255,191) protection.
    pub(crate) fn encode(data: &[u8; DATA_LEN]) -> [u8; PARITY_LEN] {
        super::primitives::encode(&GEN_POLY, data)
    }

    /// Decode codeword in-place (data || parity) for RS(255,191).
    ///
    /// Returns false if the data was not corrupted. False if the data was corrected but could be
    /// corrected. Returns an error if the data was corrupted and could not be corrected.
    pub(crate) fn decode(codeword: &mut [u8; CODEWORD_SIZE]) -> crate::Result<bool> {
        super::primitives::decode::<
            DATA_LEN,
            PARITY_LEN,
            PARITY_LEN_PLUS_ONE,
            HALF_PARITY_LEN,
            CODEWORD_SIZE,
            MAX_POLY,
        >(codeword)
    }

    #[cfg(test)]
    mod tests {
        use hex_literal::hex;

        use super::*;

        fn test_vector(data: [u8; DATA_LEN], expected_parity: [u8; PARITY_LEN]) {
            let parity = encode(&data);

            assert_eq!(parity, expected_parity);

            let mut cw = [0u8; CODEWORD_SIZE];
            cw[..DATA_LEN].copy_from_slice(&data);
            cw[DATA_LEN..].copy_from_slice(&parity);

            decode(&mut cw).expect("should decode with no errors");

            assert_eq!(&cw[..DATA_LEN], &data);
        }

        #[test]
        fn test_rs_255_191_specification_test_vector_1() {
            let data = hex!(
                "0000000000000000000000000000000000000000
                 0000000000000000000000000000000000000000
                 0000000000000000000000000000000000000000
                 0000000000000000000000000000000000000000
                 0000000000000000000000000000000000000000
                 0000000000000000000000000000000000000000
                 0000000000000000000000000000000000000000
                 0000000000000000000000000000000000000000
                 0000000000000000000000000000000000000000
                 0000000000000000000000"
            );
            let expected_parity = hex!(
                "0000000000000000000000000000000000000000
                 0000000000000000000000000000000000000000
                 0000000000000000000000000000000000000000
                 00000000"
            );
            test_vector(data, expected_parity);
        }

        #[test]
        fn test_rs_255_191_specification_test_vector_2() {
            let data = hex!(
                "ffffffffffffffffffffffffffffffffffffffff
                 ffffffffffffffffffffffffffffffffffffffff
                 ffffffffffffffffffffffffffffffffffffffff
                 ffffffffffffffffffffffffffffffffffffffff
                 ffffffffffffffffffffffffffffffffffffffff
                 ffffffffffffffffffffffffffffffffffffffff
                 ffffffffffffffffffffffffffffffffffffffff
                 ffffffffffffffffffffffffffffffffffffffff
                 ffffffffffffffffffffffffffffffffffffffff
                 ffffffffffffffffffffff"
            );
            let expected_parity = hex!(
                "ffffffffffffffffffffffffffffffffffffffff
                 ffffffffffffffffffffffffffffffffffffffff
                 ffffffffffffffffffffffffffffffffffffffff
                 ffffffff"
            );
            test_vector(data, expected_parity);
        }

        #[test]
        fn test_rs_255_191_specification_test_vector_3() {
            let data = hex!(
                "000102030405060708090a0b0c0d0e0f10111213
                 1415161718191a1b1c1d1e1f2021222324252627
                 28292a2b2c2d2e2f303132333435363738393a3b
                 3c3d3e3f404142434445464748494a4b4c4d4e4f
                 505152535455565758595a5b5c5d5e5f60616263
                 6465666768696a6b6c6d6e6f7071727374757677
                 78797a7b7c7d7e7f808182838485868788898a8b
                 8c8d8e8f909192939495969798999a9b9c9d9e9f
                 a0a1a2a3a4a5a6a7a8a9aaabacadaeafb0b1b2b3
                 b4b5b6b7b8b9babbbcbdbe"
            );
            let expected_parity = hex!(
                "792316db851127a71e19d44e5fe58400bffdc5be
                 5a73b3b90f1b660ea25f08bfced98819758eabc2
                 586966bee7b5abec7387eea89e0377f623340cf0
                 6209b500"
            );
            test_vector(data, expected_parity);
        }
    }
}

/// Implements RS(64,40)
pub(crate) mod code_64_40 {

    /// The size of the data payload.
    const DATA_LEN: usize = 40;

    /// The size of the parity bytes.
    const PARITY_LEN: usize = 24;

    /// The size of the parity bytes plus one.
    const PARITY_LEN_PLUS_ONE: usize = PARITY_LEN + 1;

    /// The half size of the parity bytes.
    const HALF_PARITY_LEN: usize = PARITY_LEN / 2;

    /// Overall codeword size.
    const CODEWORD_SIZE: usize = DATA_LEN + PARITY_LEN;

    /// Safe upper bound for intermediate polynomials.
    const MAX_POLY: usize = CODEWORD_SIZE;

    static GEN_POLY: [u8; PARITY_LEN_PLUS_ONE] =
        super::primitives::gen_poly_const::<PARITY_LEN, _>();

    /// Encode 40-byte data with RS(64,40) protection.
    pub(crate) fn encode(data: &[u8; DATA_LEN]) -> [u8; PARITY_LEN] {
        super::primitives::encode(&GEN_POLY, data)
    }

    /// Decode codeword in-place (data || parity) for RS(64,40).
    ///
    /// Returns false if the data was not corrupted. False if the data was corrected but could be
    /// corrected. Returns an error if the data was corrupted and could not be corrected.
    pub(crate) fn decode(codeword: &mut [u8; CODEWORD_SIZE]) -> crate::Result<bool> {
        super::primitives::decode::<
            DATA_LEN,
            PARITY_LEN,
            PARITY_LEN_PLUS_ONE,
            HALF_PARITY_LEN,
            CODEWORD_SIZE,
            MAX_POLY,
        >(codeword)
    }

    #[cfg(test)]
    mod tests {
        use hex_literal::hex;

        use super::*;
        use crate::tests::Lcg;

        #[test]
        fn test_encode_decode_no_errors() {
            let mut data = [0u8; DATA_LEN];

            for (i, data_item) in data.iter_mut().enumerate().take(DATA_LEN) {
                *data_item = i as u8;
            }

            let parity = encode(&data);

            let mut cw = [0u8; CODEWORD_SIZE];
            cw[..DATA_LEN].copy_from_slice(&data);
            cw[DATA_LEN..].copy_from_slice(&parity);

            decode(&mut cw).expect("should decode with no errors");

            assert_eq!(cw[..DATA_LEN], data);
        }

        #[test]
        fn test_random_correctable_errors() {
            // TODO: 0x12345678 let's this test go into an endless loop! This was already present in the old version! (it only manifests when I now use the new Lcg)
            let mut rng = Lcg::new(0x0123456789ABCDEF);
            for _ in 0..20 {
                let mut data = [0u8; DATA_LEN];

                for data_item in data.iter_mut().take(DATA_LEN) {
                    *data_item = rng.next_u8();
                }

                let parity = encode(&data);

                let mut cw = [0u8; CODEWORD_SIZE];
                cw[..DATA_LEN].copy_from_slice(&data);
                cw[DATA_LEN..].copy_from_slice(&parity);

                // Introduce up to 12 unique errors (RS(64,40) can correct up to 12 errors)
                let errors = 1 + (rng.next_usize(12));
                let mut positions = [usize::MAX; PARITY_LEN];
                let mut count = 0usize;
                while count < errors {
                    let p = rng.next_usize(CODEWORD_SIZE);
                    let mut unique = true;
                    for position_item in positions.iter().take(count) {
                        if *position_item == p {
                            unique = false;
                            break;
                        }
                    }
                    if unique {
                        positions[count] = p;
                        count += 1;
                    }
                }

                for position_item in positions.iter().take(count) {
                    let p = *position_item;
                    let mut v = rng.next_u8();
                    if v == 0 {
                        v = 1;
                    }
                    cw[p] ^= v;
                }

                decode(&mut cw).expect("should decode correctable errors");

                assert_eq!(cw[..DATA_LEN], data);
            }
        }

        #[test]
        fn test_too_many_errors_fails() {
            let mut rng = Lcg::new(0xDEADBEEF);

            let mut data = [0u8; DATA_LEN];

            for data_item in data.iter_mut().take(DATA_LEN) {
                *data_item = rng.next_u8();
            }

            let parity = encode(&data);

            let mut cw = [0u8; CODEWORD_SIZE];
            cw[..DATA_LEN].copy_from_slice(&data);
            cw[DATA_LEN..].copy_from_slice(&parity);

            // Introduce >16 errors.
            let mut positions = [usize::MAX; PARITY_LEN];
            let mut count = 0usize;
            while count < (PARITY_LEN / 2 + 1) {
                let p = rng.next_usize(CODEWORD_SIZE);
                let mut unique = true;
                for position_item in positions.iter().take(count) {
                    if *position_item == p {
                        unique = false;
                        break;
                    }
                }
                if unique {
                    positions[count] = p;
                    count += 1;
                }
            }

            for position_item in positions.iter().take(count) {
                let p = *position_item;
                let mut v = rng.next_u8();
                if v == 0 {
                    v = 1;
                }
                cw[p] ^= v;
            }

            let res = decode(&mut cw);

            assert!(res.is_err(), "decoding should fail with too many errors");
        }

        fn test_vector(data: [u8; 40], expected_parity: [u8; 24]) {
            let parity = encode(&data);

            assert_eq!(parity, expected_parity);

            let mut cw = [0u8; CODEWORD_SIZE];
            cw[..DATA_LEN].copy_from_slice(&data);
            cw[DATA_LEN..].copy_from_slice(&parity);

            decode(&mut cw).expect("should decode with no errors");

            assert_eq!(cw[..DATA_LEN], data);
        }

        #[test]
        fn test_specification_test_vector_1() {
            let data = hex!(
                "0000000000000000000000000000000000000000
                 0000000000000000000000000000000000000000"
            );
            let expected_parity = hex!(
                "0000000000000000000000000000000000000000
                 00000000"
            );
            test_vector(data, expected_parity);
            test_vector(data, expected_parity);
        }

        #[test]
        fn test_specification_test_vector_2() {
            let data = hex!(
                "ffffffffffffffffffffffffffffffffffffffff
                 ffffffffffffffffffffffffffffffffffffffff"
            );
            let expected_parity = hex!(
                "579a5af18d3b67e5bfec98bb598dc2b4a5a7714d
                 dc267cd9"
            );
            test_vector(data, expected_parity);
        }

        #[test]
        fn test_specification_test_vector_3() {
            let data = hex!(
                "000102030405060708090a0b0c0d0e0f10111213
                 1415161718191a1b1c1d1e1f2021222324252627"
            );
            let expected_parity = hex!(
                "bb68ae9f872c2d5eb1c486a104d5d0d0e77140d3
                 1e2ae52b"
            );
            test_vector(data, expected_parity);
            test_vector(data, expected_parity);
        }
    }
}

/// Implements RS(32,10)
pub(crate) mod code_32_10 {

    /// The size of the data payload.
    const DATA_LEN: usize = 10;

    /// The size of the parity bytes.
    const PARITY_LEN: usize = 22;

    /// The size of the parity bytes plus one.
    const PARITY_LEN_PLUS_ONE: usize = PARITY_LEN + 1;

    /// The half size of the parity bytes.
    const HALF_PARITY_LEN: usize = PARITY_LEN / 2;

    /// Overall codeword size.
    const CODEWORD_SIZE: usize = DATA_LEN + PARITY_LEN;

    /// Safe upper bound for intermediate polynomials.
    const MAX_POLY: usize = CODEWORD_SIZE;

    static GEN_POLY: [u8; PARITY_LEN_PLUS_ONE] =
        super::primitives::gen_poly_const::<PARITY_LEN, _>();

    /// Encode 10-byte data with RS(32,10) protection.
    pub(crate) fn encode(data: &[u8; DATA_LEN]) -> [u8; PARITY_LEN] {
        super::primitives::encode(&GEN_POLY, data)
    }

    /// Decode codeword in-place (data || parity) for RS(32,10).
    ///
    /// Returns false if the data was not corrupted. False if the data was corrected but could be
    /// corrected. Returns an error if the data was corrupted and could not be corrected.
    pub(crate) fn decode(codeword: &mut [u8; CODEWORD_SIZE]) -> crate::Result<bool> {
        super::primitives::decode::<
            DATA_LEN,
            PARITY_LEN,
            PARITY_LEN_PLUS_ONE,
            HALF_PARITY_LEN,
            CODEWORD_SIZE,
            MAX_POLY,
        >(codeword)
    }

    #[cfg(test)]
    mod tests {
        use hex_literal::hex;

        use super::*;

        fn test_vector(data: [u8; DATA_LEN], expected_parity: [u8; PARITY_LEN]) {
            let parity = encode(&data);

            assert_eq!(parity, expected_parity);

            let mut cw = [0u8; CODEWORD_SIZE];
            cw[..DATA_LEN].copy_from_slice(&data);
            cw[DATA_LEN..].copy_from_slice(&parity);

            decode(&mut cw).expect("should decode with no errors");

            assert_eq!(&cw[..DATA_LEN], &data);
        }

        #[test]
        fn test_rs_32_10_specification_test_vector_1() {
            let data = hex!("00000000000000000000");
            let expected_parity = hex!("00000000000000000000000000000000000000000000");
            test_vector(data, expected_parity);
        }

        #[test]
        fn test_rs_32_10_specification_test_vector_2() {
            let data = hex!("ffffffffffffffffffff");
            let expected_parity = hex!("6b947c9013410983bf927cd5eafba958214e0fee90ef");
            test_vector(data, expected_parity);
        }

        #[test]
        fn test_rs_32_10_specification_test_vector_3() {
            let data = hex!("00010203040506070809");
            let expected_parity = hex!("2e15b80a2d182f2a0e46a888cf8803394a8b5cdba41d");
            test_vector(data, expected_parity);
        }
    }
}

pub(crate) fn get_generator_poly<const PARITY_LEN: usize>() -> &'static [u8] {
    match PARITY_LEN {
        16 => &code_255_239::GEN_POLY[..16],
        32 => &code_255_223::GEN_POLY[..32],
        64 => &code_255_191::GEN_POLY[..64],
        _ => panic!("Unsupported parity length"),
    }
}
