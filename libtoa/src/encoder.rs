mod block_encoder;
mod ecc_encoder;
mod file_encoder;
mod streaming_encoder;

pub use block_encoder::TOABlockWriter;
pub use ecc_encoder::ECCEncoder;
#[cfg(feature = "std")]
pub use file_encoder::TOAFileEncoder;
pub use streaming_encoder::TOAStreamingEncoder;

use crate::{
    ErrorCorrection, Prefilter, lzma,
    lzma::{EncodeMode, lz::MFType},
};

/// Options for TOA compression.
#[derive(Debug, Clone, Copy)]
pub struct TOAOptions {
    /// Prefilter to apply before compression.
    pub(crate) prefilter: Prefilter,
    /// Reed-Solomon error correction level for data protection.
    pub(crate) error_correction: ErrorCorrection,
    /// Dictionary size to use for the LZMA compression algorithm as a power of two.
    pub(crate) dictionary_size_exponent: u8,
    /// LZMA literal context bits (0-8).
    pub(crate) lc: u8,
    /// LZMA literal position bits (0-4).
    pub(crate) lp: u8,
    /// LZMA position bits (0-4).
    pub(crate) pb: u8,
    /// Compression mode.
    pub(crate) mode: EncodeMode,
    /// Match finder nice length.
    pub(crate) nice_len: u16,
    /// Match finder type.
    pub(crate) mf: MFType,
    /// Match finder depth limit.
    pub(crate) depth_limit: u8,
    /// Block size as power of two exponent. If None, all data will be written in one block.
    /// Block size = 2^exponent bytes. Valid range: 10-62 (1 KiB to 4 EiB).
    pub(crate) block_size_exponent: Option<u8>,
}

impl Default for TOAOptions {
    fn default() -> Self {
        Self {
            prefilter: Prefilter::None,
            error_correction: ErrorCorrection::None,
            dictionary_size_exponent: 26,
            lc: 3,
            lp: 0,
            pb: 2,
            mode: EncodeMode::Normal,
            nice_len: 64,
            mf: MFType::BT4,
            depth_limit: 0,
            block_size_exponent: None,
        }
    }
}

impl TOAOptions {
    const PRESET_TO_DICT_SIZE_LOG2: &'static [u8] = &[
        19, // (0) 512 KiB
        20, // (1) 1 MiB
        21, // (2) 2 MiB
        22, // (3) 4 MiB
        23, // (4) 8 MiB
        24, // (5) 16 MiB
        25, // (6) 32 MiB
        26, // (7) 64 MiB
        27, // (8) 128 MiB
        28, // (9) 256 MiB
    ];

    const PRESET_TO_DEPTH_LIMIT: &'static [u8] = &[4, 8, 24, 48];

    /// Create options with a specific preset level (0-9).
    ///
    /// # Dictionary size
    ///
    ///  - Preset 0: 512 KiB
    ///  - Preset 1:   1 MiB
    ///  - Preset 2:   2 MiB
    ///  - Preset 3:   4 MiB
    ///  - Preset 4:   8 MiB
    ///  - Preset 5:  16 MiB
    ///  - Preset 6:  32 MiB
    ///  - Preset 7:  64 MiB
    ///  - Preset 8: 128 MiB
    ///  - Preset 9: 256 MiB
    ///
    /// Block size is set to the dictionary size.
    pub fn from_preset(preset: u32) -> Self {
        let preset = preset.min(9);

        let error_correction = ErrorCorrection::None;
        let prefilter = Prefilter::None;
        let lc = 3;
        let lp = 0;
        let pb = 2;
        let dictionary_size_exponent = Self::PRESET_TO_DICT_SIZE_LOG2[preset as usize];

        let mode;
        let mf;
        let nice_len;
        let depth_limit;

        if preset <= 3 {
            mode = EncodeMode::Fast;
            mf = MFType::HC4;
            nice_len = if preset <= 1 { 128 } else { 273 };
            depth_limit = Self::PRESET_TO_DEPTH_LIMIT[preset as usize];
        } else {
            mode = EncodeMode::Normal;
            mf = MFType::BT4;
            nice_len = if preset == 4 {
                16
            } else if preset == 5 {
                32
            } else {
                64
            };
            depth_limit = 0;
        }

        Self {
            prefilter,
            error_correction,
            dictionary_size_exponent,
            lc,
            lp,
            pb,
            mode,
            nice_len,
            mf,
            depth_limit,
            block_size_exponent: Some(dictionary_size_exponent),
        }
    }

    /// Sets the LZMA literal context bits.
    ///
    /// Clamped in range of 0 and 8.
    pub fn with_lc(mut self, lc: u8) -> Self {
        self.lc = lc.clamp(0, 8);
        self
    }

    /// Sets the LZMA literal position bits.
    ///
    /// Clamped in range of 0 and 4.
    pub fn with_lp(mut self, lp: u8) -> Self {
        self.lp = lp.clamp(0, 4);
        self
    }

    /// Sets the LZMA position bits (0-4).
    ///
    /// Clamped in range of 0 and 4.
    pub fn with_pb(mut self, pb: u8) -> Self {
        self.pb = pb.clamp(0, 4);
        self
    }

    /// Set the prefilter to use.
    pub fn with_prefilter(mut self, prefilter: Prefilter) -> Self {
        self.prefilter = prefilter;
        self
    }

    /// Set the Reed-Solomon error correction level for data protection.
    pub fn with_error_correction(mut self, error_correction: ErrorCorrection) -> Self {
        self.error_correction = error_correction;
        self
    }

    /// Set the dictionary size exponent of the LZMA compression algorithm.
    ///
    /// Dictionary size = 2^exponent bytes. Valid range: 16-31 (64 KiB to 4 EiB).
    pub fn with_dictionary_exponent(mut self, dictionary_size_log2: u8) -> Self {
        self.dictionary_size_exponent = dictionary_size_log2.clamp(16, 31);
        self
    }

    /// Set the block size exponent for multi-block compression.
    ///
    /// Block size = 2^exponent bytes. Valid range: 16-62 (64 KiB to 4 EiB).
    pub fn with_block_size_exponent(mut self, block_size_exponent: Option<u8>) -> Self {
        self.block_size_exponent = block_size_exponent.map(|exp| exp.clamp(16, 62));
        self
    }

    /// Get dictionary size in bytes.
    pub fn dict_size(&self) -> u32 {
        2u32.pow(self.dictionary_size_exponent as u32)
            .min(lzma::DICT_SIZE_MAX)
    }

    /// Get dictionary size as power of 2 exponent.
    pub fn dict_size_exponent(&self) -> u8 {
        self.dictionary_size_exponent
    }

    /// Get block size as power of 2 exponent.
    pub fn block_size_exponent(&self) -> Option<u8> {
        self.block_size_exponent
    }

    /// Get block size in bytes. Returns None if single-block mode.
    pub fn block_size(&self) -> Option<u64> {
        self.block_size_exponent.map(|exp| 2u64.pow(exp as u32))
    }
}
