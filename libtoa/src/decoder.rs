mod ecc_decoder;
mod streaming_decoder;

#[cfg(feature = "std")]
mod file_decoder;

#[cfg(feature = "std")]
pub use file_decoder::TOAFileDecoder;
pub use streaming_decoder::TOAStreamingDecoder;

pub(crate) use ecc_decoder::ECCDecoder;

use crate::{
    ErrorCorrection, Prefilter, Read, Result,
    lzma::{LZMA2sDecoder, filter::bcj::BCJDecoder},
};

/// All possible decoder combinations.
#[allow(clippy::large_enum_variant)]
enum Decoder<R> {
    Lzma(LZMA2sDecoder<ECCDecoder<R>>),
    BcjX86(BCJDecoder<LZMA2sDecoder<ECCDecoder<R>>>),
    BcjArm(BCJDecoder<LZMA2sDecoder<ECCDecoder<R>>>),
    BcjArmThumb(BCJDecoder<LZMA2sDecoder<ECCDecoder<R>>>),
    BcjArm64(BCJDecoder<LZMA2sDecoder<ECCDecoder<R>>>),
    BcjSparc(BCJDecoder<LZMA2sDecoder<ECCDecoder<R>>>),
    BcjPowerPc(BCJDecoder<LZMA2sDecoder<ECCDecoder<R>>>),
    BcjIa64(BCJDecoder<LZMA2sDecoder<ECCDecoder<R>>>),
    BcjRiscV(BCJDecoder<LZMA2sDecoder<ECCDecoder<R>>>),
}

impl<R: Read> Read for Decoder<R> {
    fn read(&mut self, buf: &mut [u8]) -> Result<usize> {
        match self {
            Decoder::Lzma(decoder) => decoder.read(buf),
            Decoder::BcjX86(decoder) => decoder.read(buf),
            Decoder::BcjArm(decoder) => decoder.read(buf),
            Decoder::BcjArmThumb(decoder) => decoder.read(buf),
            Decoder::BcjArm64(decoder) => decoder.read(buf),
            Decoder::BcjSparc(decoder) => decoder.read(buf),
            Decoder::BcjPowerPc(decoder) => decoder.read(buf),
            Decoder::BcjIa64(decoder) => decoder.read(buf),
            Decoder::BcjRiscV(decoder) => decoder.read(buf),
        }
    }
}

impl<R: Read> Decoder<R> {
    /// Create a new decoder chain based on the header configuration.
    #[allow(clippy::too_many_arguments)]
    fn new(
        decoder: R,
        prefilter: Prefilter,
        error_correction: ErrorCorrection,
        validate_rs: bool,
        lc: u8,
        lp: u8,
        pb: u8,
        dict_size: u32,
    ) -> Result<Self> {
        let ecc_decoder = ECCDecoder::new(decoder, error_correction, validate_rs);

        let lzma_decoder = LZMA2sDecoder::new(
            ecc_decoder,
            u32::from(lc),
            u32::from(lp),
            u32::from(pb),
            dict_size,
        );

        #[rustfmt::skip]
        let chain = match prefilter {
            Prefilter::None => Decoder::Lzma(lzma_decoder),
            Prefilter::BcjX86 => Decoder::BcjX86(BCJDecoder::new_x86(lzma_decoder, 0)),
            Prefilter::BcjArm => Decoder::BcjArm(BCJDecoder::new_arm(lzma_decoder, 0)),
            Prefilter::BcjArmThumb => Decoder::BcjArmThumb(BCJDecoder::new_arm_thumb(lzma_decoder, 0)),
            Prefilter::BcjArm64 => Decoder::BcjArm64(BCJDecoder::new_arm64(lzma_decoder, 0)),
            Prefilter::BcjSparc => Decoder::BcjSparc(BCJDecoder::new_sparc(lzma_decoder, 0)),
            Prefilter::BcjPowerPc => Decoder::BcjPowerPc(BCJDecoder::new_ppc(lzma_decoder, 0)),
            Prefilter::BcjIa64 => Decoder::BcjIa64(BCJDecoder::new_ia64(lzma_decoder, 0)),
            Prefilter::BcjRiscV => Decoder::BcjRiscV(BCJDecoder::new_riscv(lzma_decoder, 0)),
        };

        Ok(chain)
    }

    /// Extract the inner decoder from the decoder chain.
    fn into_inner(self) -> R {
        match self {
            Decoder::Lzma(decoder) => decoder.into_inner().into_inner(),
            Decoder::BcjX86(decoder) => decoder.into_inner().into_inner().into_inner(),
            Decoder::BcjArm(decoder) => decoder.into_inner().into_inner().into_inner(),
            Decoder::BcjArmThumb(decoder) => decoder.into_inner().into_inner().into_inner(),
            Decoder::BcjArm64(decoder) => decoder.into_inner().into_inner().into_inner(),
            Decoder::BcjSparc(decoder) => decoder.into_inner().into_inner().into_inner(),
            Decoder::BcjPowerPc(decoder) => decoder.into_inner().into_inner().into_inner(),
            Decoder::BcjIa64(decoder) => decoder.into_inner().into_inner().into_inner(),
            Decoder::BcjRiscV(decoder) => decoder.into_inner().into_inner().into_inner(),
        }
    }
}
