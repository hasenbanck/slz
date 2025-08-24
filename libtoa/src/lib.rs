//! # TOA Compression File Format
//!
//! This project implements the **TOA compression file format**, an experimental compression format
//! designed for streaming operation, parallel processing and corruption resilience. The format
//! uses LZMA as the primary compression algorithm with BLAKE3 cryptographic hashing and
//! Reed-Solomon error correction codes.
//!
//! ## ⚠ Specification Status ⚠
//!
//! **TOA is still experimental, do NOT use it in production yet**
//!
//! **Note: The TOA format is currently in draft mode (v0.8) and not yet frozen. The specification
//! may change in future versions.**
//!
//! ## Acknowledgement
//!
//! - The lzma code is a hard copy of the lzma-rust crate (Apache 2 license).
//! - Original Author of the lzma-rust crate was dyz1990 (Apache 2 license)
//! - The lzma-rust2 crate was a rewrite of the XZ for Java by Lasse Collin (0BSD).
//! - Major parts of XZ for Java are based on code written by Igor Pavlov in the LZMA SDK
//!   (public domain).
//!
//! ## License
//!
//! Licensed under the [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0).
#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

mod cv_stack;
mod decoder;
mod lzma;
pub mod reed_solomon;

mod encoder;

mod header;
mod metadata;
#[cfg(not(feature = "std"))]
mod no_std;
mod trailer;
#[cfg(feature = "std")]
mod work_queue;

#[cfg(feature = "std")]
pub(crate) use std::io::Error;
#[cfg(feature = "std")]
pub(crate) use std::io::Read;
#[cfg(feature = "std")]
pub(crate) use std::io::Write;

pub use cv_stack::CVStack;
#[cfg(feature = "std")]
pub use decoder::TOAFileDecoder;
pub use decoder::TOAStreamingDecoder;
#[cfg(feature = "std")]
pub use encoder::TOAFileEncoder;
pub use encoder::{ECCEncoder, TOABlockWriter, TOAOptions, TOAStreamingEncoder};
pub use header::{TOABlockHeader, TOAHeader};
pub use metadata::TOAMetadata;
#[cfg(not(feature = "std"))]
pub use no_std::Error;
#[cfg(not(feature = "std"))]
pub use no_std::Read;
#[cfg(not(feature = "std"))]
pub use no_std::Write;
pub use trailer::TOAFileTrailer;

/// Result type of the crate.
#[cfg(feature = "std")]
pub type Result<T> = core::result::Result<T, Error>;

/// Result type of the crate.
#[cfg(not(feature = "std"))]
pub type Result<T> = core::result::Result<T, Error>;

const TOA_MAGIC: [u8; 4] = [0xFE, 0xDC, 0xBA, 0x98];

const TOA_VERSION: u8 = 0x01;

/// Reed-Solomon error correction levels for data protection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCorrection {
    /// No error correction (only metadata is protected)
    None,
    /// Standard protection - RS(255,239), 6.3% overhead, corrects up to 8 bytes per 255 byte block.
    ///
    /// Handles all normal storage degradation for decades. Use unless you have a specific reason not to.
    Standard,
    /// Paranoid protection - RS(255,223), 12.5% overhead, corrects up to 16 bytes per 255 byte block.
    ///
    /// For specialized scenarios like single-copy archives on sketchy media.
    Paranoid,
    /// Extreme protection - RS(255,191), 25% overhead, corrects up to 32 bytes per 255 byte block.
    ///
    /// For century-scale preservation or very special requirements.
    Extreme,
}

impl ErrorCorrection {
    /// Get the capability bits for the header.
    pub(crate) fn capability_bits(self) -> u8 {
        match self {
            ErrorCorrection::None => 0b00,
            ErrorCorrection::Standard => 0b01,
            ErrorCorrection::Paranoid => 0b10,
            ErrorCorrection::Extreme => 0b11,
        }
    }
}

/// Prefilter types that can be applied before LZMA compression to improve compression ratios
/// for specific data types like executable files.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Prefilter {
    /// No prefilter
    None,
    /// BCJ filter for x86 (32-bit and 64-bit) executables
    BcjX86,
    /// BCJ filter for ARM executables
    BcjArm,
    /// BCJ filter for ARM Thumb executables
    BcjArmThumb,
    /// BCJ filter for ARM64 executables
    BcjArm64,
    /// BCJ filter for SPARC executables
    BcjSparc,
    /// BCJ filter for PowerPC executables
    BcjPowerPc,
    /// BCJ filter for IA-64 executables
    BcjIa64,
    /// BCJ filter for RISC-V executables
    BcjRiscV,
}

impl From<Prefilter> for u8 {
    fn from(value: Prefilter) -> Self {
        match value {
            Prefilter::None => 0x00,
            Prefilter::BcjX86 => 0x01,
            Prefilter::BcjArm => 0x02,
            Prefilter::BcjArmThumb => 0x03,
            Prefilter::BcjArm64 => 0x04,
            Prefilter::BcjSparc => 0x05,
            Prefilter::BcjPowerPc => 0x06,
            Prefilter::BcjIa64 => 0x07,
            Prefilter::BcjRiscV => 0x08,
        }
    }
}

impl TryFrom<u8> for Prefilter {
    type Error = ();

    fn try_from(value: u8) -> std::result::Result<Self, Self::Error> {
        match value {
            0x00 => Ok(Prefilter::None),
            0x01 => Ok(Prefilter::BcjX86),
            0x02 => Ok(Prefilter::BcjArm),
            0x03 => Ok(Prefilter::BcjArmThumb),
            0x04 => Ok(Prefilter::BcjArm64),
            0x05 => Ok(Prefilter::BcjSparc),
            0x06 => Ok(Prefilter::BcjPowerPc),
            0x07 => Ok(Prefilter::BcjIa64),
            0x08 => Ok(Prefilter::BcjRiscV),
            _ => Err(()),
        }
    }
}

#[cfg(feature = "std")]
#[inline(always)]
fn error_other(msg: &'static str) -> Error {
    Error::other(msg)
}

#[cfg(feature = "std")]
#[inline(always)]
fn error_invalid_input(msg: &'static str) -> Error {
    Error::new(std::io::ErrorKind::InvalidInput, msg)
}

#[cfg(feature = "std")]
#[inline(always)]
fn error_invalid_data(msg: &'static str) -> Error {
    Error::new(std::io::ErrorKind::InvalidData, msg)
}

#[cfg(feature = "std")]
#[inline(always)]
fn error_unsupported(msg: &'static str) -> Error {
    Error::new(std::io::ErrorKind::Unsupported, msg)
}

#[cfg(feature = "std")]
#[inline(always)]
fn copy_error(error: &Error) -> Error {
    Error::new(error.kind(), error.to_string())
}

#[cfg(not(feature = "std"))]
#[inline(always)]
fn error_eof() -> Error {
    Error::EOF
}

#[cfg(not(feature = "std"))]
#[inline(always)]
fn error_other(msg: &'static str) -> Error {
    Error::Other(msg)
}

#[cfg(not(feature = "std"))]
#[inline(always)]
fn error_invalid_input(msg: &'static str) -> Error {
    Error::InvalidInput(msg)
}

#[cfg(not(feature = "std"))]
#[inline(always)]
fn error_invalid_data(msg: &'static str) -> Error {
    Error::InvalidData(msg)
}

#[cfg(not(feature = "std"))]
#[inline(always)]
fn error_out_of_memory(msg: &'static str) -> Error {
    Error::OutOfMemory(msg)
}

#[cfg(not(feature = "std"))]
#[inline(always)]
fn error_unsupported(msg: &'static str) -> Error {
    Error::Unsupported(msg)
}

#[cfg(not(feature = "std"))]
#[inline(always)]
fn copy_error(error: &Error) -> Error {
    *error
}

#[cfg(feature = "std")]
/// Optimized copy function with 64KiB buffer for better performance.
pub fn copy_wide<R: Read, W: Write>(reader: &mut R, writer: &mut W) -> Result<u64> {
    const BUFFER_SIZE: usize = 64 * 1024; // 64 KiB buffer
    let mut buf = [0u8; BUFFER_SIZE];
    let mut written = 0u64;

    loop {
        match reader.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => {
                writer.write_all(&buf[..n])?;
                written += n as u64;
            }
            Err(e) => return Err(e),
        }
    }

    Ok(written)
}

#[cfg(feature = "std")]
/// A reader that limits reading to a specific number of bytes and uses BufReader internally.
pub(crate) struct LimitedReader<R> {
    inner: std::io::BufReader<R>,
    remaining: u64,
}

#[cfg(feature = "std")]
impl<R: Read> LimitedReader<R> {
    /// Create a new LimitedReader that will read at most `limit` bytes from the inner reader.
    pub(crate) fn new(reader: R, limit: u64) -> Self {
        Self {
            inner: std::io::BufReader::with_capacity(64 << 10, reader),
            remaining: limit,
        }
    }
}

#[cfg(feature = "std")]
impl<R: Read> Read for LimitedReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> Result<usize> {
        if self.remaining == 0 {
            return Ok(0);
        }
        let max_read = (buf.len() as u64).min(self.remaining) as usize;
        let bytes_read = self.inner.read(&mut buf[..max_read])?;
        self.remaining -= bytes_read as u64;
        Ok(bytes_read)
    }
}

#[cfg(test)]
mod tests {
    pub(crate) struct Lcg(u64);

    impl Lcg {
        pub(crate) fn new(seed: u64) -> Self {
            Lcg(seed)
        }

        pub(crate) fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_mul(0xDA942042E4DD58B5);
            self.0.wrapping_shr(64)
        }

        pub(crate) fn next_u8(&mut self) -> u8 {
            let next = self.next_u64();
            (next >> 16) as u8
        }

        pub(crate) fn next_usize(&mut self, max: usize) -> usize {
            let next = self.next_u64();
            next as usize % max
        }

        pub(crate) fn fill_buffer(&mut self, buf: &mut [u8]) {
            for chunk in buf.chunks_mut(8) {
                let next = self.next_u64();
                let bytes = next.to_le_bytes();
                chunk.copy_from_slice(&bytes[..chunk.len()]);
            }
        }
    }
}
