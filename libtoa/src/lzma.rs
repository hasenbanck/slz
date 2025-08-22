//! This module is based on a hard copy of the lzma-rust2 crate.
//!
//! Original Author of the lzma-rust crate was dyz1990 (Apache 2 license)
//! The lzma-rust2 crate was a rewrite of the XZ for Java by Lasse Collin (0BSD).
//! Major parts of XZ for Java are based on code written by Igor Pavlov in the LZMA SDK
//! (public domain).

pub(crate) mod decoder;
pub(crate) mod enc;
pub mod filter;
pub(crate) mod lz;
mod lzma2s_reader;
pub(crate) mod range_dec;
pub(crate) mod state;

pub use enc::{EncodeMode, LZMA2sWriter, LZMAOptions};
pub use lzma2s_reader::LZMA2sReader;

use crate::{
    Read, Result, Write,
    lzma::state::{STATES, State},
};

/// The maximal size of a dictionary.
pub const DICT_SIZE_MAX: u32 = u32::MAX & !15_u32;

const LOW_SYMBOLS: usize = 1 << 3;
const MID_SYMBOLS: usize = 1 << 3;
const HIGH_SYMBOLS: usize = 1 << 8;

const POS_STATES_MAX: usize = 1 << 4;
const MATCH_LEN_MIN: usize = 2;
const MATCH_LEN_MAX: usize = MATCH_LEN_MIN + LOW_SYMBOLS + MID_SYMBOLS + HIGH_SYMBOLS - 1;

const DIST_STATES: usize = 4;
const DIST_SLOTS: usize = 1 << 6;
const DIST_MODEL_START: usize = 4;
const DIST_MODEL_END: usize = 14;
const FULL_DISTANCES: usize = 1 << (DIST_MODEL_END / 2);

const ALIGN_BITS: usize = 4;
const ALIGN_SIZE: usize = 1 << ALIGN_BITS;
const ALIGN_MASK: usize = ALIGN_SIZE - 1;

const REPS: usize = 4;

const SHIFT_BITS: u32 = 8;
const TOP_MASK: u32 = 0xFF000000;
const BIT_MODEL_TOTAL_BITS: u32 = 11;
const BIT_MODEL_TOTAL: u32 = 1 << BIT_MODEL_TOTAL_BITS;
const PROB_INIT: u16 = (BIT_MODEL_TOTAL / 2) as u16;
const MOVE_BITS: u32 = 5;
const DIST_SPECIAL_INDEX: [usize; 10] = [0, 2, 4, 8, 12, 20, 28, 44, 60, 92];
const DIST_SPECIAL_END: [usize; 10] = [2, 4, 8, 12, 20, 28, 44, 60, 92, 124];

const RC_BIT_MODEL_OFFSET: u32 = (1u32 << MOVE_BITS)
    .wrapping_sub(1)
    .wrapping_sub(BIT_MODEL_TOTAL);

pub(crate) struct LZMACoder {
    pub(crate) pos_mask: u32,
    pub(crate) reps: [i32; REPS],
    pub(crate) state: State,
    pub(crate) is_match: [[u16; POS_STATES_MAX]; STATES],
    pub(crate) is_rep: [u16; STATES],
    pub(crate) is_rep0: [u16; STATES],
    pub(crate) is_rep1: [u16; STATES],
    pub(crate) is_rep2: [u16; STATES],
    pub(crate) is_rep0_long: [[u16; POS_STATES_MAX]; STATES],
    pub(crate) dist_slots: [[u16; DIST_SLOTS]; DIST_STATES],
    dist_special: [u16; 124],
    dist_align: [u16; ALIGN_SIZE],
}

pub(crate) fn coder_get_dict_size(len: usize) -> usize {
    if len < DIST_STATES + MATCH_LEN_MIN {
        len - MATCH_LEN_MIN
    } else {
        DIST_STATES - 1
    }
}

pub(crate) fn get_dist_state(len: u32) -> u32 {
    (if (len as usize) < DIST_STATES + MATCH_LEN_MIN {
        len as usize - MATCH_LEN_MIN
    } else {
        DIST_STATES - 1
    }) as u32
}

impl LZMACoder {
    pub(crate) fn new(pb: usize) -> Self {
        let mut c = Self {
            pos_mask: (1 << pb) - 1,
            reps: Default::default(),
            state: Default::default(),
            is_match: Default::default(),
            is_rep: Default::default(),
            is_rep0: Default::default(),
            is_rep1: Default::default(),
            is_rep2: Default::default(),
            is_rep0_long: Default::default(),
            dist_slots: [[Default::default(); DIST_SLOTS]; DIST_STATES],
            dist_special: [Default::default(); 124],
            dist_align: Default::default(),
        };
        c.reset();
        c
    }

    pub(crate) fn reset(&mut self) {
        self.reps = [0; REPS];
        self.state.reset();
        for ele in self.is_match.iter_mut() {
            init_probs(ele);
        }
        init_probs(&mut self.is_rep);
        init_probs(&mut self.is_rep0);
        init_probs(&mut self.is_rep1);
        init_probs(&mut self.is_rep2);

        for ele in self.is_rep0_long.iter_mut() {
            init_probs(ele);
        }
        for ele in self.dist_slots.iter_mut() {
            init_probs(ele);
        }
        init_probs(&mut self.dist_special);
        init_probs(&mut self.dist_align);
    }

    #[inline(always)]
    pub(crate) fn get_dist_special(&mut self, i: usize) -> &mut [u16] {
        &mut self.dist_special[DIST_SPECIAL_INDEX[i]..DIST_SPECIAL_END[i]]
    }
}

#[inline(always)]
pub(crate) fn init_probs(probs: &mut [u16]) {
    probs.fill(PROB_INIT);
}

pub(crate) struct LiteralCoder {
    lc: u32,
    literal_pos_mask: u32,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct LiteralSubCoder {
    probs: [u16; 0x300],
}

impl LiteralSubCoder {
    pub fn new() -> Self {
        let probs = [PROB_INIT; 0x300];
        Self { probs }
    }

    pub fn reset(&mut self) {
        self.probs = [PROB_INIT; 0x300];
    }
}

impl LiteralCoder {
    pub fn new(lc: u32, lp: u32) -> Self {
        Self {
            lc,
            literal_pos_mask: (1 << lp) - 1,
        }
    }

    pub(crate) fn get_sub_coder_index(&self, prev_byte: u32, pos: u32) -> u32 {
        let low = prev_byte >> (8 - self.lc);
        let high = (pos & self.literal_pos_mask) << self.lc;
        low + high
    }
}

pub(crate) struct LengthCoder {
    choice: [u16; 2],
    low: [[u16; LOW_SYMBOLS]; POS_STATES_MAX],
    mid: [[u16; MID_SYMBOLS]; POS_STATES_MAX],
    high: [u16; HIGH_SYMBOLS],
}

impl LengthCoder {
    pub fn new() -> Self {
        Self {
            choice: Default::default(),
            low: Default::default(),
            mid: Default::default(),
            high: [0; HIGH_SYMBOLS],
        }
    }

    pub fn reset(&mut self) {
        init_probs(&mut self.choice);
        for ele in self.low.iter_mut() {
            init_probs(ele);
        }
        for ele in self.mid.iter_mut() {
            init_probs(ele);
        }
        init_probs(&mut self.high);
    }
}

trait ByteReader {
    fn read_u8(&mut self) -> Result<u8>;

    fn read_u32(&mut self) -> Result<u32>;
}

trait ByteWriter {
    fn write_u8(&mut self, value: u8) -> Result<()>;
}

impl<T: Read> ByteReader for T {
    #[inline(always)]
    fn read_u8(&mut self) -> Result<u8> {
        let mut buf = [0; 1];
        self.read_exact(&mut buf)?;
        Ok(buf[0])
    }

    #[inline(always)]
    fn read_u32(&mut self) -> Result<u32> {
        let mut buf = [0; 4];
        self.read_exact(buf.as_mut())?;
        Ok(u32::from_be_bytes(buf))
    }
}

impl<T: Write> ByteWriter for T {
    #[inline(always)]
    fn write_u8(&mut self, value: u8) -> Result<()> {
        self.write_all(&[value])
    }
}
