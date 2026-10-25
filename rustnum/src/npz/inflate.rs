//! DEFLATE decompressor (RFC 1951): stored, fixed-Huffman and dynamic-Huffman blocks.
//!
//! The decoder never panics and always terminates on corrupt input: every loop iteration
//! consumes input bits or returns an error, and the output size can be capped by the caller.

use std::fmt;

/// Why a DEFLATE stream could not be decoded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InflateError {
    /// The input ended in the middle of the stream.
    UnexpectedEof,
    /// Block type 3 (reserved).
    InvalidBlockType,
    /// A stored block whose LEN and NLEN fields disagree.
    StoredLengthMismatch,
    /// A dynamic block declares more than 286 literal/length or 30 distance codes.
    TooManyCodes,
    /// The Huffman code lengths are over-subscribed, incomplete or malformed.
    InvalidCodeLengths,
    /// A bit sequence that is not a code of the current Huffman table.
    InvalidCode,
    /// A literal/length symbol 286 or 287, which never occur in valid data.
    InvalidLengthSymbol,
    /// A distance symbol 30 or 31, which never occur in valid data.
    InvalidDistanceSymbol,
    /// A match that reaches back before the start of the output.
    DistanceTooFar,
    /// The literal/length table has no end-of-block code.
    MissingEndOfBlock,
    /// The output would exceed the caller's limit.
    OutputLimitExceeded,
}

impl fmt::Display for InflateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            InflateError::UnexpectedEof => "unexpected end of deflate stream",
            InflateError::InvalidBlockType => "invalid deflate block type",
            InflateError::StoredLengthMismatch => "stored block length check failed",
            InflateError::TooManyCodes => "too many length or distance codes",
            InflateError::InvalidCodeLengths => "invalid huffman code lengths",
            InflateError::InvalidCode => "invalid huffman code",
            InflateError::InvalidLengthSymbol => "invalid length symbol",
            InflateError::InvalidDistanceSymbol => "invalid distance symbol",
            InflateError::DistanceTooFar => "distance reaches before start of output",
            InflateError::MissingEndOfBlock => "huffman table has no end-of-block code",
            InflateError::OutputLimitExceeded => "decompressed data exceeds the expected size",
        };
        f.write_str(s)
    }
}

impl std::error::Error for InflateError {}

/// Base match length of length symbols 257..=285.
pub(super) const LEN_BASE: [u16; 29] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131,
    163, 195, 227, 258,
];
/// Extra bits of length symbols 257..=285.
pub(super) const LEN_EXTRA: [u8; 29] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
];
/// Base distance of distance symbols 0..=29.
pub(super) const DIST_BASE: [u16; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
    2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
];
/// Extra bits of distance symbols 0..=29.
pub(super) const DIST_EXTRA: [u8; 30] = [
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13,
    13,
];
/// Order in which code-length code lengths are stored in a dynamic block header.
pub(super) const CLEN_ORDER: [usize; 19] = [
    16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
];

/// LSB-first bit reader over a byte slice.
struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
    buf: u64,
    cnt: u32,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        BitReader {
            data,
            pos: 0,
            buf: 0,
            cnt: 0,
        }
    }

    #[inline]
    fn refill(&mut self) {
        while self.cnt <= 56 && self.pos < self.data.len() {
            self.buf |= (self.data[self.pos] as u64) << self.cnt;
            self.pos += 1;
            self.cnt += 8;
        }
    }

    /// Read `n <= 32` bits.
    #[inline]
    fn bits(&mut self, n: u32) -> Result<u32, InflateError> {
        if self.cnt < n {
            self.refill();
            if self.cnt < n {
                return Err(InflateError::UnexpectedEof);
            }
        }
        let v = (self.buf & ((1u64 << n) - 1)) as u32;
        self.buf >>= n;
        self.cnt -= n;
        Ok(v)
    }

    /// Drop the bits up to the next byte boundary.
    fn align(&mut self) {
        let drop = self.cnt % 8;
        self.buf >>= drop;
        self.cnt -= drop;
    }

    /// Append `n` raw bytes to `out` (the reader must be byte aligned).
    fn copy_bytes(&mut self, n: usize, out: &mut Vec<u8>) -> Result<(), InflateError> {
        let mut left = n;
        while left > 0 && self.cnt >= 8 {
            out.push(self.buf as u8);
            self.buf >>= 8;
            self.cnt -= 8;
            left -= 1;
        }
        if left > 0 {
            let end = self
                .pos
                .checked_add(left)
                .filter(|&e| e <= self.data.len())
                .ok_or(InflateError::UnexpectedEof)?;
            out.extend_from_slice(&self.data[self.pos..end]);
            self.pos = end;
        }
        Ok(())
    }

    /// Number of input bytes fully or partly consumed (whole unread bytes in the bit
    /// buffer are given back).
    fn consumed(&self) -> usize {
        self.pos - (self.cnt / 8) as usize
    }
}

/// A canonical Huffman decoding table indexed by `max_len` look-ahead bits.
///
/// Entry layout: `symbol << 4 | code_length`; 0 marks an unused bit pattern.
struct Huffman {
    table: Vec<u16>,
    max_len: u32,
}

impl Huffman {
    fn new(lengths: &[u8], is_code_length_code: bool) -> Result<Huffman, InflateError> {
        let mut count = [0u32; 16];
        for &l in lengths {
            count[l as usize] += 1;
        }
        count[0] = 0;
        let max_len = (1..=15u32)
            .rev()
            .find(|&l| count[l as usize] > 0)
            .unwrap_or(0);
        if max_len == 0 {
            // No codes at all: legal (a block of literals only has no distance codes),
            // but any attempt to decode a symbol fails.
            return Ok(Huffman {
                table: vec![0; 2],
                max_len: 1,
            });
        }
        let mut left: i32 = 1;
        for &c in &count[1..] {
            left <<= 1;
            left -= c as i32;
            if left < 0 {
                return Err(InflateError::InvalidCodeLengths);
            }
        }
        // Incomplete codes are only tolerated for a lone one-bit code (as zlib does).
        if left > 0 && (is_code_length_code || max_len != 1) {
            return Err(InflateError::InvalidCodeLengths);
        }
        let mut next = [0u32; 16];
        let mut code = 0u32;
        for l in 1..16 {
            code = (code + count[l - 1]) << 1;
            next[l] = code;
        }
        let size = 1usize << max_len;
        let mut table = vec![0u16; size];
        for (sym, &l) in lengths.iter().enumerate() {
            if l == 0 {
                continue;
            }
            let l = u32::from(l);
            let c = next[l as usize];
            next[l as usize] += 1;
            let rev = (c.reverse_bits() >> (32 - l)) as usize;
            let entry = ((sym as u16) << 4) | l as u16;
            let mut idx = rev;
            while idx < size {
                table[idx] = entry;
                idx += 1 << l;
            }
        }
        Ok(Huffman { table, max_len })
    }

    #[inline]
    fn decode(&self, br: &mut BitReader<'_>) -> Result<u16, InflateError> {
        if br.cnt < self.max_len {
            br.refill();
        }
        let e = self.table[(br.buf & ((1u64 << self.max_len) - 1)) as usize];
        let len = u32::from(e & 15);
        if len == 0 {
            return Err(if br.cnt < self.max_len {
                InflateError::UnexpectedEof
            } else {
                InflateError::InvalidCode
            });
        }
        if len > br.cnt {
            return Err(InflateError::UnexpectedEof);
        }
        br.buf >>= len;
        br.cnt -= len;
        Ok(e >> 4)
    }
}

fn fixed_tables() -> Result<(Huffman, Huffman), InflateError> {
    let mut lit = [0u8; 288];
    for (i, l) in lit.iter_mut().enumerate() {
        *l = match i {
            0..=143 => 8,
            144..=255 => 9,
            256..=279 => 7,
            _ => 8,
        };
    }
    Ok((Huffman::new(&lit, false)?, Huffman::new(&[5u8; 32], false)?))
}

fn dynamic_tables(br: &mut BitReader<'_>) -> Result<(Huffman, Huffman), InflateError> {
    let hlit = br.bits(5)? as usize + 257;
    let hdist = br.bits(5)? as usize + 1;
    let hclen = br.bits(4)? as usize + 4;
    if hlit > 286 || hdist > 30 {
        return Err(InflateError::TooManyCodes);
    }
    let mut clens = [0u8; 19];
    for &idx in CLEN_ORDER.iter().take(hclen) {
        clens[idx] = br.bits(3)? as u8;
    }
    let clh = Huffman::new(&clens, true)?;
    let total = hlit + hdist;
    let mut lens = [0u8; 286 + 30];
    let mut i = 0;
    while i < total {
        let sym = clh.decode(br)?;
        let (value, repeat) = match sym {
            0..=15 => (sym as u8, 1),
            16 => {
                if i == 0 {
                    return Err(InflateError::InvalidCodeLengths);
                }
                (lens[i - 1], 3 + br.bits(2)? as usize)
            }
            17 => (0, 3 + br.bits(3)? as usize),
            _ => (0, 11 + br.bits(7)? as usize),
        };
        if i + repeat > total {
            return Err(InflateError::InvalidCodeLengths);
        }
        lens[i..i + repeat].fill(value);
        i += repeat;
    }
    if lens[256] == 0 {
        return Err(InflateError::MissingEndOfBlock);
    }
    Ok((
        Huffman::new(&lens[..hlit], false)?,
        Huffman::new(&lens[hlit..total], false)?,
    ))
}

/// Decompress a raw DEFLATE stream, producing at most `limit` bytes.
///
/// Returns the data and the number of input bytes the stream occupied (bytes after the
/// final block, such as a ZIP data descriptor, are not counted).
pub fn inflate_with_limit(input: &[u8], limit: usize) -> Result<(Vec<u8>, usize), InflateError> {
    let mut br = BitReader::new(input);
    let mut out: Vec<u8> =
        Vec::with_capacity(limit.min(input.len().saturating_mul(3)).min(1 << 26));
    let mut fixed: Option<(Huffman, Huffman)> = None;
    loop {
        let last = br.bits(1)? == 1;
        match br.bits(2)? {
            0 => {
                br.align();
                let len = br.bits(16)? as usize;
                let nlen = br.bits(16)? as usize;
                if len != (!nlen & 0xFFFF) {
                    return Err(InflateError::StoredLengthMismatch);
                }
                if out.len() + len > limit {
                    return Err(InflateError::OutputLimitExceeded);
                }
                br.copy_bytes(len, &mut out)?;
            }
            1 => {
                if fixed.is_none() {
                    fixed = Some(fixed_tables()?);
                }
                if let Some((lit, dist)) = fixed.as_ref() {
                    inflate_block(&mut br, &mut out, lit, dist, limit)?;
                }
            }
            2 => {
                let (lit, dist) = dynamic_tables(&mut br)?;
                inflate_block(&mut br, &mut out, &lit, &dist, limit)?;
            }
            _ => return Err(InflateError::InvalidBlockType),
        }
        if last {
            break;
        }
    }
    Ok((out, br.consumed()))
}

fn inflate_block(
    br: &mut BitReader<'_>,
    out: &mut Vec<u8>,
    lit: &Huffman,
    dist: &Huffman,
    limit: usize,
) -> Result<(), InflateError> {
    loop {
        let sym = lit.decode(br)? as usize;
        if sym < 256 {
            if out.len() >= limit {
                return Err(InflateError::OutputLimitExceeded);
            }
            out.push(sym as u8);
        } else if sym == 256 {
            return Ok(());
        } else {
            let li = sym - 257;
            if li >= 29 {
                return Err(InflateError::InvalidLengthSymbol);
            }
            let len = LEN_BASE[li] as usize + br.bits(u32::from(LEN_EXTRA[li]))? as usize;
            let ds = dist.decode(br)? as usize;
            if ds >= 30 {
                return Err(InflateError::InvalidDistanceSymbol);
            }
            let d = DIST_BASE[ds] as usize + br.bits(u32::from(DIST_EXTRA[ds]))? as usize;
            if d > out.len() {
                return Err(InflateError::DistanceTooFar);
            }
            if out.len() + len > limit {
                return Err(InflateError::OutputLimitExceeded);
            }
            let start = out.len() - d;
            if d >= len {
                out.extend_from_within(start..start + len);
            } else {
                for k in 0..len {
                    let b = out[start + k];
                    out.push(b);
                }
            }
        }
    }
}

/// Decompress a raw DEFLATE stream with no size limit.
#[allow(dead_code)] // byte-level API exercised by the unit tests; not used by the array layer
pub fn inflate(input: &[u8]) -> Result<Vec<u8>, InflateError> {
    inflate_with_limit(input, usize::MAX).map(|(v, _)| v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::npz::testutil::{self, Python};

    const HELLO_FIXED: &[u8] = &[0xcb, 0x48, 0xcd, 0xc9, 0xc9, 0x57, 0xc8, 0x40, 0x27, 0x01];
    const TEXT: &[u8] = b"The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs.";
    /// Level-9 zlib output for `TEXT`; the block is dynamic Huffman.
    const TEXT_DYNAMIC: &[u8] = &[
        0xb5, 0xcb, 0xc7, 0x01, 0x80, 0x20, 0x10, 0x05, 0xd1, 0x56, 0x7e, 0x05, 0xd4, 0xe2, 0xc1,
        0x06, 0x40, 0x49, 0x06, 0x56, 0xb2, 0x50, 0xbd, 0xdb, 0x84, 0xe7, 0x79, 0xb3, 0x3a, 0x8d,
        0x58, 0xfd, 0x76, 0x42, 0x25, 0xea, 0x01, 0x86, 0x5e, 0x1c, 0xf5, 0x7e, 0x32, 0xa8, 0xe9,
        0x84, 0xc2, 0xf9, 0x92, 0x73, 0x60, 0x27, 0x2b, 0xb0, 0xfe, 0x86, 0x17, 0xc9, 0xee, 0x1e,
        0x50, 0x8c, 0xba, 0x2f, 0x0e, 0xc6, 0x37, 0xcd, 0x69, 0xea, 0x80, 0xcb, 0xc7, 0x4a, 0x89,
        0x5f, 0x9b, 0xc5, 0x07,
    ];

    #[test]
    fn fixed_block_from_zlib() {
        assert_eq!(inflate(HELLO_FIXED).unwrap(), b"hello hello hello hello");
    }

    #[test]
    fn dynamic_block_from_zlib() {
        assert_eq!(
            TEXT_DYNAMIC[0] >> 1 & 3,
            2,
            "fixture must be a dynamic block"
        );
        assert_eq!(inflate(TEXT_DYNAMIC).unwrap(), TEXT);
    }

    #[test]
    fn stored_block() {
        let s = [0x01, 0x03, 0x00, 0xfc, 0xff, b'a', b'b', b'c'];
        assert_eq!(inflate(&s).unwrap(), b"abc");
        let (out, used) = inflate_with_limit(&s, 10).unwrap();
        assert_eq!((out.as_slice(), used), (&b"abc"[..], 8));
    }

    #[test]
    fn empty_stream() {
        assert_eq!(inflate(&[0x03, 0x00]).unwrap(), b"");
    }

    #[test]
    fn overlapping_match_repeats_bytes() {
        // Literal 'a' then a match of length 10 at distance 1.
        let z = crate::npz::deflate::deflate_with(&[b'a'; 11], crate::npz::deflate::Mode::Fixed);
        assert_eq!(inflate(&z).unwrap(), [b'a'; 11]);
    }

    #[test]
    fn reports_consumed_bytes_and_ignores_trailer() {
        let mut z = HELLO_FIXED.to_vec();
        z.extend_from_slice(b"TRAILER");
        let (out, used) = inflate_with_limit(&z, usize::MAX).unwrap();
        assert_eq!(out, b"hello hello hello hello");
        assert_eq!(used, HELLO_FIXED.len());
    }

    #[test]
    fn empty_input_is_eof() {
        assert_eq!(inflate(&[]), Err(InflateError::UnexpectedEof));
    }

    #[test]
    fn reserved_block_type_is_rejected() {
        assert_eq!(inflate(&[0x07, 0x00]), Err(InflateError::InvalidBlockType));
    }

    #[test]
    fn stored_length_mismatch_is_rejected() {
        let s = [0x01, 0x03, 0x00, 0x00, 0x00, b'a', b'b', b'c'];
        assert_eq!(inflate(&s), Err(InflateError::StoredLengthMismatch));
    }

    #[test]
    fn truncated_stored_block_is_eof() {
        let s = [0x01, 0x05, 0x00, 0xfa, 0xff, b'a', b'b'];
        assert_eq!(inflate(&s), Err(InflateError::UnexpectedEof));
    }

    #[test]
    fn every_truncation_of_a_valid_stream_errors_without_panicking() {
        for stream in [HELLO_FIXED, TEXT_DYNAMIC] {
            for cut in 0..stream.len() {
                assert!(inflate(&stream[..cut]).is_err(), "cut {cut}");
            }
        }
    }

    #[test]
    fn distance_before_start_is_rejected() {
        // Fixed block whose first symbol is a match (length 3, distance 1).
        // Bits, LSB first: BFINAL=1, BTYPE=01 -> 1,1,0; length symbol 257 is the 7-bit code
        // 0000001 (sent MSB first); distance symbol 0 is 00000; then end-of-block 0000000.
        let mut bits = vec![1u8, 1, 0];
        bits.extend([0, 0, 0, 0, 0, 0, 1]);
        bits.extend([0, 0, 0, 0, 0]);
        bits.extend([0, 0, 0, 0, 0, 0, 0]);
        let mut bytes = vec![0u8; bits.len().div_ceil(8)];
        for (i, b) in bits.iter().enumerate() {
            bytes[i / 8] |= b << (i % 8);
        }
        assert_eq!(inflate(&bytes), Err(InflateError::DistanceTooFar));
    }

    #[test]
    fn output_limit_is_enforced() {
        assert_eq!(
            inflate_with_limit(HELLO_FIXED, 5),
            Err(InflateError::OutputLimitExceeded)
        );
        assert!(inflate_with_limit(HELLO_FIXED, 23).is_ok());
        assert_eq!(
            inflate_with_limit(&[0x01, 0x03, 0x00, 0xfc, 0xff, 1, 2, 3], 2),
            Err(InflateError::OutputLimitExceeded)
        );
    }

    #[test]
    fn garbage_never_panics_or_hangs() {
        for seed in 0..3000u64 {
            let len = (seed % 200) as usize;
            let data = testutil::pseudo_random(len, seed);
            let _ = inflate_with_limit(&data, 1 << 20);
        }
    }

    #[test]
    fn bit_flips_in_valid_stream_never_panic() {
        for i in 0..TEXT_DYNAMIC.len() * 8 {
            let mut z = TEXT_DYNAMIC.to_vec();
            z[i / 8] ^= 1 << (i % 8);
            let _ = inflate_with_limit(&z, 1 << 20);
        }
    }

    #[test]
    fn endless_empty_stored_blocks_end_at_eof() {
        // Non-final empty stored blocks, repeated: must run out of input, not loop.
        let block = [0x00, 0x00, 0x00, 0xff, 0xff];
        let data: Vec<u8> = block.iter().copied().cycle().take(5000).collect();
        assert_eq!(inflate(&data), Err(InflateError::UnexpectedEof));
    }

    #[test]
    fn oversubscribed_and_incomplete_codes_are_rejected() {
        assert!(Huffman::new(&[1, 1, 1], false).is_err());
        assert!(Huffman::new(&[1, 2], false).is_err());
        assert!(Huffman::new(&[1], false).is_ok());
        assert!(Huffman::new(&[1, 1], false).is_ok());
        assert!(Huffman::new(&[1], true).is_err());
    }

    #[test]
    fn dynamic_header_with_too_many_codes_is_rejected() {
        // BFINAL=1, BTYPE=10, HLIT=31 (288 literal/length codes).
        let bytes = [0b1111_1101, 0b0000_0011, 0, 0, 0, 0];
        assert_eq!(inflate(&bytes), Err(InflateError::TooManyCodes));
    }

    /// Python writes `[u32 count] ([u32 len][raw][u32 len][deflated])*` for many cases.
    const PY_CASES: &str = r#"
import sys, zlib, struct, random
import numpy as np
rng = random.Random(7)
nprng = np.random.default_rng(7)
payloads = [
    b"", b"a", b"ab" * 3, bytes(range(256)) * 3,
    b"hello world " * 5000,
    bytes(300000),
    bytes(rng.getrandbits(8) for _ in range(70000)),
    nprng.standard_normal(20000).tobytes(),
    (nprng.integers(0, 50, 60000)).astype(np.int32).tobytes(),
    (b"ACGT" * 100 + bytes(rng.getrandbits(2) + 65 for _ in range(500))) * 40,
]
cases = []
for p in payloads:
    for level in (0, 1, 6, 9):
        for strat in (zlib.Z_DEFAULT_STRATEGY, zlib.Z_FILTERED, zlib.Z_HUFFMAN_ONLY, zlib.Z_RLE, zlib.Z_FIXED):
            c = zlib.compressobj(level, zlib.DEFLATED, -15, 9, strat)
            cases.append((p, c.compress(p) + c.flush()))
out = sys.stdout.buffer
out.write(struct.pack("<I", len(cases)))
for raw, comp in cases:
    out.write(struct.pack("<I", len(raw))); out.write(raw)
    out.write(struct.pack("<I", len(comp))); out.write(comp)
"#;

    #[test]
    fn decodes_everything_python_zlib_produces() {
        let Some(py) = Python::get() else { return };
        let buf = py.run(PY_CASES, &[], &[]);
        let rd = |p: &mut usize| -> Vec<u8> {
            let n = u32::from_le_bytes([buf[*p], buf[*p + 1], buf[*p + 2], buf[*p + 3]]) as usize;
            *p += 4;
            let s = buf[*p..*p + n].to_vec();
            *p += n;
            s
        };
        let count = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
        let mut p = 4;
        assert_eq!(count, 10 * 4 * 5);
        for case in 0..count {
            let raw = rd(&mut p);
            let comp = rd(&mut p);
            let got = inflate(&comp).unwrap_or_else(|e| panic!("case {case}: {e}"));
            assert!(got == raw, "case {case}: output differs");
            let (_, used) = inflate_with_limit(&comp, raw.len()).unwrap();
            assert_eq!(used, comp.len(), "case {case}: consumed");
        }
    }
}
