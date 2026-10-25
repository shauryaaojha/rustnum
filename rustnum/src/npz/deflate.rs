//! DEFLATE compressor (RFC 1951).
//!
//! LZ77 with a hash-chain matcher and one-step lazy matching, followed by Huffman coding.
//! Each block is emitted as whichever of stored, fixed-Huffman or dynamic-Huffman is smallest
//! (see [`Mode`] to force one). The output is a raw DEFLATE stream (no zlib or gzip wrapper).

use super::inflate::{CLEN_ORDER, LEN_BASE, LEN_EXTRA};

const WSIZE: usize = 32768;
const MIN_MATCH: usize = 3;
const MAX_MATCH: usize = 258;
const HASH_BITS: u32 = 15;
/// Tokens per block; smaller blocks adapt better, bigger ones amortise the header.
const BLOCK_TOKENS: usize = 1 << 15;
const MAX_STORED: usize = 65535;
const MAX_CHAIN: usize = 96;
/// Stop searching once a match this long is found.
const NICE_MATCH: usize = 160;
/// Do not try a lazy match when the current match is at least this long.
const LAZY_MAX: usize = 48;
/// A 3-byte match farther away than this costs more than three literals.
const TOO_FAR: usize = 4096;
const MATCH_FLAG: u32 = 1 << 31;

/// Block types the encoder may use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// Only stored (uncompressed) blocks.
    Stored,
    /// Only fixed-Huffman blocks.
    #[allow(dead_code)]
    // byte-level API exercised by the unit tests; not used by the array layer
    Fixed,
    /// Only dynamic-Huffman blocks.
    #[allow(dead_code)]
    // byte-level API exercised by the unit tests; not used by the array layer
    Dynamic,
    /// Per block, the smallest of the three.
    Auto,
}

const fn build_len_sym() -> [u8; MAX_MATCH + 1] {
    let mut t = [0u8; MAX_MATCH + 1];
    let mut code = 0;
    while code < 29 {
        let base = LEN_BASE[code] as usize;
        let n = 1usize << LEN_EXTRA[code];
        let mut k = 0;
        while k < n {
            if base + k <= MAX_MATCH {
                t[base + k] = code as u8;
            }
            k += 1;
        }
        code += 1;
    }
    // 258 has its own code (285) although code 284 could also express it.
    t[MAX_MATCH] = 28;
    t
}

/// Length code index (0..29, i.e. symbol - 257) for each match length.
static LEN_SYM: [u8; MAX_MATCH + 1] = build_len_sym();

/// Distance code for `dist` (1..=32768): (code, extra bit count, extra value).
fn dist_code(dist: usize) -> (usize, u32, u32) {
    if dist <= 4 {
        return (dist - 1, 0, 0);
    }
    let d = dist - 1;
    let nb = (usize::BITS - 1 - d.leading_zeros()) as usize; // floor(log2(d)) >= 2
    let code = 2 * nb + ((d >> (nb - 1)) & 1);
    let extra_bits = (nb - 1) as u32;
    (code, extra_bits, (d & ((1 << extra_bits) - 1)) as u32)
}

struct BitWriter {
    out: Vec<u8>,
    acc: u64,
    nbits: u32,
}

impl BitWriter {
    #[inline]
    fn put(&mut self, value: u32, n: u32) {
        debug_assert!(n <= 32 && (n == 32 || value < (1 << n)));
        self.acc |= u64::from(value) << self.nbits;
        self.nbits += n;
        while self.nbits >= 8 {
            self.out.push(self.acc as u8);
            self.acc >>= 8;
            self.nbits -= 8;
        }
    }

    fn align(&mut self) {
        if self.nbits > 0 {
            self.out.push(self.acc as u8);
            self.acc = 0;
            self.nbits = 0;
        }
    }
}

/// Canonical Huffman codes (bit-reversed, ready for an LSB-first writer).
fn codes_from_lengths(lens: &[u8]) -> Vec<u16> {
    let mut count = [0u32; 16];
    for &l in lens {
        count[l as usize] += 1;
    }
    count[0] = 0;
    let mut next = [0u32; 16];
    let mut code = 0u32;
    for l in 1..16 {
        code = (code + count[l - 1]) << 1;
        next[l] = code;
    }
    lens.iter()
        .map(|&l| {
            if l == 0 {
                0
            } else {
                let c = next[l as usize];
                next[l as usize] += 1;
                (c.reverse_bits() >> (32 - u32::from(l))) as u16
            }
        })
        .collect()
}

/// Length-limited Huffman code lengths for the given symbol frequencies.
///
/// Symbols with frequency 0 get length 0; a single used symbol gets length 1.
fn lengths_from_freqs(freqs: &[u32], max_len: usize) -> Vec<u8> {
    let mut lens = vec![0u8; freqs.len()];
    let mut leaves: Vec<(u32, usize)> = freqs
        .iter()
        .enumerate()
        .filter(|&(_, &f)| f > 0)
        .map(|(s, &f)| (f, s))
        .collect();
    match leaves.len() {
        0 => return lens,
        1 => {
            lens[leaves[0].1] = 1;
            return lens;
        }
        _ => {}
    }
    leaves.sort_unstable();
    let n = leaves.len();
    let total_nodes = 2 * n - 1;
    let mut weight = vec![0u64; total_nodes];
    let mut parent = vec![0usize; total_nodes];
    for (w, l) in weight.iter_mut().zip(&leaves) {
        *w = u64::from(l.0);
    }
    // Two-queue Huffman construction: leaves are sorted, internal nodes are created in
    // non-decreasing weight order.
    let (mut li, mut ii, mut next) = (0usize, n, n);
    while next < total_nodes {
        let mut picked = [0usize; 2];
        for p in picked.iter_mut() {
            if li < n && (ii >= next || weight[li] <= weight[ii]) {
                *p = li;
                li += 1;
            } else {
                *p = ii;
                ii += 1;
            }
        }
        weight[next] = weight[picked[0]] + weight[picked[1]];
        parent[picked[0]] = next;
        parent[picked[1]] = next;
        next += 1;
    }
    let mut depth = vec![0usize; total_nodes];
    for k in (0..total_nodes - 1).rev() {
        depth[k] = depth[parent[k]] + 1;
    }
    let mut num = vec![0i64; max_len + 1];
    for &d in depth.iter().take(n) {
        num[d.min(max_len)] += 1;
    }
    // Repair the Kraft sum after clamping over-long codes (same scheme as miniz).
    let mut total: i64 = (1..=max_len).map(|l| num[l] << (max_len - l)).sum();
    while total > 1i64 << max_len {
        num[max_len] -= 1;
        for l in (1..max_len).rev() {
            if num[l] > 0 {
                num[l] -= 1;
                num[l + 1] += 2;
                break;
            }
        }
        total -= 1;
    }
    // The least frequent symbols (front of `leaves`) get the longest codes.
    let mut idx = 0;
    for l in (1..=max_len).rev() {
        for _ in 0..num[l] {
            lens[leaves[idx].1] = l as u8;
            idx += 1;
        }
    }
    lens
}

/// Make sure at least two symbols are in use (some decoders reject one-code tables).
fn ensure_two(freqs: &mut [u32]) {
    let mut used = freqs.iter().filter(|&&f| f > 0).count();
    for f in freqs.iter_mut() {
        if used >= 2 {
            break;
        }
        if *f == 0 {
            *f = 1;
            used += 1;
        }
    }
}

fn cost(freqs: &[u32], lens: &[u8]) -> u64 {
    freqs
        .iter()
        .zip(lens)
        .map(|(&f, &l)| u64::from(f) * u64::from(l))
        .sum()
}

/// Run-length encode code lengths with symbols 16/17/18: (symbol, extra bit count, extra value).
fn rle_lengths(lens: &[u8]) -> Vec<(u8, u32, u32)> {
    let mut out = Vec::new();
    let n = lens.len();
    let run_of = |i: usize, v: u8| lens[i..].iter().take_while(|&&x| x == v).count();
    let mut i = 0;
    while i < n {
        let v = lens[i];
        if v == 0 {
            let run = run_of(i, 0);
            if run >= 11 {
                let r = run.min(138);
                out.push((18, 7, (r - 11) as u32));
                i += r;
            } else if run >= 3 {
                out.push((17, 3, (run - 3) as u32));
                i += run;
            } else {
                out.push((0, 0, 0));
                i += 1;
            }
        } else {
            out.push((v, 0, 0));
            i += 1;
            let mut run = run_of(i, v);
            while run >= 3 {
                let r = run.min(6);
                out.push((16, 2, (r - 3) as u32));
                i += r;
                run -= r;
            }
        }
    }
    out
}

fn write_tokens(
    bw: &mut BitWriter,
    tokens: &[u32],
    lit_codes: &[u16],
    lit_lens: &[u8],
    dist_codes: &[u16],
    dist_lens: &[u8],
) {
    for &t in tokens {
        if t & MATCH_FLAG == 0 {
            let s = t as usize;
            bw.put(u32::from(lit_codes[s]), u32::from(lit_lens[s]));
        } else {
            let len = ((t >> 16) & 0xFF) as usize + MIN_MATCH;
            let dist = (t & 0xFFFF) as usize + 1;
            let lc = LEN_SYM[len] as usize;
            let s = 257 + lc;
            bw.put(u32::from(lit_codes[s]), u32::from(lit_lens[s]));
            bw.put(
                (len - LEN_BASE[lc] as usize) as u32,
                u32::from(LEN_EXTRA[lc]),
            );
            let (dc, eb, ev) = dist_code(dist);
            bw.put(u32::from(dist_codes[dc]), u32::from(dist_lens[dc]));
            bw.put(ev, eb);
        }
    }
    bw.put(u32::from(lit_codes[256]), u32::from(lit_lens[256]));
}

struct Encoder {
    bw: BitWriter,
    fixed_lit_lens: Vec<u8>,
    fixed_lit_codes: Vec<u16>,
    fixed_dist_codes: Vec<u16>,
}

impl Encoder {
    fn new(capacity: usize) -> Encoder {
        let mut lit = vec![8u8; 288];
        lit[144..256].fill(9);
        lit[256..280].fill(7);
        Encoder {
            bw: BitWriter {
                out: Vec::with_capacity(capacity),
                acc: 0,
                nbits: 0,
            },
            fixed_lit_codes: codes_from_lengths(&lit),
            fixed_lit_lens: lit,
            fixed_dist_codes: codes_from_lengths(&[5u8; 30]),
        }
    }

    fn stored(&mut self, raw: &[u8], last: bool) {
        let mut chunks = raw.chunks(MAX_STORED).peekable();
        if chunks.peek().is_none() {
            self.stored_chunk(&[], last);
            return;
        }
        while let Some(c) = chunks.next() {
            self.stored_chunk(c, last && chunks.peek().is_none());
        }
    }

    fn stored_chunk(&mut self, chunk: &[u8], last: bool) {
        self.bw.put(u32::from(last), 1);
        self.bw.put(0, 2);
        self.bw.align();
        let len = chunk.len() as u32;
        self.bw.put(len, 16);
        self.bw.put(!len & 0xFFFF, 16);
        self.bw.out.extend_from_slice(chunk);
    }

    /// Emit one block covering the source bytes `raw`, described by `tokens`.
    fn block(&mut self, tokens: &[u32], raw: &[u8], last: bool, mode: Mode) {
        if mode == Mode::Stored {
            self.stored(raw, last);
            return;
        }
        let mut lit_freq = [0u32; 286];
        let mut dist_freq = [0u32; 30];
        let mut extra_bits = 0u64;
        for &t in tokens {
            if t & MATCH_FLAG == 0 {
                lit_freq[t as usize] += 1;
            } else {
                let len = ((t >> 16) & 0xFF) as usize + MIN_MATCH;
                let dist = (t & 0xFFFF) as usize + 1;
                let lc = LEN_SYM[len] as usize;
                lit_freq[257 + lc] += 1;
                let (dc, eb, _) = dist_code(dist);
                dist_freq[dc] += 1;
                extra_bits += u64::from(LEN_EXTRA[lc]) + u64::from(eb);
            }
        }
        lit_freq[256] = 1;

        let fixed_bits =
            3 + cost(&lit_freq, &self.fixed_lit_lens) + cost(&dist_freq, &[5u8; 30]) + extra_bits;

        let mut lf = lit_freq;
        let mut df = dist_freq;
        ensure_two(&mut lf);
        ensure_two(&mut df);
        let lit_lens = lengths_from_freqs(&lf, 15);
        let dist_lens = lengths_from_freqs(&df, 15);
        let hlit = lit_lens
            .iter()
            .rposition(|&l| l > 0)
            .map_or(257, |p| p + 1)
            .max(257);
        let hdist = dist_lens
            .iter()
            .rposition(|&l| l > 0)
            .map_or(1, |p| p + 1)
            .max(1);
        let mut all = Vec::with_capacity(hlit + hdist);
        all.extend_from_slice(&lit_lens[..hlit]);
        all.extend_from_slice(&dist_lens[..hdist]);
        let rle = rle_lengths(&all);
        let mut cl_freq = [0u32; 19];
        let mut cl_extra = 0u64;
        for &(s, eb, _) in &rle {
            cl_freq[s as usize] += 1;
            cl_extra += u64::from(eb);
        }
        let mut cf = cl_freq;
        ensure_two(&mut cf);
        let cl_lens = lengths_from_freqs(&cf, 7);
        let hclen = CLEN_ORDER
            .iter()
            .rposition(|&s| cl_lens[s] > 0)
            .map_or(4, |p| p + 1)
            .max(4);
        let dyn_bits = 3
            + 14
            + 3 * hclen as u64
            + cost(&cl_freq, &cl_lens)
            + cl_extra
            + cost(&lit_freq, &lit_lens)
            + cost(&dist_freq, &dist_lens)
            + extra_bits;

        let chunks = raw.len().div_ceil(MAX_STORED).max(1) as u64;
        let stored_bits = raw.len() as u64 * 8 + chunks * 35 + 7;

        let use_dynamic = match mode {
            Mode::Fixed => false,
            Mode::Dynamic => true,
            _ => dyn_bits < fixed_bits,
        };
        let chosen = if use_dynamic { dyn_bits } else { fixed_bits };
        if mode == Mode::Auto && stored_bits <= chosen {
            self.stored(raw, last);
            return;
        }

        if use_dynamic {
            let lit_codes = codes_from_lengths(&lit_lens);
            let dist_codes = codes_from_lengths(&dist_lens);
            let cl_codes = codes_from_lengths(&cl_lens);
            self.bw.put(u32::from(last), 1);
            self.bw.put(2, 2);
            self.bw.put((hlit - 257) as u32, 5);
            self.bw.put((hdist - 1) as u32, 5);
            self.bw.put((hclen - 4) as u32, 4);
            for &s in CLEN_ORDER.iter().take(hclen) {
                self.bw.put(u32::from(cl_lens[s]), 3);
            }
            for &(s, eb, ev) in &rle {
                self.bw.put(
                    u32::from(cl_codes[s as usize]),
                    u32::from(cl_lens[s as usize]),
                );
                self.bw.put(ev, eb);
            }
            write_tokens(
                &mut self.bw,
                tokens,
                &lit_codes,
                &lit_lens,
                &dist_codes,
                &dist_lens,
            );
        } else {
            self.bw.put(u32::from(last), 1);
            self.bw.put(1, 2);
            write_tokens(
                &mut self.bw,
                tokens,
                &self.fixed_lit_codes,
                &self.fixed_lit_lens,
                &self.fixed_dist_codes,
                &[5u8; 30],
            );
        }
    }

    fn finish(mut self) -> Vec<u8> {
        self.bw.align();
        self.bw.out
    }
}

/// Hash-chain match finder over the whole input.
struct Matcher<'a> {
    data: &'a [u8],
    /// Most recent position (+1; 0 = none) for each 3-byte hash.
    head: Vec<usize>,
    /// Previous position (+1) with the same hash, indexed by `pos % WSIZE`.
    prev: Vec<usize>,
}

impl<'a> Matcher<'a> {
    fn new(data: &'a [u8]) -> Self {
        Matcher {
            data,
            head: vec![0; 1 << HASH_BITS],
            prev: vec![0; WSIZE],
        }
    }

    #[inline]
    fn hash(&self, i: usize) -> usize {
        let d = self.data;
        let v = u32::from(d[i]) | u32::from(d[i + 1]) << 8 | u32::from(d[i + 2]) << 16;
        (v.wrapping_mul(0x9E37_79B1) >> (32 - HASH_BITS)) as usize
    }

    #[inline]
    fn insert(&mut self, i: usize) {
        if i + MIN_MATCH <= self.data.len() {
            let h = self.hash(i);
            self.prev[i & (WSIZE - 1)] = self.head[h];
            self.head[h] = i + 1;
        }
    }

    fn common_len(&self, a: usize, b: usize, max: usize) -> usize {
        let d = self.data;
        let mut l = 0;
        while l + 8 <= max {
            let x = load64(d, a + l) ^ load64(d, b + l);
            if x != 0 {
                return l + (x.trailing_zeros() / 8) as usize;
            }
            l += 8;
        }
        while l < max && d[a + l] == d[b + l] {
            l += 1;
        }
        l
    }

    /// Longest earlier match for position `i` as `(length, distance)`, then register `i`.
    /// Length 0 means no usable match.
    fn find_insert(&mut self, i: usize) -> (usize, usize) {
        let n = self.data.len();
        if i + MIN_MATCH > n {
            return (0, 0);
        }
        let max_len = MAX_MATCH.min(n - i);
        let mut best_len = MIN_MATCH - 1;
        let mut best_dist = 0;
        let mut cand = self.head[self.hash(i)];
        let mut chain = MAX_CHAIN;
        while cand != 0 && chain > 0 {
            let c = cand - 1;
            if i - c > WSIZE {
                break;
            }
            if self.data[c + best_len] == self.data[i + best_len] {
                let l = self.common_len(c, i, max_len);
                if l > best_len {
                    best_len = l;
                    best_dist = i - c;
                    if l >= NICE_MATCH.min(max_len) {
                        break;
                    }
                }
            }
            let next = self.prev[c & (WSIZE - 1)];
            if next == 0 || next > c {
                break;
            }
            cand = next;
            chain -= 1;
        }
        self.insert(i);
        if best_dist == 0 || (best_len == MIN_MATCH && best_dist > TOO_FAR) {
            return (0, 0);
        }
        (best_len, best_dist)
    }
}

#[inline]
fn load64(d: &[u8], i: usize) -> u64 {
    u64::from_le_bytes([
        d[i],
        d[i + 1],
        d[i + 2],
        d[i + 3],
        d[i + 4],
        d[i + 5],
        d[i + 6],
        d[i + 7],
    ])
}

fn match_token(len: usize, dist: usize) -> u32 {
    MATCH_FLAG | (((len - MIN_MATCH) as u32) << 16) | (dist - 1) as u32
}

/// Compress `data` into a raw DEFLATE stream, choosing the best block type per block.
pub fn deflate(data: &[u8]) -> Vec<u8> {
    deflate_with(data, Mode::Auto)
}

/// Compress `data` into a raw DEFLATE stream using the given block-type policy.
pub fn deflate_with(data: &[u8], mode: Mode) -> Vec<u8> {
    let n = data.len();
    let mut enc = Encoder::new(n / 2 + 16);
    if mode == Mode::Stored {
        enc.stored(data, true);
        return enc.finish();
    }
    let mut m = Matcher::new(data);
    let mut tokens: Vec<u32> = Vec::with_capacity(BLOCK_TOKENS.min(n) + 1);
    let mut block_start = 0;
    let mut i = 0;
    let mut cur = if n > 0 { m.find_insert(0) } else { (0, 0) };
    while i < n {
        if tokens.len() >= BLOCK_TOKENS {
            enc.block(&tokens, &data[block_start..i], false, mode);
            tokens.clear();
            block_start = i;
        }
        let (len, dist) = cur;
        if len < MIN_MATCH {
            tokens.push(u32::from(data[i]));
            i += 1;
            cur = if i < n { m.find_insert(i) } else { (0, 0) };
            continue;
        }
        if len < LAZY_MAX && i + 1 < n {
            let nxt = m.find_insert(i + 1);
            if nxt.0 > len {
                tokens.push(u32::from(data[i]));
                i += 1;
                cur = nxt;
                continue;
            }
            tokens.push(match_token(len, dist));
            for p in i + 2..i + len {
                m.insert(p);
            }
        } else {
            tokens.push(match_token(len, dist));
            for p in i + 1..i + len {
                m.insert(p);
            }
        }
        i += len;
        cur = if i < n { m.find_insert(i) } else { (0, 0) };
    }
    enc.block(&tokens, &data[block_start..], true, mode);
    enc.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::npz::inflate::{DIST_BASE, DIST_EXTRA, inflate};
    use crate::npz::testutil::{self, Python};

    fn roundtrip(data: &[u8], mode: Mode) -> usize {
        let z = deflate_with(data, mode);
        let back = inflate(&z).unwrap_or_else(|e| panic!("{mode:?} len {}: {e}", data.len()));
        assert!(
            back == data,
            "{mode:?} roundtrip differs for len {}",
            data.len()
        );
        z.len()
    }

    const MODES: [Mode; 4] = [Mode::Stored, Mode::Fixed, Mode::Dynamic, Mode::Auto];

    #[test]
    fn empty_input() {
        for m in MODES {
            roundtrip(&[], m);
        }
        assert_eq!(deflate(&[]), [0x03, 0x00]);
    }

    #[test]
    fn single_byte_and_tiny_inputs() {
        for m in MODES {
            for n in 1..=12usize {
                roundtrip(&vec![0x41; n], m);
                roundtrip(&testutil::pseudo_random(n, n as u64), m);
            }
        }
    }

    #[test]
    fn random_data_every_mode() {
        for (len, seed) in [(100, 1), (4096, 2), (70_000, 3), (200_000, 4)] {
            let d = testutil::pseudo_random(len, seed);
            for m in MODES {
                roundtrip(&d, m);
            }
        }
    }

    #[test]
    fn random_data_stays_near_input_size_in_auto_mode() {
        let d = testutil::pseudo_random(100_000, 11);
        let n = roundtrip(&d, Mode::Auto);
        assert!(n <= d.len() + d.len() / 1000 + 32, "auto expanded to {n}");
    }

    #[test]
    fn highly_compressible_data() {
        let zeros = vec![0u8; 300_000];
        let n = roundtrip(&zeros, Mode::Auto);
        assert!(n < 1000, "300 KB of zeros took {n} bytes");
        let text: Vec<u8> = b"the cat sat on the mat. "
            .iter()
            .copied()
            .cycle()
            .take(100_000)
            .collect();
        let n = roundtrip(&text, Mode::Auto);
        assert!(n < 1500, "periodic text took {n} bytes");
    }

    #[test]
    fn data_larger_than_one_stored_block_and_window() {
        // > 131072 bytes: several stored chunks and several LZ77 blocks.
        let mut d = Vec::new();
        for i in 0..40_000u32 {
            d.extend_from_slice(&(i % 1000).to_le_bytes());
        }
        assert!(d.len() > 131_072);
        for m in MODES {
            roundtrip(&d, m);
        }
    }

    #[test]
    fn all_byte_values_and_runs() {
        let mut d: Vec<u8> = (0..=255u8).collect();
        d.extend((0..=255u8).rev());
        for b in 0..=255u8 {
            d.extend(std::iter::repeat_n(b, (b as usize % 7) + 1));
        }
        for m in MODES {
            roundtrip(&d, m);
        }
    }

    #[test]
    fn matches_at_max_distance_and_max_length() {
        // The same pattern exactly 32768 bytes apart exercises distance 32768.
        let pattern = testutil::pseudo_random(300, 5);
        let mut d = pattern.clone();
        d.extend(testutil::pseudo_random(32768 - 300, 6));
        d.extend_from_slice(&pattern);
        d.extend(vec![9u8; 1000]);
        for m in MODES {
            roundtrip(&d, m);
        }
    }

    #[test]
    fn dynamic_beats_fixed_on_skewed_data() {
        let d: Vec<u8> = testutil::pseudo_random(60_000, 8)
            .iter()
            .map(|b| if *b < 200 { b'e' } else { b % 8 + b'a' })
            .collect();
        let dynamic = roundtrip(&d, Mode::Dynamic);
        let fixed = roundtrip(&d, Mode::Fixed);
        let auto = roundtrip(&d, Mode::Auto);
        assert!(dynamic < fixed, "dynamic {dynamic} fixed {fixed}");
        assert!(auto <= dynamic.min(fixed) + 2);
    }

    #[test]
    fn length_limited_codes_form_a_complete_prefix_code() {
        // Fibonacci-like frequencies force very deep trees; the limit must hold.
        let mut f = vec![0u32; 40];
        let (mut a, mut b) = (1u32, 1u32);
        for x in f.iter_mut() {
            *x = a;
            let c = a.saturating_add(b);
            a = b;
            b = c;
        }
        for limit in [15usize, 7] {
            let lens = lengths_from_freqs(&f, limit);
            assert!(lens.iter().all(|&l| l >= 1 && usize::from(l) <= limit));
            let kraft: u64 = lens.iter().map(|&l| 1u64 << (limit - usize::from(l))).sum();
            assert_eq!(kraft, 1u64 << limit, "limit {limit}");
        }
    }

    #[test]
    fn distance_codes_match_the_rfc_table() {
        assert_eq!(dist_code(1), (0, 0, 0));
        assert_eq!(dist_code(4), (3, 0, 0));
        assert_eq!(dist_code(5), (4, 1, 0));
        assert_eq!(dist_code(6), (4, 1, 1));
        assert_eq!(dist_code(7), (5, 1, 0));
        assert_eq!(dist_code(24577), (29, 13, 0));
        assert_eq!(dist_code(32768), (29, 13, 8191));
        for d in 1..=32768usize {
            let (c, eb, ev) = dist_code(d);
            assert_eq!(DIST_BASE[c] as usize + ev as usize, d);
            assert_eq!(u32::from(DIST_EXTRA[c]), eb);
        }
    }

    #[test]
    fn length_symbol_table_matches_the_rfc() {
        for (len, &sym) in LEN_SYM.iter().enumerate().skip(3) {
            let c = sym as usize;
            let base = LEN_BASE[c] as usize;
            assert!(
                len >= base && len - base < (1usize << LEN_EXTRA[c]),
                "len {len}"
            );
        }
        assert_eq!(LEN_SYM[258], 28);
        assert_eq!(LEN_SYM[3], 0);
    }

    const PY_DECOMPRESS: &str = "import sys, zlib, struct
b = sys.stdin.buffer.read()
n = struct.unpack('<I', b[:4])[0]
sys.stdout.buffer.write(zlib.decompress(b[4:4+n], -15))";

    #[test]
    fn python_zlib_decompresses_our_streams() {
        let Some(py) = Python::get() else { return };
        let inputs: Vec<Vec<u8>> = vec![
            vec![],
            vec![7],
            testutil::pseudo_random(90_000, 21),
            vec![0; 200_000],
            b"abcabcabcabc".repeat(10_000),
            (0..30_000u32)
                .flat_map(|i| ((i % 313) as f32 * 0.25).to_le_bytes())
                .collect(),
        ];
        for (i, data) in inputs.iter().enumerate() {
            for m in MODES {
                let z = deflate_with(data, m);
                let mut stdin = (z.len() as u32).to_le_bytes().to_vec();
                stdin.extend_from_slice(&z);
                let out = py.run(PY_DECOMPRESS, &[], &stdin);
                assert!(out == *data, "input {i} mode {m:?}");
            }
        }
    }

    #[test]
    fn compression_ratio_is_close_to_zlib() {
        let Some(py) = Python::get() else { return };
        // Mixed sample: smooth floats, small integers, repeated text.
        let mut d: Vec<u8> = (0..20_000)
            .flat_map(|i| (f64::from(i) * 0.01).sin().to_le_bytes())
            .collect();
        d.extend((0..40_000u32).flat_map(|i| (i % 211).to_le_bytes()));
        d.extend(b"it was the best of times, it was the worst of times. ".repeat(500));
        let ours = deflate(&d).len();
        let out = py.run(
            "import sys, zlib
d = sys.stdin.buffer.read()
c = zlib.compressobj(6, zlib.DEFLATED, -15)
print(len(c.compress(d) + c.flush()))",
            &[],
            &d,
        );
        let zlib_len: usize = String::from_utf8(out).unwrap().trim().parse().unwrap();
        eprintln!(
            "deflate ratio sample: input {} ours {ours} zlib-6 {zlib_len}",
            d.len()
        );
        assert!(
            (ours as f64) < zlib_len as f64 * 1.25,
            "ours {ours} vs zlib {zlib_len}"
        );
    }

    /// Run with `cargo test --release -- --ignored --nocapture bench_report`.
    #[test]
    #[ignore]
    fn bench_report() {
        let Some(py) = Python::get() else { return };
        let datasets: Vec<(&str, Vec<u8>)> = vec![
            (
                "sin f64",
                (0..200_000)
                    .flat_map(|i| (f64::from(i) * 0.001).sin().to_le_bytes())
                    .collect(),
            ),
            ("random bytes", testutil::pseudo_random(1_000_000, 3)),
            (
                "int64 mod 97",
                (0..200_000i64)
                    .flat_map(|i| (i % 97).to_le_bytes())
                    .collect(),
            ),
            ("zeros", vec![0; 2_000_000]),
            (
                "text",
                b"it was the best of times, it was the worst of times, ".repeat(20_000),
            ),
        ];
        for (name, d) in &datasets {
            let t = std::time::Instant::now();
            let ours = deflate(d);
            let secs = t.elapsed().as_secs_f64();
            let t2 = std::time::Instant::now();
            let _ = inflate(&ours).unwrap();
            let isecs = t2.elapsed().as_secs_f64();
            let out = py.run(
                "import sys, zlib
d = sys.stdin.buffer.read()
print(len(zlib.compress(d, 6)), len(zlib.compress(d, 9)))",
                &[],
                d,
            );
            println!(
                "{name:14} in {:8} ours {:8} zlib6/9 {} | deflate {:.0} MB/s inflate {:.0} MB/s",
                d.len(),
                ours.len(),
                String::from_utf8(out).unwrap().trim(),
                d.len() as f64 / secs / 1e6,
                d.len() as f64 / isecs / 1e6
            );
        }
    }
}
