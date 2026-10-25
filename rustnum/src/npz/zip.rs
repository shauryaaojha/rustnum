//! ZIP archive reader and writer (PKWARE APPNOTE 6.3.x subset).
//!
//! Reading: end-of-central-directory (with zip64 locator and record), central directory,
//! local headers, data descriptors, methods 0 (stored) and 8 (deflate), CRC-32 and size
//! verification. When there is no central directory at all (a truncated or streamed
//! archive), [`read_zip`] falls back to scanning local headers.
//!
//! Writing: stored or deflated members, central directory, zip64 structures when a size,
//! offset or entry count does not fit the classic fields.

use std::borrow::Cow;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::sync::Mutex;

use super::NpzError;
use super::crc32::crc32;
use super::deflate::deflate;
use super::inflate::inflate_with_limit;

const LOCAL_SIG: u32 = 0x0403_4B50;
const CENTRAL_SIG: u32 = 0x0201_4B50;
const EOCD_SIG: u32 = 0x0605_4B50;
const EOCD64_SIG: u32 = 0x0606_4B50;
const LOCATOR64_SIG: u32 = 0x0706_4B50;
const DESCRIPTOR_SIG: u32 = 0x0807_4B50;

const EOCD_LEN: usize = 22;
const MAX_U16: u64 = 0xFFFF;
const MAX_U32: u64 = 0xFFFF_FFFF;
/// MS-DOS date for 1980-01-01, the same fixed timestamp NumPy writes.
const DOS_DATE: u16 = 0x0021;

fn corrupt(msg: &str) -> NpzError {
    NpzError::Corrupt(msg.to_string())
}

// ---------------------------------------------------------------------------------------
// Random-access byte sources
// ---------------------------------------------------------------------------------------

/// Something archive bytes can be read from at arbitrary offsets.
pub(super) trait ByteSource {
    /// Total size in bytes.
    fn size(&self) -> Result<u64, NpzError>;
    /// Exactly `n` bytes starting at `offset`; a range past the end is a truncation error.
    fn read_at(&self, offset: u64, n: usize) -> Result<Cow<'_, [u8]>, NpzError>;
}

impl ByteSource for [u8] {
    fn size(&self) -> Result<u64, NpzError> {
        Ok(self.len() as u64)
    }

    fn read_at(&self, offset: u64, n: usize) -> Result<Cow<'_, [u8]>, NpzError> {
        let start = usize::try_from(offset).map_err(|_| NpzError::Truncated)?;
        let end = start.checked_add(n).ok_or(NpzError::Truncated)?;
        self.get(start..end)
            .map(Cow::Borrowed)
            .ok_or(NpzError::Truncated)
    }
}

impl ByteSource for Mutex<File> {
    fn size(&self) -> Result<u64, NpzError> {
        let f = self.lock().map_err(|_| corrupt("file lock poisoned"))?;
        Ok(f.metadata()?.len())
    }

    fn read_at(&self, offset: u64, n: usize) -> Result<Cow<'_, [u8]>, NpzError> {
        let mut f = self.lock().map_err(|_| corrupt("file lock poisoned"))?;
        let size = f.metadata()?.len();
        if offset.checked_add(n as u64).is_none_or(|e| e > size) {
            return Err(NpzError::Truncated);
        }
        f.seek(SeekFrom::Start(offset))?;
        let mut buf = vec![0u8; n];
        f.read_exact(&mut buf)?;
        Ok(Cow::Owned(buf))
    }
}

// ---------------------------------------------------------------------------------------
// Little-endian field access
// ---------------------------------------------------------------------------------------

fn u16_at(b: &[u8], o: usize) -> Result<u16, NpzError> {
    b.get(o..o + 2)
        .map(|s| u16::from_le_bytes([s[0], s[1]]))
        .ok_or(NpzError::Truncated)
}

fn u32_at(b: &[u8], o: usize) -> Result<u32, NpzError> {
    b.get(o..o + 4)
        .map(|s| u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
        .ok_or(NpzError::Truncated)
}

fn u64_at(b: &[u8], o: usize) -> Result<u64, NpzError> {
    b.get(o..o + 8)
        .map(|s| u64::from_le_bytes([s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]]))
        .ok_or(NpzError::Truncated)
}

// ---------------------------------------------------------------------------------------
// Reading
// ---------------------------------------------------------------------------------------

/// One central-directory record.
#[derive(Clone, Debug)]
pub(super) struct ZipEntry {
    /// Member name (UTF-8, invalid bytes replaced).
    pub name: String,
    name_raw: Vec<u8>,
    method: u16,
    crc: u32,
    comp_size: u64,
    size: u64,
    local_offset: u64,
}

impl ZipEntry {
    /// Uncompressed size from the central directory.
    pub fn uncompressed_size(&self) -> u64 {
        self.size
    }
}

/// Zip64 extended-information fields, in the order the specification lists them.
struct Zip64Extra {
    size: Option<u64>,
    comp_size: Option<u64>,
    offset: Option<u64>,
}

/// Parse an extra-field blob and pull out the zip64 block (id 0x0001).
///
/// Only fields whose classic counterpart is `0xFFFFFFFF` are present, so the caller says
/// which ones it expects.
fn parse_zip64_extra(
    extra: &[u8],
    want_size: bool,
    want_comp: bool,
    want_offset: bool,
) -> Result<Zip64Extra, NpzError> {
    let mut out = Zip64Extra {
        size: None,
        comp_size: None,
        offset: None,
    };
    let mut p = 0;
    while p + 4 <= extra.len() {
        let id = u16_at(extra, p)?;
        let len = usize::from(u16_at(extra, p + 2)?);
        let body = extra
            .get(p + 4..p + 4 + len)
            .ok_or_else(|| corrupt("extra field overruns its record"))?;
        if id == 0x0001 {
            let mut q = 0;
            if want_size {
                out.size = Some(u64_at(body, q)?);
                q += 8;
            }
            if want_comp {
                out.comp_size = Some(u64_at(body, q)?);
                q += 8;
            }
            if want_offset {
                out.offset = Some(u64_at(body, q)?);
            }
            return Ok(out);
        }
        p += 4 + len;
    }
    Ok(out)
}

/// Locate and read the central directory of an archive.
///
/// Returns `NpzError::NotAZip` when no end-of-central-directory record exists.
pub(super) fn read_index<S: ByteSource + ?Sized>(src: &S) -> Result<Vec<ZipEntry>, NpzError> {
    let size = src.size()?;
    if size < EOCD_LEN as u64 {
        return Err(NpzError::NotAZip);
    }
    let tail_len = size.min((EOCD_LEN + 0xFFFF) as u64) as usize;
    let tail_start = size - tail_len as u64;
    let tail = src.read_at(tail_start, tail_len)?;
    let sig = EOCD_SIG.to_le_bytes();
    let eocd = (0..=tail.len() - EOCD_LEN)
        .rev()
        .find(|&i| {
            tail[i..i + 4] == sig
                && u16_at(&tail, i + 20).is_ok_and(|c| i + EOCD_LEN + usize::from(c) <= tail.len())
        })
        .ok_or(NpzError::NotAZip)?;
    let eocd_abs = tail_start + eocd as u64;
    let disk = u16_at(&tail, eocd + 4)?;
    let cd_disk = u16_at(&tail, eocd + 6)?;
    let mut total = u64::from(u16_at(&tail, eocd + 10)?);
    let mut cd_size = u64::from(u32_at(&tail, eocd + 12)?);
    let mut cd_off = u64::from(u32_at(&tail, eocd + 16)?);
    // Where the central directory should end if nothing is prepended to the archive.
    let mut cd_end_expected = eocd_abs;

    let needs64 = total == MAX_U16 || cd_size == MAX_U32 || cd_off == MAX_U32;
    if needs64 && eocd_abs >= 20 {
        let loc = src.read_at(eocd_abs - 20, 20)?;
        if u32_at(&loc, 0)? == LOCATOR64_SIG {
            let rec_off = u64_at(&loc, 8)?;
            let rec = src.read_at(rec_off, 56)?;
            if u32_at(&rec, 0)? != EOCD64_SIG {
                return Err(corrupt("bad zip64 end-of-central-directory signature"));
            }
            if u32_at(&rec, 16)? != 0 || u32_at(&rec, 20)? != 0 {
                return Err(NpzError::Unsupported("multi-disk archives".into()));
            }
            total = u64_at(&rec, 32)?;
            cd_size = u64_at(&rec, 40)?;
            cd_off = u64_at(&rec, 48)?;
            cd_end_expected = rec_off;
        }
    } else if disk != 0 || cd_disk != 0 {
        return Err(NpzError::Unsupported("multi-disk archives".into()));
    }

    // Tolerate data prepended to the archive (self-extractors): offsets are then shifted.
    let mut prefix = 0u64;
    if cd_size > 0 {
        let has_sig = |off: u64| {
            src.read_at(off, 4)
                .is_ok_and(|b| u32_at(&b, 0).is_ok_and(|s| s == CENTRAL_SIG))
        };
        if !has_sig(cd_off)
            && let Some(adj) = cd_end_expected.checked_sub(cd_size)
            && adj > cd_off
            && has_sig(adj)
        {
            prefix = adj - cd_off;
        }
    }

    let cd_len = usize::try_from(cd_size).map_err(|_| NpzError::Truncated)?;
    let cd = src.read_at(
        cd_off.checked_add(prefix).ok_or(NpzError::Truncated)?,
        cd_len,
    )?;
    if total > (cd.len() / 46) as u64 {
        return Err(corrupt("central directory too small for its entry count"));
    }
    let mut entries = Vec::with_capacity(total as usize);
    let mut p = 0usize;
    for _ in 0..total {
        if u32_at(&cd, p)? != CENTRAL_SIG {
            return Err(corrupt("bad central directory signature"));
        }
        let flags = u16_at(&cd, p + 8)?;
        let method = u16_at(&cd, p + 10)?;
        let crc = u32_at(&cd, p + 16)?;
        let mut comp_size = u64::from(u32_at(&cd, p + 20)?);
        let mut usize_ = u64::from(u32_at(&cd, p + 24)?);
        let name_len = usize::from(u16_at(&cd, p + 28)?);
        let extra_len = usize::from(u16_at(&cd, p + 30)?);
        let comment_len = usize::from(u16_at(&cd, p + 32)?);
        let entry_disk = u16_at(&cd, p + 34)?;
        let mut local_offset = u64::from(u32_at(&cd, p + 42)?);
        let name_raw = cd
            .get(p + 46..p + 46 + name_len)
            .ok_or(NpzError::Truncated)?
            .to_vec();
        let extra = cd
            .get(p + 46 + name_len..p + 46 + name_len + extra_len)
            .ok_or(NpzError::Truncated)?;
        let next = p + 46 + name_len + extra_len + comment_len;
        if next > cd.len() {
            return Err(NpzError::Truncated);
        }
        if entry_disk != 0 && entry_disk != 0xFFFF {
            return Err(NpzError::Unsupported("multi-disk archives".into()));
        }
        let want_size = usize_ == MAX_U32;
        let want_comp = comp_size == MAX_U32;
        let want_off = local_offset == MAX_U32;
        if want_size || want_comp || want_off {
            let z = parse_zip64_extra(extra, want_size, want_comp, want_off)?;
            if want_size {
                usize_ = z.size.ok_or_else(|| corrupt("missing zip64 size"))?;
            }
            if want_comp {
                comp_size = z.comp_size.ok_or_else(|| corrupt("missing zip64 size"))?;
            }
            if want_off {
                local_offset = z.offset.ok_or_else(|| corrupt("missing zip64 offset"))?;
            }
        }
        let name = String::from_utf8_lossy(&name_raw).into_owned();
        p = next;
        if flags & 1 != 0 {
            return Err(NpzError::Unsupported(format!("encrypted member {name:?}")));
        }
        if name.ends_with('/') && usize_ == 0 {
            continue; // directory entry
        }
        entries.push(ZipEntry {
            name,
            name_raw,
            method,
            crc,
            comp_size,
            size: usize_,
            local_offset: local_offset
                .checked_add(prefix)
                .ok_or_else(|| corrupt("local header offset overflow"))?,
        });
    }
    Ok(entries)
}

/// Decompress (if needed) and verify one member's data.
fn decode_member(
    name: &str,
    method: u16,
    crc: u32,
    size: u64,
    raw: &[u8],
) -> Result<Vec<u8>, NpzError> {
    let data = match method {
        0 => {
            if raw.len() as u64 != size {
                return Err(NpzError::SizeMismatch {
                    name: name.to_string(),
                    expected: size,
                    found: raw.len() as u64,
                });
            }
            raw.to_vec()
        }
        8 => {
            let limit = usize::try_from(size).unwrap_or(usize::MAX);
            let (out, _) = inflate_with_limit(raw, limit)?;
            if out.len() as u64 != size {
                return Err(NpzError::SizeMismatch {
                    name: name.to_string(),
                    expected: size,
                    found: out.len() as u64,
                });
            }
            out
        }
        m => {
            return Err(NpzError::Unsupported(format!(
                "compression method {m} in member {name:?}"
            )));
        }
    };
    let found = crc32(&data);
    if found != crc {
        return Err(NpzError::Crc {
            name: name.to_string(),
            expected: crc,
            found,
        });
    }
    Ok(data)
}

/// Read and verify one member through its local header.
pub(super) fn read_entry<S: ByteSource + ?Sized>(
    src: &S,
    e: &ZipEntry,
) -> Result<Vec<u8>, NpzError> {
    let hdr = src.read_at(e.local_offset, 30)?;
    if u32_at(&hdr, 0)? != LOCAL_SIG {
        return Err(corrupt("bad local header signature"));
    }
    let name_len = usize::from(u16_at(&hdr, 26)?);
    let extra_len = u64::from(u16_at(&hdr, 28)?);
    let name_off = e.local_offset.checked_add(30).ok_or(NpzError::Truncated)?;
    if name_len > 0 {
        let local_name = src.read_at(name_off, name_len)?;
        if *local_name != e.name_raw[..] {
            return Err(corrupt(
                "member name differs between local and central headers",
            ));
        }
    } else if !e.name_raw.is_empty() {
        return Err(corrupt(
            "member name differs between local and central headers",
        ));
    }
    let data_off = name_off
        .checked_add(name_len as u64 + extra_len)
        .ok_or(NpzError::Truncated)?;
    let comp = usize::try_from(e.comp_size).map_err(|_| NpzError::Truncated)?;
    let raw = src.read_at(data_off, comp)?;
    decode_member(&e.name, e.method, e.crc, e.size, &raw)
}

/// Fallback for archives without a central directory: walk the local headers in order.
fn scan_local(data: &[u8]) -> Result<Vec<(String, Vec<u8>)>, NpzError> {
    let mut out = Vec::new();
    let mut pos = 0usize;
    while pos + 4 <= data.len() {
        if u32_at(data, pos)? != LOCAL_SIG {
            break; // central directory, end record, or junk after the last member
        }
        let flags = u16_at(data, pos + 6)?;
        let method = u16_at(data, pos + 8)?;
        let crc = u32_at(data, pos + 14)?;
        let mut comp_size = u64::from(u32_at(data, pos + 18)?);
        let mut size = u64::from(u32_at(data, pos + 22)?);
        let name_len = usize::from(u16_at(data, pos + 26)?);
        let extra_len = usize::from(u16_at(data, pos + 28)?);
        let name_raw = data
            .get(pos + 30..pos + 30 + name_len)
            .ok_or(NpzError::Truncated)?;
        let extra = data
            .get(pos + 30 + name_len..pos + 30 + name_len + extra_len)
            .ok_or(NpzError::Truncated)?;
        let name = String::from_utf8_lossy(name_raw).into_owned();
        let start = pos + 30 + name_len + extra_len;
        if flags & 1 != 0 {
            return Err(NpzError::Unsupported(format!("encrypted member {name:?}")));
        }
        if flags & 8 != 0 {
            // Sizes and CRC follow the data; only a deflate stream knows where it ends.
            if method != 8 {
                return Err(NpzError::Unsupported(
                    "streamed stored member without a central directory".into(),
                ));
            }
            let rest = data.get(start..).ok_or(NpzError::Truncated)?;
            let (bytes, used) = inflate_with_limit(rest, usize::MAX)?;
            let mut p = start + used;
            if u32_at(data, p)? == DESCRIPTOR_SIG {
                p += 4;
            }
            let d_crc = u32_at(data, p)?;
            let (d32c, d32u) = (
                u64::from(u32_at(data, p + 4)?),
                u64::from(u32_at(data, p + 8)?),
            );
            if d32c == used as u64 && d32u == bytes.len() as u64 {
                p += 12;
            } else if u64_at(data, p + 4)? == used as u64
                && u64_at(data, p + 12)? == bytes.len() as u64
            {
                p += 20;
            } else {
                return Err(corrupt("data descriptor does not match the member data"));
            }
            let found = crc32(&bytes);
            if found != d_crc {
                return Err(NpzError::Crc {
                    name,
                    expected: d_crc,
                    found,
                });
            }
            out.push((name, bytes));
            pos = p;
        } else {
            let want_size = size == MAX_U32;
            let want_comp = comp_size == MAX_U32;
            if want_size || want_comp {
                let z = parse_zip64_extra(extra, want_size, want_comp, false)?;
                size = z.size.unwrap_or(size);
                comp_size = z.comp_size.unwrap_or(comp_size);
            }
            let comp = usize::try_from(comp_size).map_err(|_| NpzError::Truncated)?;
            let end = start.checked_add(comp).ok_or(NpzError::Truncated)?;
            let raw = data.get(start..end).ok_or(NpzError::Truncated)?;
            out.push((name.clone(), decode_member(&name, method, crc, size, raw)?));
            pos = end;
        }
    }
    if out.is_empty() {
        return Err(NpzError::NotAZip);
    }
    // A clean archive continues with the central directory (or end record). Running off the
    // end instead is indistinguishable from a truncated file, so refuse it.
    match u32_at(data, pos) {
        Ok(CENTRAL_SIG | EOCD_SIG) => Ok(out),
        _ => Err(NpzError::Truncated),
    }
}

/// Read every member of a ZIP archive held in memory.
///
/// Members come back in central-directory order as `(name, bytes)`. Directory entries are
/// skipped. Each member's CRC-32 and size are verified.
pub fn read_zip(bytes: &[u8]) -> Result<Vec<(String, Vec<u8>)>, NpzError> {
    match read_index(bytes) {
        Ok(entries) => entries
            .iter()
            .map(|e| read_entry(bytes, e).map(|d| (e.name.clone(), d)))
            .collect(),
        Err(NpzError::NotAZip) => scan_local(bytes),
        Err(e) => Err(e),
    }
}

// ---------------------------------------------------------------------------------------
// Writing
// ---------------------------------------------------------------------------------------

fn put16(out: &mut Vec<u8>, v: u16) {
    out.extend_from_slice(&v.to_le_bytes());
}
fn put32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}
fn put64(out: &mut Vec<u8>, v: u64) {
    out.extend_from_slice(&v.to_le_bytes());
}

/// Write a ZIP archive.
///
/// With `compress`, each member is deflated when that makes it smaller and stored otherwise;
/// without it every member is stored. Zip64 structures are added where a size, offset or the
/// entry count needs them.
///
/// # Panics
/// If a member name is longer than 65535 bytes (the `npz` layer checks this beforehand).
pub fn write_zip(members: &[(String, Vec<u8>)], compress: bool) -> Vec<u8> {
    write_zip_impl(members, compress, false)
}

/// [`write_zip`] with an option to force zip64 records for every member (for testing and
/// for readers that must be exercised without 4 GiB of data).
pub(super) fn write_zip_impl(
    members: &[(String, Vec<u8>)],
    compress: bool,
    force_zip64: bool,
) -> Vec<u8> {
    struct Central {
        name: Vec<u8>,
        method: u16,
        crc: u32,
        comp: u64,
        size: u64,
        offset: u64,
        zip64: bool,
        flags: u16,
    }
    let total: usize = members.iter().map(|(n, d)| n.len() + d.len() + 100).sum();
    let mut out: Vec<u8> = Vec::with_capacity(total);
    let mut central: Vec<Central> = Vec::with_capacity(members.len());

    for (name, data) in members {
        let name_bytes = name.as_bytes();
        assert!(
            name_bytes.len() <= 0xFFFF,
            "zip member name longer than 65535 bytes"
        );
        let deflated = if compress && !data.is_empty() {
            let d = deflate(data);
            (d.len() < data.len()).then_some(d)
        } else {
            None
        };
        let (method, payload): (u16, &[u8]) = match &deflated {
            Some(d) => (8, d),
            None => (0, data),
        };
        let size = data.len() as u64;
        let comp = payload.len() as u64;
        let zip64 = force_zip64 || size >= MAX_U32 || comp >= MAX_U32;
        let flags: u16 = if name.is_ascii() { 0 } else { 0x0800 };
        let crc = crc32(data);
        let offset = out.len() as u64;

        put32(&mut out, LOCAL_SIG);
        put16(
            &mut out,
            if zip64 {
                45
            } else if method == 8 {
                20
            } else {
                10
            },
        );
        put16(&mut out, flags);
        put16(&mut out, method);
        put16(&mut out, 0); // time
        put16(&mut out, DOS_DATE);
        put32(&mut out, crc);
        put32(&mut out, if zip64 { MAX_U32 as u32 } else { comp as u32 });
        put32(&mut out, if zip64 { MAX_U32 as u32 } else { size as u32 });
        put16(&mut out, name_bytes.len() as u16);
        put16(&mut out, if zip64 { 20 } else { 0 });
        out.extend_from_slice(name_bytes);
        if zip64 {
            put16(&mut out, 0x0001);
            put16(&mut out, 16);
            put64(&mut out, size);
            put64(&mut out, comp);
        }
        out.extend_from_slice(payload);
        central.push(Central {
            name: name_bytes.to_vec(),
            method,
            crc,
            comp,
            size,
            offset,
            zip64,
            flags,
        });
    }

    let cd_offset = out.len() as u64;
    for c in &central {
        let off_big = force_zip64 || c.offset >= MAX_U32;
        let extra_len = 8 * (2 * usize::from(c.zip64) + usize::from(off_big));
        let needed: u16 = if c.zip64 || off_big {
            45
        } else if c.method == 8 {
            20
        } else {
            10
        };
        put32(&mut out, CENTRAL_SIG);
        put16(&mut out, needed); // version made by
        put16(&mut out, needed); // version needed
        put16(&mut out, c.flags);
        put16(&mut out, c.method);
        put16(&mut out, 0);
        put16(&mut out, DOS_DATE);
        put32(&mut out, c.crc);
        put32(
            &mut out,
            if c.zip64 {
                MAX_U32 as u32
            } else {
                c.comp as u32
            },
        );
        put32(
            &mut out,
            if c.zip64 {
                MAX_U32 as u32
            } else {
                c.size as u32
            },
        );
        put16(&mut out, c.name.len() as u16);
        put16(
            &mut out,
            if extra_len > 0 {
                extra_len as u16 + 4
            } else {
                0
            },
        );
        put16(&mut out, 0); // comment length
        put16(&mut out, 0); // disk number
        put16(&mut out, 0); // internal attributes
        put32(&mut out, 0); // external attributes
        put32(
            &mut out,
            if off_big {
                MAX_U32 as u32
            } else {
                c.offset as u32
            },
        );
        out.extend_from_slice(&c.name);
        if extra_len > 0 {
            put16(&mut out, 0x0001);
            put16(&mut out, extra_len as u16);
            if c.zip64 {
                put64(&mut out, c.size);
                put64(&mut out, c.comp);
            }
            if off_big {
                put64(&mut out, c.offset);
            }
        }
    }
    let cd_size = out.len() as u64 - cd_offset;
    let count = central.len() as u64;

    let zip64_end = force_zip64 || count >= MAX_U16 || cd_size >= MAX_U32 || cd_offset >= MAX_U32;
    if zip64_end {
        let rec_offset = out.len() as u64;
        put32(&mut out, EOCD64_SIG);
        put64(&mut out, 44);
        put16(&mut out, 45);
        put16(&mut out, 45);
        put32(&mut out, 0);
        put32(&mut out, 0);
        put64(&mut out, count);
        put64(&mut out, count);
        put64(&mut out, cd_size);
        put64(&mut out, cd_offset);
        put32(&mut out, LOCATOR64_SIG);
        put32(&mut out, 0);
        put64(&mut out, rec_offset);
        put32(&mut out, 1);
    }
    put32(&mut out, EOCD_SIG);
    put16(&mut out, 0);
    put16(&mut out, 0);
    let c16 = if zip64_end { 0xFFFF } else { count as u16 };
    put16(&mut out, c16);
    put16(&mut out, c16);
    put32(
        &mut out,
        if zip64_end {
            MAX_U32 as u32
        } else {
            cd_size as u32
        },
    );
    put32(
        &mut out,
        if zip64_end {
            MAX_U32 as u32
        } else {
            cd_offset as u32
        },
    );
    put16(&mut out, 0);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::npz::testutil::{self, Python, TempDir};

    fn members(n: usize) -> Vec<(String, Vec<u8>)> {
        (0..n)
            .map(|i| {
                let mut d = testutil::pseudo_random(300 + i * 17, i as u64);
                d.extend(std::iter::repeat_n(b'z', 1000));
                (format!("m{i}.npy"), d)
            })
            .collect()
    }

    #[test]
    fn empty_archive_roundtrip() {
        let z = write_zip(&[], true);
        assert_eq!(z.len(), 22);
        assert!(read_zip(&z).unwrap().is_empty());
    }

    #[test]
    fn stored_roundtrip() {
        let m = members(5);
        assert_eq!(read_zip(&write_zip(&m, false)).unwrap(), m);
    }

    #[test]
    fn deflate_roundtrip_and_smaller() {
        let m = members(5);
        let z = write_zip(&m, true);
        assert_eq!(read_zip(&z).unwrap(), m);
        assert!(z.len() < write_zip(&m, false).len());
    }

    #[test]
    fn empty_and_one_byte_members() {
        let m = vec![
            ("empty".to_string(), vec![]),
            ("one".to_string(), vec![42]),
            ("zeros".to_string(), vec![0; 100_000]),
        ];
        for compress in [false, true] {
            assert_eq!(read_zip(&write_zip(&m, compress)).unwrap(), m);
        }
    }

    #[test]
    fn incompressible_member_falls_back_to_stored() {
        let m = vec![("r".to_string(), testutil::pseudo_random(5000, 1))];
        let z = write_zip(&m, true);
        let idx = read_index(&z[..]).unwrap();
        assert_eq!(idx[0].method, 0);
        assert_eq!(read_zip(&z).unwrap(), m);
    }

    #[test]
    fn unicode_names() {
        let m = vec![("naïve-数据.npy".to_string(), vec![1, 2, 3])];
        assert_eq!(read_zip(&write_zip(&m, true)).unwrap(), m);
    }

    #[test]
    fn crc_mismatch_is_detected() {
        let m = vec![("a".to_string(), b"hello world".to_vec())];
        let mut z = write_zip(&m, false);
        let pos = z.windows(11).position(|w| w == b"hello world").unwrap();
        z[pos] ^= 1;
        assert!(matches!(read_zip(&z), Err(NpzError::Crc { .. })));
    }

    #[test]
    fn truncated_archive_is_an_error() {
        let z = write_zip(&members(3), true);
        let cd = z
            .windows(4)
            .position(|w| w == CENTRAL_SIG.to_le_bytes())
            .unwrap();
        // Any cut that damages member data (or leaves nothing to find) must fail.
        for cut in [0, 1, 10, 21, 40, 100, cd / 2, cd - 1, cd] {
            assert!(read_zip(&z[..cut]).is_err(), "cut at {cut}");
        }
        // Cutting only inside the central directory loses no member data; the local-header
        // scan still recovers everything and verifies each CRC.
        assert_eq!(read_zip(&z[..z.len() - 1]).unwrap().len(), 3);
    }

    #[test]
    fn corrupting_any_byte_never_panics() {
        let m = vec![
            ("a".to_string(), vec![5u8; 300]),
            ("b".to_string(), testutil::pseudo_random(200, 4)),
        ];
        for archive in [write_zip(&m, true), write_zip_impl(&m, true, true)] {
            for i in 0..archive.len() {
                for flip in [0x01u8, 0x80, 0xFF] {
                    let mut z = archive.clone();
                    z[i] ^= flip;
                    let _ = read_zip(&z);
                }
            }
        }
    }

    #[test]
    fn absurd_zip64_offsets_are_errors_not_overflows() {
        let m = vec![("a".to_string(), vec![1u8; 10])];
        let mut z = write_zip_impl(&m, false, true);
        // Central zip64 extra: size, comp size, offset; set the offset to u64::MAX.
        let cd = z
            .windows(4)
            .position(|w| w == CENTRAL_SIG.to_le_bytes())
            .unwrap();
        let off_field = cd + 46 + 1 + 4 + 16;
        z[off_field..off_field + 8].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(read_zip(&z).is_err());
    }

    #[test]
    fn garbage_is_not_a_zip() {
        assert!(matches!(read_zip(b""), Err(NpzError::NotAZip)));
        assert!(matches!(
            read_zip(&testutil::pseudo_random(500, 9)),
            Err(NpzError::NotAZip)
        ));
    }

    #[test]
    fn unsupported_method_is_reported() {
        let m = vec![("a".to_string(), b"abc".to_vec())];
        let mut z = write_zip(&m, false);
        // central directory method field: signature + 10
        let cd = z
            .windows(4)
            .position(|w| w == CENTRAL_SIG.to_le_bytes())
            .unwrap();
        z[cd + 10] = 99;
        assert!(matches!(read_zip(&z), Err(NpzError::Unsupported(_))));
    }

    #[test]
    fn size_lie_is_detected() {
        let m = vec![("a".to_string(), vec![7u8; 5000])];
        let mut z = write_zip(&m, true);
        let cd = z
            .windows(4)
            .position(|w| w == CENTRAL_SIG.to_le_bytes())
            .unwrap();
        z[cd + 24] = z[cd + 24].wrapping_sub(1); // claim a smaller uncompressed size
        assert!(read_zip(&z).is_err());
    }

    #[test]
    fn forced_zip64_roundtrip() {
        let m = members(4);
        for compress in [false, true] {
            let z = write_zip_impl(&m, compress, true);
            assert_eq!(read_zip(&z).unwrap(), m);
        }
    }

    #[test]
    fn many_entries_use_zip64_end_record() {
        let m: Vec<(String, Vec<u8>)> = (0..70_000).map(|i| (format!("{i}"), vec![])).collect();
        let z = write_zip(&m, false);
        assert!(z.windows(4).any(|w| w == EOCD64_SIG.to_le_bytes()));
        let back = read_zip(&z).unwrap();
        assert_eq!(back.len(), 70_000);
        assert_eq!(back[69_999].0, "69999");
    }

    #[test]
    fn prepended_data_is_tolerated() {
        let m = members(2);
        let mut z = b"#!/bin/sh\nexit\n".to_vec();
        z.extend(write_zip(&m, true));
        assert_eq!(read_zip(&z).unwrap(), m);
    }

    #[test]
    fn missing_central_directory_falls_back_to_local_scan() {
        let m = members(3);
        let z = write_zip(&m, true);
        let cd = z
            .windows(4)
            .position(|w| w == CENTRAL_SIG.to_le_bytes())
            .unwrap();
        // Keep the start of the (now damaged) central directory: all local data is intact.
        assert_eq!(read_zip(&z[..cd + 10]).unwrap(), m);
        // Cutting exactly at a member boundary looks like truncation and is refused.
        assert!(matches!(read_zip(&z[..cd]), Err(NpzError::Truncated)));
    }

    #[test]
    fn python_reads_our_archives() {
        let Some(py) = Python::get() else { return };
        let dir = TempDir::new();
        let m = vec![
            ("a.npy".to_string(), testutil::pseudo_random(1000, 5)),
            ("b.npy".to_string(), vec![9u8; 50_000]),
            ("empty".to_string(), vec![]),
            ("naïve.txt".to_string(), b"unicode".to_vec()),
        ];
        let script = "import sys, zipfile, zlib\n\
            sys.stdout.reconfigure(encoding='utf-8')\n\
            z = zipfile.ZipFile(sys.argv[1])\n\
            assert z.testzip() is None\n\
            for n in z.namelist():\n    d = z.read(n)\n    print(n, len(d), zlib.crc32(d))\n";
        for (label, bytes) in [
            ("stored", write_zip(&m, false)),
            ("deflate", write_zip(&m, true)),
            ("zip64", write_zip_impl(&m, true, true)),
        ] {
            let path = dir.path(&format!("{label}.zip"));
            std::fs::write(&path, bytes).unwrap();
            let out = py.run(script, &[path.to_str().unwrap()], &[]);
            let text = String::from_utf8(out).unwrap();
            let got: Vec<&str> = text.lines().collect();
            assert_eq!(got.len(), m.len(), "{label}");
            for ((name, data), line) in m.iter().zip(&got) {
                let want = format!("{name} {} {}", data.len(), crc32(data));
                assert_eq!(*line, want, "{label}");
            }
        }
    }

    #[test]
    fn python_reads_many_entry_archive() {
        let Some(py) = Python::get() else { return };
        let dir = TempDir::new();
        let m: Vec<(String, Vec<u8>)> = (0..66_000).map(|i| (format!("f{i}"), vec![])).collect();
        let path = dir.path("many.zip");
        std::fs::write(&path, write_zip(&m, false)).unwrap();
        let out = py.run(
            "import sys, zipfile\nprint(len(zipfile.ZipFile(sys.argv[1]).namelist()))",
            &[path.to_str().unwrap()],
            &[],
        );
        assert_eq!(String::from_utf8(out).unwrap().trim(), "66000");
    }

    /// Archive written by Python's `zipfile` in the given style; returns the file path.
    fn python_zip(py: &Python, dir: &TempDir, style: &str) -> std::path::PathBuf {
        let path = dir.path(&format!("{style}.zip"));
        let script = r#"
import sys, zipfile, io
style, path = sys.argv[1], sys.argv[2]
payloads = {"a.bin": bytes(range(256)) * 40, "b.txt": b"hello hello hello " * 500, "e": b""}
class Unseekable(io.RawIOBase):
    def __init__(self, f): self.f = f
    def writable(self): return True
    def write(self, b): return self.f.write(b)
if style == "stream":
    raw = open(path, "wb")
    z = zipfile.ZipFile(Unseekable(raw), "w", zipfile.ZIP_DEFLATED)
elif style == "stored":
    z = zipfile.ZipFile(path, "w", zipfile.ZIP_STORED)
else:
    z = zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED)
for n, d in payloads.items():
    if style == "zip64":
        with z.open(n, "w", force_zip64=True) as f: f.write(d)
    elif style == "stream":
        zi = zipfile.ZipInfo(n)
        zi.compress_type = zipfile.ZIP_DEFLATED
        with z.open(zi, "w") as f: f.write(d)
    else:
        z.writestr(n, d)
z.close()
if style == "stream": raw.close()
"#;
        py.run(script, &[style, path.to_str().unwrap()], &[]);
        path
    }

    fn python_payloads() -> Vec<(String, Vec<u8>)> {
        let a: Vec<u8> = (0..=255u8).cycle().take(256 * 40).collect();
        let b: Vec<u8> = b"hello hello hello "
            .iter()
            .copied()
            .cycle()
            .take(18 * 500)
            .collect();
        vec![
            ("a.bin".into(), a),
            ("b.txt".into(), b),
            ("e".into(), vec![]),
        ]
    }

    #[test]
    fn reads_python_zipfile_archives_of_every_style() {
        let Some(py) = Python::get() else { return };
        let dir = TempDir::new();
        for style in ["stored", "deflate", "zip64", "stream"] {
            let path = python_zip(py, &dir, style);
            let bytes = std::fs::read(&path).unwrap();
            assert_eq!(read_zip(&bytes).unwrap(), python_payloads(), "{style}");
        }
    }

    #[test]
    fn streamed_python_archive_without_central_directory() {
        let Some(py) = Python::get() else { return };
        let dir = TempDir::new();
        let path = python_zip(py, &dir, "stream");
        let bytes = std::fs::read(&path).unwrap();
        // Python marks streamed members with a data descriptor (flag bit 3).
        assert_ne!(u16::from_le_bytes([bytes[6], bytes[7]]) & 8, 0);
        let cd = bytes
            .windows(4)
            .position(|w| w == CENTRAL_SIG.to_le_bytes())
            .unwrap();
        assert_eq!(read_zip(&bytes[..cd + 10]).unwrap(), python_payloads());
    }

    #[test]
    fn file_source_reads_one_member_lazily() {
        let m = members(4);
        let dir = TempDir::new();
        let path = dir.path("lazy.zip");
        std::fs::write(&path, write_zip(&m, true)).unwrap();
        let src = Mutex::new(File::open(&path).unwrap());
        let idx = read_index(&src).unwrap();
        assert_eq!(idx.len(), 4);
        assert_eq!(read_entry(&src, &idx[2]).unwrap(), m[2].1);
    }
}
