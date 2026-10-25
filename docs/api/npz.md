# `.npz` API (`np::io`) — proposed section for docs/API.md §13

Same conventions as `docs/API.md`. NumPy equivalents: `np.savez`, `np.savez_compressed`,
`np.load(path)` on a `.npz` file (`NpzFile`). Implementation lives in the private module
`rustnum/src/npz/` (pure Rust, zero dependencies); the four public items below are re-exported
once from `io.rs` (`np::io::savez`, ...). Nothing else in `npz/` is public.

## 13b. `.npz` archives — `np::io::*`

```rust
pub fn savez<T: Element>(path: impl AsRef<Path>, arrays: &[(&str, &Array<T>)]) -> Result<()>
pub fn savez_compressed<T: Element>(path: impl AsRef<Path>, arrays: &[(&str, &Array<T>)]) -> Result<()>
pub fn load_npz<T: Element>(path: impl AsRef<Path>) -> Result<Vec<(String, Array<T>)>>

pub struct NpzFile { /* private */ }
impl NpzFile {
    pub fn open(path: impl AsRef<Path>) -> Result<NpzFile>
    pub fn keys(&self) -> Vec<String>
    pub fn get<T: Element>(&self, name: &str) -> Result<Array<T>>
}
```

| Item | NumPy | Behaviour |
|---|---|---|
| `savez(path, &[("a", &a), ("b", &b)])` | `np.savez(path, a=a, b=b)` | Writes a ZIP archive with one stored (uncompressed) member `<name>.npy` per array, in the given order. |
| `savez_compressed(...)` | `np.savez_compressed(...)` | Same, DEFLATE-compressed. A member that does not get smaller is stored instead. |
| `load_npz::<T>(path)` | `dict(np.load(path))` | Reads every member, in archive order, names without `.npy`. All members must have dtype `T`, otherwise `Error::DType`. |
| `NpzFile::open(path)` | `np.load(path)` | Reads only the table of contents (the ZIP central directory). Keeps the file open. |
| `NpzFile::keys()` | `.files` | Member names without `.npy`, archive order. |
| `NpzFile::get::<T>(name)` | `npzfile[name]` | Seeks to the member, inflates it, verifies its CRC-32, decodes the `.npy`. `name` may be given with or without `.npy`. If a name occurs twice, the last member wins (as in a Python dict). |

### Rules
- `savez*` takes arrays of one element type `T` per call. Different dtypes in one file are read
  with `NpzFile::get` using a different `T` per member; writing mixed dtypes needs a separate
  call design (see open questions).
- Names must be non-empty, unique and at most 65531 bytes (65535 minus `.npy`); a violation returns
  `Error::Format`. Positional names (`arr_0`) are not generated: NumPy's `*args` form has no
  Rust counterpart, pass explicit names.
- Written files are plain ZIP: version-needed 1.0 or 2.0, MS-DOS timestamp 1980-01-01 (like NumPy),
  UTF-8 flag set for non-ASCII names, no data descriptors. Zip64 records are written only when a
  member, offset or the entry count exceeds the classic limits (4 GiB, 65534 entries).
- Readable inputs: methods 0 (stored) and 8 (deflate); zip64 sizes, offsets and end records;
  data descriptors; archives with prepended data; an archive with no central directory is recovered
  by scanning local headers if it ends with a central-directory or end record. Encrypted members,
  other compression methods and multi-disk archives return `Error::Format`.
- Every member's CRC-32 and uncompressed size are verified; a mismatch returns `Error::Format`.
  A corrupt member does not stop other members of an `NpzFile` from being read.
- Corrupt input never panics and never loops: the DEFLATE decoder caps its output at the size
  declared in the archive (zip-bomb safe).
- A `.npy` member written by NumPy (any header version `from_npy_bytes` accepts) is accepted
  unchanged; the bytes we write for a member are exactly what `to_npy_bytes` returns.

### Errors
| Situation | Error |
|---|---|
| File cannot be opened / written | `Error::Io` |
| Not a ZIP file, truncated, bad CRC, unsupported method, bad deflate stream, missing name, bad name | `Error::Format(String)` |
| Member dtype is not `T` | `Error::DType` (from `from_npy_bytes`) |

### Internal layers (private, for maintainers)
| Module | Contents |
|---|---|
| `npz/crc32.rs` | `crc32(&[u8]) -> u32`, incremental `Crc32`; slicing-by-8 tables built at compile time |
| `npz/inflate.rs` | `inflate`, `inflate_with_limit(input, limit) -> (Vec<u8>, bytes_consumed)`; RFC 1951 stored / fixed / dynamic blocks |
| `npz/deflate.rs` | `deflate`, `deflate_with(data, Mode)`; hash-chain LZ77 with lazy matching; per block the smallest of stored / fixed / dynamic Huffman; output within 0.1 % of zlib level 6 on the benchmark sample |
| `npz/zip.rs` | `read_zip(&[u8])`, `write_zip(&[(String, Vec<u8>)], compress)`; random-access `ByteSource` so `NpzFile` reads one member without loading the file |
| `npz/mod.rs` | `NpzError`, `npz_to_bytes`, `npz_from_bytes`, `write_npz_file`, `read_npz_file`, `NpzArchive` (byte-level; `.npy` blobs in and out) |
| `npz/api.rs` | The array-level functions above, built on `io::{to_npy_bytes, from_npy_bytes}` |

### Open questions for the spec
1. Mixed dtypes in one `savez` call: either accept `&[(&str, &dyn NpyEncodable)]` or add a small
   builder. Not added, to avoid inventing public names.
2. `np.load` returns an `NpzFile` for both `.npy` and `.npz`; here `load_npy` and `NpzFile::open`
   stay separate, matching the existing split in §13.
3. Should `NpzFile` also offer `len()` / `contains()`? The byte-level `NpzArchive` already does.
