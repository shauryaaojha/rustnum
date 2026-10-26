//! Buffer-cache controls, analogous to NumPy's internal allocation cache.
//!
//! ```
//! rustnum::memory::clear_pool();
//! assert!(rustnum::memory::pool_stats().cached_bytes <= rustnum::memory::pool_stats().limit_bytes);
//! ```
/// Allocation-cache counters, analogous to NumPy's internal allocation cache.
/// ```
/// let stats = rustnum::memory::pool_stats();
/// assert!(stats.cached_bytes <= stats.limit_bytes);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct PoolStats {
    /// Bytes held in cached Vec capacities, analogous to NumPy's cached data allocations.
    /// ```
    /// assert!(rustnum::memory::pool_stats().cached_bytes <= rustnum::memory::pool_stats().limit_bytes);
    /// ```
    pub cached_bytes: usize,
    /// Hard cache cap in bytes, analogous to NumPy's bounded allocation cache.
    /// ```
    /// let _cap = rustnum::memory::pool_stats().limit_bytes;
    /// ```
    pub limit_bytes: usize,
    /// Number of cached buffers, analogous to NumPy's cached allocations.
    /// ```
    /// let _buffers = rustnum::memory::pool_stats().cached_buffers;
    /// ```
    pub cached_buffers: usize,
    /// Successful buffer lookups, analogous to NumPy allocation-cache hits.
    /// ```
    /// let _hits = rustnum::memory::pool_stats().hits;
    /// ```
    pub hits: usize,
    /// Unsuccessful enabled large-buffer lookups, analogous to NumPy cache misses.
    /// ```
    /// let _misses = rustnum::memory::pool_stats().misses;
    /// ```
    pub misses: usize,
}
/// Reports cache use, analogous to inspecting NumPy's allocation cache.
/// ```
/// let s = rustnum::memory::pool_stats();
/// assert!(s.cached_bytes <= s.limit_bytes);
/// ```
pub fn pool_stats() -> PoolStats {
    let p = crate::pool::pool()
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    PoolStats {
        cached_bytes: p.bytes,
        limit_bytes: p.limit,
        cached_buffers: p.buffers(),
        hits: p.hits,
        misses: p.misses,
    }
}
/// Releases cached buffers to the system allocator, analogous to clearing NumPy's allocation cache.
/// ```
/// rustnum::memory::clear_pool();
/// assert_eq!(rustnum::memory::pool_stats().cached_bytes, 0);
/// ```
pub fn clear_pool() {
    crate::pool::pool()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clear();
}
/// Sets the hard cache cap in bytes, analogous to bounding NumPy's allocation cache.
///
/// Zero disables caching. Lowering the cap below current use clears the cache.
/// ```
/// rustnum::memory::set_pool_limit(0);
/// assert_eq!(rustnum::memory::pool_stats().limit_bytes, 0);
/// ```
pub fn set_pool_limit(bytes: usize) {
    let mut p = crate::pool::pool()
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    p.limit = bytes;
    if p.bytes > bytes {
        p.clear();
    }
}
