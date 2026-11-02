//! Safe, initialized Vec cache. Exact-capacity buckets avoid retaining oversized buffers.
use crate::Element;
use std::{
    any::{Any, TypeId},
    collections::HashMap,
    sync::{Mutex, OnceLock},
};
pub(crate) const MIN_BYTES: usize = 1 << 20;
type Buckets = HashMap<(TypeId, usize), Vec<Box<dyn Any + Send>>>;
pub(crate) struct Pool {
    pub(crate) limit: usize,
    pub(crate) bytes: usize,
    pub(crate) hits: usize,
    pub(crate) misses: usize,
    buckets: Buckets,
}
pub(crate) fn pool() -> &'static Mutex<Pool> {
    static POOL: OnceLock<Mutex<Pool>> = OnceLock::new();
    POOL.get_or_init(|| {
        Mutex::new(Pool {
            limit: std::env::var("RUSTNUM_POOL_BYTES")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(2 * (1 << 30) / 4),
            bytes: 0,
            hits: 0,
            misses: 0,
            buckets: HashMap::new(),
        })
    })
}
impl Pool {
    pub(crate) fn clear(&mut self) {
        self.buckets.clear();
        self.bytes = 0;
    }
    pub(crate) fn buffers(&self) -> usize {
        self.buckets.values().map(Vec::len).sum()
    }
}
pub(crate) fn release<T: Element>(data: Vec<T>) {
    let bytes = data.capacity().saturating_mul(std::mem::size_of::<T>());
    // `take(n)` looks buffers up by length, so an over-allocated buffer would never be reused
    // and would only hold pool budget.
    if bytes < MIN_BYTES || data.capacity() != data.len() {
        return;
    }
    let mut p = pool().lock().unwrap_or_else(|e| e.into_inner());
    if bytes > p.limit.saturating_sub(p.bytes) {
        return;
    }
    p.bytes += bytes;
    p.buckets
        .entry((TypeId::of::<T>(), data.capacity()))
        .or_default()
        .push(Box::new(data));
}
pub(crate) fn take<T: Element>(n: usize) -> Option<Vec<T>> {
    let bytes = n.saturating_mul(std::mem::size_of::<T>());
    if bytes < MIN_BYTES {
        return None;
    }
    let mut p = pool().lock().unwrap_or_else(|e| e.into_inner());
    if p.limit == 0 {
        return None;
    }
    if let Some(data) = p
        .buckets
        .get_mut(&(TypeId::of::<T>(), n))
        .and_then(Vec::pop)
    {
        p.bytes -= bytes;
        p.hits += 1;
        return Some(*data.downcast::<Vec<T>>().expect("pool type key"));
    }
    p.misses += 1;
    None
}
pub(crate) fn filled<T: Element>(n: usize, value: T) -> Vec<T> {
    if let Some(mut data) = take(n) {
        data.resize(n, value);
        data.fill(value);
        data
    } else {
        vec![value; n]
    }
}
// Serial producer initializes every element; no spare capacity is ever read.
pub(crate) fn collect<T: Element>(n: usize, iter: impl Iterator<Item = T>) -> Vec<T> {
    let mut data = take(n).unwrap_or_else(|| Vec::with_capacity(n));
    data.clear();
    data.extend(iter);
    data
}

// Only for kernels that overwrite all n elements before exposing the result.
// Cached elements are already valid T values; fresh storage is initialized safely.
pub(crate) fn output<T: Element>(n: usize) -> Vec<T> {
    if let Some(mut data) = take(n) {
        data.resize(n, T::default());
        data
    } else {
        vec![T::default(); n]
    }
}
