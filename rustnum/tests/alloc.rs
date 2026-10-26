//! Counts actual allocator calls and requested bytes, separately identifying data buffers.
use rustnum as np;
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
struct Counting;
// Only the measuring thread is counted, so harness and rayon-worker noise cannot leak in.
thread_local! { static ENABLED: Cell<bool> = const { Cell::new(false) }; }
static CALLS: AtomicUsize = AtomicUsize::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);
static DATA: AtomicUsize = AtomicUsize::new(0);
static DATA_SIZE: AtomicUsize = AtomicUsize::new(0);
fn record(size: usize) {
    if ENABLED.try_with(Cell::get).unwrap_or(false) {
        CALLS.fetch_add(1, Relaxed);
        BYTES.fetch_add(size, Relaxed);
        if size == DATA_SIZE.load(Relaxed) {
            DATA.fetch_add(1, Relaxed);
        }
    }
}
// SAFETY: all allocation operations delegate unchanged to the system allocator;
// atomic counters do not allocate. The allocation audit and memory_probe benchmark
// exercise this instrumentation, which is confined to this integration-test binary.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record(layout.size());
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record(layout.size());
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn dealloc(&self, p: *mut u8, layout: Layout) {
        unsafe { System.dealloc(p, layout) }
    }
    unsafe fn realloc(&self, p: *mut u8, layout: Layout, n: usize) -> *mut u8 {
        record(n);
        unsafe { System.realloc(p, layout, n) }
    }
}
#[global_allocator]
static ALLOCATOR: Counting = Counting;
#[derive(Debug, PartialEq)]
struct Counts {
    calls: usize,
    bytes: usize,
    data: usize,
}
fn measure<R>(size: usize, f: impl FnOnce() -> R) -> (R, Counts) {
    CALLS.store(0, Relaxed);
    BYTES.store(0, Relaxed);
    DATA.store(0, Relaxed);
    DATA_SIZE.store(size, Relaxed);
    ENABLED.with(|e| e.set(true));
    let result = f();
    ENABLED.with(|e| e.set(false));
    (
        result,
        Counts {
            calls: CALLS.load(Relaxed),
            bytes: BYTES.load(Relaxed),
            data: DATA.load(Relaxed),
        },
    )
}
#[track_caller]
fn zero(c: Counts) {
    assert_eq!(
        c,
        Counts {
            calls: 0,
            bytes: 0,
            data: 0
        }
    );
}
#[test]
fn allocation_contracts() {
    np::memory::set_pool_limit(0);
    // Warm lazy runtime initialization outside the measured regions.
    drop(np::ones::<f64>([32768]).into_tanh());
    let a = np::ones::<f64>([1024]);
    let (a, c) = measure(8192, || a / 2.);
    zero(c);
    let (a, c) = measure(8192, || a.into_tanh());
    zero(c);
    let b = np::ones::<f64>([1024]);
    let (a, c) = measure(8192, || a + &b);
    zero(c);
    let (a, c) = measure(8192, || a - b);
    zero(c);
    let (mut a, c) = measure(8192, || -a);
    zero(c);
    let (_, c) = measure(8192, || a *= 2.);
    zero(c);
    let (_, c) = measure(8192, || a.mapv_inplace(|x| x + 1.));
    zero(c);
    let (_, c) = measure(8192, || a.mapv_inplace_parallel(|x| x + 1.));
    zero(c);
    let (a, c) = measure(8192, || a.into_astype::<f64>());
    zero(c);
    let (_, c) = measure(8192, || a.sum());
    zero(c);
    let (_, c) = measure(8192, || a.var());
    zero(c);
    let (_, c) = measure(8192, || a.nansum());
    zero(c);
    let matrix = np::ones::<f64>([32, 32]);
    for axis in [0, 1] {
        let (sum, c) = measure(256, || matrix.sum_axis(axis));
        println!("sum_axis({axis}): {c:?}");
        assert_eq!(c.data, 1);
        assert_eq!(c.calls, 4);
        assert_eq!(c.bytes, 256 + 16 + 8 + 40);
        assert_eq!(sum.to_vec(), [32.; 32]);
        let (_, c) = measure(256, || matrix.mean_axis(axis));
        assert_eq!(c.calls, 4);
        assert_eq!(c.bytes, 320);
        assert_eq!(c.data, 1);
        let (_, c) = measure(256, || matrix.var_axis(axis));
        assert_eq!(c.calls, 4);
        assert_eq!(c.bytes, 320);
        assert_eq!(c.data, 1);
    }
    let b = np::ones::<f64>([32, 32]);
    drop(matrix.matmul(&b));
    let (_, baseline) = measure(8192, || matrix.matmul(&b));
    let (result, chain) = measure(8192, || (matrix.matmul(&b) / 32.).into_tanh());
    println!("matmul alone: {baseline:?}; heavy step: {chain:?}");
    assert_eq!(chain, baseline);
    assert_eq!(chain.data, 1);
    assert!(result.iter().all(|x| x == 1f64.tanh()));
    println!(
        "owned operators, into_tanh, scalar assignment, same-type consuming cast, scalar reductions: 0 allocations / 0 bytes"
    );
}
