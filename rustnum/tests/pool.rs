use rustnum as np;
#[test]
fn limits_types_initialization_and_concurrent_returns() {
    const N: usize = 1 << 17;
    const BYTES: usize = N * 8;
    np::memory::set_pool_limit(2 * BYTES);
    np::memory::clear_pool();
    let a = np::full([N], 7f64);
    let pointer = a.as_slice().unwrap().as_ptr();
    drop(a);
    assert_eq!(np::memory::pool_stats().cached_bytes, BYTES);
    let a = np::full([N], 9f64);
    assert_eq!(pointer, a.as_slice().unwrap().as_ptr());
    assert!(a.iter().all(|x| x == 9.));
    assert!(np::memory::pool_stats().hits > 0);
    drop(a);
    let integers = np::full([N], 3i64);
    assert_ne!(
        integers.as_slice().unwrap().as_ptr() as usize,
        pointer as usize
    );
    drop(integers);
    assert_eq!(np::memory::pool_stats().cached_bytes, 2 * BYTES);
    let excess = np::full([2 * N], 1i32);
    drop(excess);
    assert_eq!(np::memory::pool_stats().cached_bytes, 2 * BYTES);
    np::memory::set_pool_limit(BYTES);
    assert_eq!(np::memory::pool_stats().cached_bytes, 0);
    let a = np::full([N], 1f64);
    let b = np::full([N], 2f64);
    drop(a);
    drop(b);
    assert_eq!(np::memory::pool_stats().cached_bytes, BYTES);
    np::memory::clear_pool();
    assert_eq!(np::memory::pool_stats().cached_buffers, 0);
    std::thread::scope(|scope| {
        for _ in 0..4 {
            scope.spawn(|| {
                for _ in 0..4 {
                    let a = np::full([N], 5f64);
                    assert_eq!(a[[N - 1]], 5.);
                }
            });
        }
    });
    assert!(np::memory::pool_stats().cached_bytes <= BYTES);
    np::memory::set_pool_limit(0);
    drop(np::full([N], 1f64));
    assert_eq!(np::memory::pool_stats().cached_bytes, 0);
}
