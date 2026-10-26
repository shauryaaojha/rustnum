//! Bulk `f32 <-> f16` conversion: the F16C path (used by `astype`) against the scalar algorithm.
//!
//! This is the benchmark that justifies the `unsafe` F16C intrinsics in `src/half.rs`
//! (AGENTS.md rule 3). Run it with `cargo bench -p rustnum --bench half`.

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use rustnum as np;
use rustnum::f16;
use std::hint::black_box;

fn convert(c: &mut Criterion) {
    let n = 1 << 20;
    let src: Vec<f32> = (0..n)
        .map(|i| ((i % 65536) as f32 - 32768.0) * 0.013)
        .collect();
    let a = np::Array::from(src.clone());
    let h = a.astype::<f16>();
    let mut group = c.benchmark_group("f32_f16");
    group.throughput(Throughput::Elements(n as u64));
    group.bench_function("f32_to_f16_astype_bulk", |b| {
        b.iter(|| black_box(a.astype::<f16>()));
    });
    group.bench_function("f32_to_f16_scalar", |b| {
        let mut out = vec![f16::from_bits(0); n];
        b.iter(|| {
            f16::slice_from_f32_scalar(black_box(&src), &mut out);
            black_box(&out);
        });
    });
    group.bench_function("f16_to_f32_astype_bulk", |b| {
        b.iter(|| black_box(h.astype::<f32>()));
    });
    group.bench_function("f16_to_f32_scalar", |b| {
        let v = h.to_vec();
        b.iter(|| black_box(v.iter().map(|x| x.to_f32()).collect::<Vec<f32>>()));
    });
    group.finish();
}

criterion_group!(benches, convert);
criterion_main!(benches);
