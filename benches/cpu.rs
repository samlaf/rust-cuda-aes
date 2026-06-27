//! CPU AES-128 throughput benchmarks (criterion). Runs without a GPU/CUDA:
//! `cargo bench -p aes-bench --bench cpu`.
//!
//! As more backends land, add their `bench_function`s to the same group so
//! criterion reports them side by side (and, once a batch/CTR kernel exists, a
//! sibling `gpu.rs` bench can share the `aes_bench` workload helpers).

use aes_bench::{demo_blocks, demo_round_keys, N_BLOCKS};
use aes_cpu::scalar;
use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};

fn bench_cpu(c: &mut Criterion) {
    let rk = demo_round_keys();
    let blocks = demo_blocks(N_BLOCKS);
    let mut out = vec![[0u32; 4]; N_BLOCKS];

    let mut group = c.benchmark_group("aes128-ecb");
    group.throughput(Throughput::Bytes((N_BLOCKS * 16) as u64));
    group.bench_function("cpu/scalar", |b| {
        b.iter(|| scalar::encrypt_blocks(black_box(&rk), black_box(&blocks), &mut out));
    });
    group.finish();
}

criterion_group!(benches, bench_cpu);
criterion_main!(benches);
