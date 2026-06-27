//! GPU AES-128 benchmarks (criterion). Requires an NVIDIA GPU + CUDA:
//! `cargo bench -p aes-bench --features gpu --bench gpu`.
//!
//! Two groups, measured through a single `AesGpu` (so the CUDA context is
//! initialized once):
//!
//! - `aes128-ecb` / `gpu/batch` — throughput of the one-thread-per-block kernel
//!   over `N_BLOCKS`, end-to-end (upload, launch, copy back). Shares the group
//!   and workload with `cpu.rs`, so the numbers are directly comparable.
//! - `aes128-latency` / `gpu/single-block` — per-launch round-trip latency of
//!   the `<<<1,1>>>` single-block kernel. Deliberately has NO throughput metric:
//!   it measures the fixed launch + tiny-transfer + sync cost (16 bytes of
//!   actual work), which is exactly the overhead that batching amortizes away.

use aes_bench::{demo_block, demo_blocks, demo_round_keys, N_BLOCKS};
use aes_gpu::AesGpu;
use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};

fn bench_gpu(c: &mut Criterion) {
    let gpu = AesGpu::new().expect("CUDA init failed (needs an NVIDIA GPU)");
    let rk = demo_round_keys();

    // Throughput: the batch kernel over the shared workload.
    {
        let blocks = demo_blocks(N_BLOCKS);
        let mut out = vec![[0u32; 4]; N_BLOCKS];
        let mut group = c.benchmark_group("aes128-ecb");
        group.throughput(Throughput::Bytes((N_BLOCKS * 16) as u64));
        group.bench_function("gpu/batch", |b| {
            b.iter(|| {
                gpu.encrypt_blocks(&rk, &blocks, &mut out)
                    .expect("kernel launch failed");
            });
        });
        group.finish();
    }

    // Latency: one block per launch — the per-launch floor, not throughput.
    {
        let pt = demo_block();
        let mut group = c.benchmark_group("aes128-latency");
        group.bench_function("gpu/single-block", |b| {
            b.iter(|| {
                gpu.encrypt_block(black_box(pt), black_box(&rk))
                    .expect("kernel launch failed")
            });
        });
        group.finish();
    }
}

criterion_group!(benches, bench_gpu);
criterion_main!(benches);
