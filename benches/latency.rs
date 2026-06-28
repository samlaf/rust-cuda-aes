//! AES-128 GPU single-block latency (criterion). GPU-only:
//!
//! ```bash
//! cargo bench -p aes-bench --features gpu --bench latency
//! ```
//!
//! Per-launch round-trip of the CTR kernel over a single block — the fixed
//! launch + tiny-transfer + sync cost (16 bytes of real work) that batching
//! amortizes away. Deliberately no throughput metric.

use aes_bench::{demo_block, demo_counter0, demo_round_keys};
use aes_gpu::AesGpu;
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_latency(c: &mut Criterion) {
    let gpu = AesGpu::new().expect("CUDA init failed (needs an NVIDIA GPU)");
    let rk = demo_round_keys();
    let counter0 = demo_counter0();
    let blocks = [demo_block()];
    let mut out = [[0u32; 4]];

    let mut group = c.benchmark_group("aes128-latency");
    group.bench_function("gpu/single-block", |b| {
        b.iter(|| {
            gpu.encrypt_ctr(black_box(&rk), counter0, black_box(&blocks), &mut out, 1)
                .expect("kernel launch failed")
        });
    });
    group.finish();
}

criterion_group!(benches, bench_latency);
criterion_main!(benches);
