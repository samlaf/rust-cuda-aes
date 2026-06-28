//! AES-128 GPU single-block latency (criterion). GPU-only:
//!
//! ```bash
//! cargo bench -p aes-bench --features gpu --bench latency
//! ```
//!
//! Per-launch round-trip of the `<<<1,1>>>` single-block kernel — the fixed
//! launch + tiny-transfer + sync cost (16 bytes of real work) that batching
//! amortizes away. Deliberately no throughput metric.

use aes_bench::{demo_block, demo_round_keys};
use aes_gpu::AesGpu;
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_latency(c: &mut Criterion) {
    let gpu = AesGpu::new().expect("CUDA init failed (needs an NVIDIA GPU)");
    let rk = demo_round_keys();
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

criterion_group!(benches, bench_latency);
criterion_main!(benches);
