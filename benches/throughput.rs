//! AES-128 throughput: every backend variant in one comparable group.
//!
//! Iterates the `aes_bench` variant registry and benches each entry in one
//! criterion group over the shared workload, so the numbers line up directly.
//! This file is named for the measurement axis (throughput), pairing with
//! `latency.rs`; the criterion group is named for the cipher mode (`aes128-ctr`).
//!
//! CPU variants run by default; `--features gpu` builds and appends the GPU
//! variants to the same group:
//!
//! ```bash
//! cargo bench -p aes-bench --bench throughput                 # CPU variants only
//! cargo bench -p aes-bench --features gpu --bench throughput  # CPU + GPU, side by side
//! ```

use aes_bench::{cpu_variants, demo_blocks, demo_counter0, demo_round_keys, Aes, N_BLOCKS};
use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};

fn bench_throughput(c: &mut Criterion) {
    let rk = demo_round_keys();
    let counter0 = demo_counter0();
    let blocks = demo_blocks(N_BLOCKS);
    let mut out = vec![[0u32; 4]; N_BLOCKS];

    // CPU variants always; GPU variants only when the feature (and CUDA) is on.
    #[allow(unused_mut)]
    let mut variants: Vec<Box<dyn Aes>> = cpu_variants();
    #[cfg(feature = "gpu")]
    variants.extend(aes_bench::gpu_variants());

    let mut group = c.benchmark_group("aes128-ctr");
    group.throughput(Throughput::Bytes((N_BLOCKS * 16) as u64));
    for v in &variants {
        group.bench_function(v.name(), |b| {
            b.iter(|| v.encrypt_ctr(black_box(&rk), counter0, black_box(&blocks), &mut out));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_throughput);
criterion_main!(benches);
