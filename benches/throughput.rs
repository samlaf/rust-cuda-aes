//! AES-128 throughput: every backend variant in one comparable group.
//!
//! Iterates the `aes_bench` variant registry and benches each entry in one
//! criterion group over the shared workload, so the numbers line up directly.
//! This file is named for the measurement axis (throughput), pairing with
//! `latency.rs`; the criterion group is named for the cipher mode (`aes128-ctr`).
//!
//! CPU variants run by default; `--features gpu` builds and appends the GPU
//! variants to the same group. Two Linux-only cycle-counter backends swap
//! criterion's wall-clock timer so the report is in **cycles/byte** directly:
//! `--features perf` (perf_event PMU) and `--features aperf` (APERF MSR, for hosts
//! with no exposed PMU; needs root — see aperf.rs). Both drop the `*-parallel`
//! variants (a per-thread counter can't see the rayon workers); if both are set,
//! `aperf` wins.
//!
//! ```bash
//! cargo bench -p aes-bench --bench throughput                  # wall-clock: time + GiB/s
//! cargo bench -p aes-bench --features gpu --bench throughput   # CPU + GPU, side by side
//!
//! # cycles/byte, PMU present (perf_event); AES_PERF_EVENT=instructions → insns/byte:
//! cargo bench -p aes-bench --features perf --bench throughput
//!
//! # cycles/byte, no PMU (APERF MSR, e.g. Azure Zen 5): needs root for /dev/cpu/N/msr,
//! # and `sudo cargo` breaks rustup, so build as your user, run the built bin as root:
//! sudo modprobe msr
//! BIN=$(cargo bench -p aes-bench --features aperf --bench throughput --no-run \
//!         --message-format=json 2>/dev/null \
//!       | jq -r 'select(.executable and .target.name=="throughput") | .executable' | tail -1)
//! sudo taskset -c 0 "$BIN" --bench     # pins to AES_APERF_CPU (default 0)
//! ```

use aes_bench::{cpu_variants, demo_blocks, demo_counter0, demo_round_keys, Aes, N_BLOCKS};
use criterion::measurement::Measurement;
use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};

// Cycle-counting `Measurement`s are compiled into this bench (criterion is a
// dev-dep, so they can't live in the `aes_bench` lib) and only on Linux behind a
// feature. Two backends: `perf` (perf_event PMU) for PMU-capable hosts, and
// `aperf` (APERF MSR) for hosts with no exposed PMU (e.g. the Azure Zen 5 slice).
// `aperf` wins if both are set, since it's the one that works where `perf` can't.
#[cfg(all(feature = "perf", not(feature = "aperf")))]
#[path = "perf.rs"]
mod perf;

#[cfg(feature = "aperf")]
#[path = "aperf.rs"]
mod aperf;

// Generic over the measurement so the same body serves both the wall-clock timer
// (default) and the perf-counter one (`--features perf`).
fn bench_throughput<M: Measurement>(c: &mut Criterion<M>) {
    // Fail fast (before criterion's 3s warm-up) with an actionable message if the
    // perf counters aren't readable, rather than panicking inside the measurement.
    // (The `aperf` backend fails fast in `AperfMeasurement::new()` instead.)
    #[cfg(all(feature = "perf", not(feature = "aperf")))]
    if let Err(e) = perf::check_available() {
        let hint = match e.kind() {
            std::io::ErrorKind::PermissionDenied => {
                "kernel.perf_event_paranoid is too high — try \
                 `sudo sysctl kernel.perf_event_paranoid=1`."
            }
            _ => {
                "this host exposes no hardware PMU (common on cloud VMs, incl. Azure \
                 general-compute — `perf stat` shows `cycles` as <not supported>). Run on \
                 a host with a vPMU, or drop `--features perf` for the wall-clock bench \
                 and derive cyc/byte from a verified clock."
            }
        };
        eprintln!("\naes-bench --features perf: can't read hardware counters: {e}\n{hint}\n");
        std::process::exit(1);
    }

    let rk = demo_round_keys();
    let counter0 = demo_counter0();
    let blocks = demo_blocks(N_BLOCKS);
    let mut out = vec![[0u32; 4]; N_BLOCKS];

    // CPU variants always; GPU variants only when the feature (and CUDA) is on.
    #[allow(unused_mut)]
    let mut variants: Vec<Box<dyn Aes>> = cpu_variants();
    #[cfg(feature = "gpu")]
    variants.extend(aes_bench::gpu_variants());

    // A per-core cycle counter (perf_event or APERF) is thread-scoped: on the
    // `*-parallel` variants it would count only the bench thread, not the rayon
    // workers, so cycles/byte would be meaningless. Restrict the counter runs to
    // the single-core ladder (post 1); the parallel variants stay wall-clock-only.
    #[cfg(any(feature = "perf", feature = "aperf"))]
    variants.retain(|v| !v.name().ends_with("-parallel"));

    let mut group = c.benchmark_group("aes128-ctr");
    group.throughput(Throughput::Bytes((N_BLOCKS * 16) as u64));
    for v in &variants {
        group.bench_function(v.name(), |b| {
            b.iter(|| v.encrypt_ctr(black_box(&rk), counter0, black_box(&blocks), &mut out));
        });
    }
    group.finish();
}

// Pick the measurement: `aperf` (APERF MSR) wins if set, else `perf` (perf_event
// PMU), else criterion's default wall-clock timer.
#[cfg(feature = "aperf")]
criterion_group! {
    name = benches;
    config = Criterion::default().with_measurement(aperf::AperfMeasurement::new());
    targets = bench_throughput
}

#[cfg(all(feature = "perf", not(feature = "aperf")))]
criterion_group! {
    name = benches;
    config = Criterion::default().with_measurement(perf::PerfMeasurement::from_env());
    targets = bench_throughput
}

#[cfg(not(any(feature = "perf", feature = "aperf")))]
criterion_group!(benches, bench_throughput);

criterion_main!(benches);
