//! A criterion [`Measurement`] backed by a Linux `perf_event` hardware counter,
//! so the throughput bench reports **cycles/byte** (or instructions/byte)
//! *directly* instead of wall-clock time — with criterion's usual sampling,
//! confidence intervals, and outlier handling.
//!
//! Why this beats deriving cyc/byte from GiB/s and an assumed clock: the
//! `CPU_CYCLES` event counts *actual elapsed core cycles*, so the number is
//! independent of turbo/throttling — it **drops the constant-clock assumption**
//! entirely. (We deliberately use `CPU_CYCLES`, not `REF_CPU_CYCLES`, which counts
//! at a fixed nominal frequency and would smuggle the assumption back in.)
//!
//! Linux-only and behind the `perf` feature: `perf_event_open(2)` doesn't exist
//! elsewhere, and the AES-NI/VAES benches are x86_64-only anyway. The generic
//! `cycles`/`instructions` events are architecture-neutral — the kernel maps them
//! to the right raw PMU event per vendor — so they read identically on AMD (Zen)
//! and Intel. Requires a readable PMU (`kernel.perf_event_paranoid` low enough,
//! and on a VM a vPMU actually exposed to the guest).
//!
//! Thread-scoping caveat: a counter is bound to the calling thread, so it sees
//! only the bench thread's cycles — correct for the single-core kernels, wrong for
//! the rayon `*-parallel` variants (their worker threads aren't counted). The
//! throughput bench drops the parallel variants when this measurement is active.

use criterion::measurement::{Measurement, ValueFormatter};
use criterion::Throughput;
use perf_event::events::Hardware;
use perf_event::{Builder, Counter};

/// Which hardware event to count.
#[derive(Clone, Copy)]
pub enum Event {
    /// `PERF_COUNT_HW_CPU_CYCLES` — actual elapsed core cycles (frequency-aware).
    Cycles,
    /// `PERF_COUNT_HW_INSTRUCTIONS` — instructions retired.
    Instructions,
}

impl Event {
    fn hardware(self) -> Hardware {
        match self {
            Event::Cycles => Hardware::CPU_CYCLES,
            Event::Instructions => Hardware::INSTRUCTIONS,
        }
    }

    fn unit(self) -> &'static str {
        match self {
            Event::Cycles => "cycles",
            Event::Instructions => "instructions",
        }
    }

    fn per_byte_unit(self) -> &'static str {
        match self {
            Event::Cycles => "cycles/byte",
            Event::Instructions => "instructions/byte",
        }
    }

    fn per_element_unit(self) -> &'static str {
        match self {
            Event::Cycles => "cycles/element",
            Event::Instructions => "instructions/element",
        }
    }
}

/// Pick the event from `AES_PERF_EVENT` (`cycles` | `instructions`), defaulting to
/// cycles — so one bench binary reports either metric without a rebuild:
/// `AES_PERF_EVENT=instructions cargo bench -p aes-bench --features perf --bench throughput`.
pub fn event_from_env() -> Event {
    match std::env::var("AES_PERF_EVENT").as_deref() {
        Ok("instructions" | "insns" | "instr") => Event::Instructions,
        _ => Event::Cycles,
    }
}

/// Probe for a readable hardware PMU by opening, enabling, and reading a cycles
/// counter once. Returns the underlying error otherwise — `NotFound` (ENOENT) on
/// hosts with no vPMU (most cloud VMs, incl. Azure general-compute; `perf stat`
/// shows `cycles` as `<not supported>`), or `PermissionDenied` (EACCES) when
/// `kernel.perf_event_paranoid` is too high. Lets the bench fail up front with a
/// clear message instead of panicking mid-warmup inside [`Measurement::start`].
pub fn check_available() -> std::io::Result<()> {
    let mut counter = Builder::new().kind(Hardware::CPU_CYCLES).build()?;
    counter.enable()?;
    counter.disable()?;
    counter.read()?;
    Ok(())
}

/// A criterion measurement that counts one hardware event per iteration.
pub struct PerfMeasurement {
    event: Event,
    formatter: PerfFormatter,
}

impl PerfMeasurement {
    pub fn new(event: Event) -> Self {
        Self { event, formatter: PerfFormatter { event } }
    }

    /// Build one from `AES_PERF_EVENT` (see [`event_from_env`]).
    pub fn from_env() -> Self {
        Self::new(event_from_env())
    }
}

impl Measurement for PerfMeasurement {
    // A fresh, running counter is the per-sample intermediate: no shared mutable
    // state (the trait hands us `&self`), and one open fd per sample — negligible
    // against a sample's worth of iterations.
    type Intermediate = Counter;
    type Value = u64;

    fn start(&self) -> Counter {
        let mut counter = Builder::new()
            .kind(self.event.hardware())
            .build()
            .expect(
                "perf_event_open failed — is a PMU exposed to this host and \
                 kernel.perf_event_paranoid low enough?",
            );
        counter.enable().expect("failed to enable perf counter");
        counter
    }

    fn end(&self, mut counter: Counter) -> u64 {
        counter.disable().expect("failed to disable perf counter");
        counter.read().expect("failed to read perf counter")
    }

    fn add(&self, v1: &u64, v2: &u64) -> u64 {
        v1 + v2
    }

    fn zero(&self) -> u64 {
        0
    }

    fn to_f64(&self, value: &u64) -> f64 {
        *value as f64
    }

    fn formatter(&self) -> &dyn ValueFormatter {
        &self.formatter
    }
}

struct PerfFormatter {
    event: Event,
}

impl ValueFormatter for PerfFormatter {
    fn scale_values(&self, _typical: f64, _values: &mut [f64]) -> &'static str {
        // Raw event counts; no SI-style rescaling.
        self.event.unit()
    }

    fn scale_throughputs(&self, _typical: f64, throughput: &Throughput, values: &mut [f64]) -> &'static str {
        // `values` are per-iteration event counts; `throughput` is bytes (or
        // elements) per iteration. Dividing gives the per-byte metric we want.
        match *throughput {
            Throughput::Bytes(n) | Throughput::BytesDecimal(n) => {
                let n = n as f64;
                for v in values.iter_mut() {
                    *v /= n;
                }
                self.event.per_byte_unit()
            }
            Throughput::Elements(n) => {
                let n = n as f64;
                for v in values.iter_mut() {
                    *v /= n;
                }
                self.event.per_element_unit()
            }
        }
    }

    fn scale_for_machines(&self, _values: &mut [f64]) -> &'static str {
        self.event.unit()
    }
}
