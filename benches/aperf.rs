//! A criterion [`Measurement`] that counts **actual core cycles via the APERF
//! MSR** (`IA32_APERF`, `0xE8`), read through `/dev/cpu/N/msr` — the fallback for
//! hosts that expose no PMU to `perf_event_open` but still let the guest read the
//! P-state MSRs. That's the situation on the Azure Zen 5 (Fasv7) slice: `perf
//! stat -e cycles` is `<not supported>`, but `rdmsr 0xE8` works and increments.
//!
//! APERF increments at the core's *actual* frequency and freezes while the core
//! is halted, so between [`start`](Measurement::start) and [`end`] of a busy
//! bench loop the delta **is** the retired cycle count — identical in meaning to
//! the PMU's `CPU_CYCLES`, and with no constant-clock assumption (which matters
//! here: single-core boost is ~4.5 GHz, not the all-core ~4.17). Combined with
//! `Throughput::Bytes`, the report is in **cycles/byte** directly.
//!
//! Requirements: the `msr` kernel module loaded and read access to
//! `/dev/cpu/N/msr` (run the bench under `sudo`). The measured thread is pinned to
//! core `N` (env `AES_APERF_CPU`, default 0) so the work accrues on the same core
//! whose APERF we read, and can't migrate mid-measurement. Linux + `aperf` feature
//! only. Like any per-core counter it's meaningless for the multi-threaded
//! `*-parallel` variants — the throughput bench drops those under this feature.

use std::fs::File;
use std::os::unix::fs::FileExt;

use criterion::measurement::{Measurement, ValueFormatter};
use criterion::Throughput;

/// `IA32_APERF` — the actual-performance MSR, used as the file offset into
/// `/dev/cpu/N/msr` (the msr driver maps offset → MSR number, 8 bytes per read).
const IA32_APERF: u64 = 0xE8;

/// Which core to pin to / read APERF from (`AES_APERF_CPU`, default 0).
fn target_cpu() -> usize {
    std::env::var("AES_APERF_CPU")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0)
}

/// Pin the current thread to `cpu` so its cycles accrue on the core we read.
fn pin_to_cpu(cpu: usize) {
    // SAFETY: standard libc affinity dance; `set` is zeroed before use and `0`
    // targets the calling thread.
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_ZERO(&mut set);
        libc::CPU_SET(cpu, &mut set);
        if libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set) != 0 {
            eprintln!("aperf: warning: failed to pin to cpu {cpu}; APERF reads may drift");
        }
    }
}

/// Cycle-counting measurement backed by the APERF MSR.
pub struct AperfMeasurement {
    msr: File,
    formatter: AperfFormatter,
}

impl AperfMeasurement {
    /// Pin to the target core, open its MSR file, and confirm APERF is readable.
    /// Criterion constructs the measurement and runs the benches on the same
    /// (main) thread, so pinning here pins the thread that does the work.
    pub fn new() -> Self {
        let cpu = target_cpu();
        pin_to_cpu(cpu);
        let path = format!("/dev/cpu/{cpu}/msr");
        let msr = File::open(&path).unwrap_or_else(|e| {
            panic!(
                "aperf: open {path} failed ({e}) — load the module (`sudo modprobe msr`) \
                 and run the bench as root (`sudo -E ... cargo bench`)."
            )
        });
        let m = Self { msr, formatter: AperfFormatter };
        // Fail fast if the MSR isn't actually readable, rather than mid-warmup.
        let _ = m.read_aperf();
        m
    }

    fn read_aperf(&self) -> u64 {
        let mut buf = [0u8; 8];
        self.msr
            .read_exact_at(&mut buf, IA32_APERF)
            .expect("aperf: rdmsr APERF via /dev/cpu/*/msr failed");
        u64::from_le_bytes(buf)
    }
}

impl Default for AperfMeasurement {
    fn default() -> Self {
        Self::new()
    }
}

impl Measurement for AperfMeasurement {
    type Intermediate = u64; // APERF snapshot at start
    type Value = u64; // cycle delta over the batch

    fn start(&self) -> u64 {
        self.read_aperf()
    }

    fn end(&self, start: u64) -> u64 {
        // wrapping_sub tolerates the (astronomically rare) 64-bit APERF wrap.
        self.read_aperf().wrapping_sub(start)
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

struct AperfFormatter;

impl ValueFormatter for AperfFormatter {
    fn scale_values(&self, _typical: f64, _values: &mut [f64]) -> &'static str {
        "cycles"
    }

    fn scale_throughputs(&self, _typical: f64, throughput: &Throughput, values: &mut [f64]) -> &'static str {
        match *throughput {
            Throughput::Bytes(n) | Throughput::BytesDecimal(n) => {
                let n = n as f64;
                for v in values.iter_mut() {
                    *v /= n;
                }
                "cycles/byte"
            }
            Throughput::Elements(n) => {
                let n = n as f64;
                for v in values.iter_mut() {
                    *v /= n;
                }
                "cycles/element"
            }
        }
    }

    fn scale_for_machines(&self, _values: &mut [f64]) -> &'static str {
        "cycles"
    }
}
