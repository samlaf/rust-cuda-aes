# Measuring cycles/byte honestly (when your cloud VM has no PMU)

*Methods note for the series. **OUTLINE / stub** — the through-line and the hard-won
facts are captured below; the multi-machine PMU results (§7) are pending data.*

> The three posts in this series ([1: single-core], [2: multi-core], [3: GPU]) all
> lean on one number — **cycles/byte** — as the fair, clock-normalized measure of
> per-core efficiency. Getting that number *honestly* on cloud CPUs turned out
> involved enough to factor out here, so the posts can just link to it. This is the
> "how we measured, and what we can and can't measure" note.

[1: single-core]: 1-single-core-aes.md
[2: multi-core]: 2-multi-core-aes.md
[3: GPU]: 3-gpu.md

---

## 1. Why cycles/byte, not GiB/s

- GiB/s is what you feel, but it **hides the clock**: the same code reports a
  different GiB/s at a different frequency, so you can't compare across machines,
  boost states, or SMT configs. Cycles/byte divides the clock out — it's *work per
  byte*, the property of the code + microarchitecture, not the power/thermal state.
- It's the right metric *within* an architecture. It does **not** transfer across
  architectures (an AESENC-cycle on Zen ≠ one on a GPU SM), so post 3's cross-device
  comparison drops back to throughput at a stated buffer size.

## 2. The ceiling model

- The AES-NI/VAES throughput ceiling is set by the AES execution units, not the
  instruction count: `ceiling = 10 rounds × 0.5 (recip. throughput) ÷ bytes-per-instr`.
- 128-bit (16 B/instr) → **0.3125 cyc/byte**; VAES halves it at each width
  (256-bit → 0.15625, 512-bit → 0.078125).
- Validated against [uops.info]: on Zen 5, `VAESENC` at *every* width (XMM/YMM/ZMM)
  is **TP 0.50, latency 4, 1 µop, ports FP0/1**. Two ports × 1/cycle = the "2 AES
  units" the ceiling assumes; TP staying 0.50 at ZMM is the native-512-datapath
  proof. (Latency 4 × 2 ports = 8 in flight to saturate — the interleave width.)

[uops.info]: https://uops.info

## 3. What we'd measure it with, ideally: the PMU

- The **Performance Monitoring Unit** exposes hardware counters (PMCs) via
  `perf_event_open` / `perf`: core cycles, **instructions retired**, cache
  refs/misses (L1/L2/L3), branch mispredicts, stalled cycles front/back-end,
  **µops dispatched per execution port**, TLB misses — in counting *or* sampling
  mode (`perf record`/`annotate` attributes events to source lines).
- That's the tool that would let us *prove* post 1's rung-4 claim (the byte repack
  saturates the load/store ports) by counting the port µops directly — not just
  argue it.

## 4. The cloud reality: counter access is a lottery

Whether a VM exposes a PMU to the guest is a **hypervisor policy set per VM
series/host generation**, not a CPU capability. On the box this series runs on it's
simply off:

- **Azure Fasv7 (Zen 5, EPYC 9V45):** no vPMU. `perf stat -e cycles` →
  `<not supported>`; `perf_event_open` for a hardware event → **ENOENT**. Not a
  `perf_event_paranoid` gate (that's EACCES), and there's no in-guest knob.
- Dead ends worth recording so nobody re-walks them: **Azure Monitor** "performance
  counters" are OS metrics (CPU%, mem) collected by the agent, not the PMU;
  **`aws/aperf`** is a perf *aggregator* that wraps `perf`, so it hits the same
  wall; **`turbostat`** bails on the too-new Zen 5 (`unsupported platform`,
  `Bzy_MHz=0`).

| VM | PMU via `perf`? | how we measure cyc/byte |
|---|---|---|
| Azure Fasv7 (Zen 5) | ✗ no vPMU | APERF MSR (cycles only) — §5 |
| Zen 2 / T4 node | ✓ | `perf_event` — full events *(pending run)* |
| Azure HB-series (Zen HPC) | ✓ (+ AMD uProf) | `perf_event` / uProf — deep events *(pending)* |

## 5. The APERF workaround (no PMU → still get cycles)

- **APERF** (`IA32_APERF`, MSR `0xE8`) counts *actual unhalted core cycles* — the
  same quantity as the PMU's `CPU_CLK_UNHALTED`, but it's a **P-state MSR, not a
  PMC**, so it's readable via `/dev/cpu/N/msr` even when `perf_event_open` isn't.
  On the Fasv7, `sudo rdmsr -p0 0xE8` works and increments; that's the door Azure
  left open. AMD implements it identically on Zen.
- It gives **cycles, and only cycles** → cycles/byte. It does *not* give
  instructions/byte, IPC, cache/branch stats, per-port µops, or sampling — those
  still need a real PMU (§7).
- Harness: `benches/aperf.rs`, a criterion `Measurement` that reads APERF via
  `pread` on `/dev/cpu/N/msr` at start/end of each sample (delta = cycles). Pins to
  a core (`AES_APERF_CPU`, default 0) so the work accrues on the core we read; run
  under `sudo`. Enable with `--features aperf`.

## 6. The harness: a criterion custom `Measurement`

- criterion has **one pluggable measurement per run**; its `Value` collapses to a
  single `f64` that all its stats operate on. So each run yields one metric family:
  wall-clock → `time` + `GiB/s`; APERF → `cycles` + `cycles/byte`. To get all four
  columns you run **twice and merge** (GiB/s from the wall-clock run, cyc/byte from
  the APERF run) — which we cross-check by confirming they imply the same clock.
- Two backends, same design: `benches/perf.rs` (`--features perf`, `perf_event` —
  for PMU-capable boxes; also does instructions/byte) and `benches/aperf.rs`
  (`--features aperf`, APERF — for no-PMU boxes). Both report cyc/byte directly.
- **Thread-scoping trap:** either counter is per-thread, so it's meaningless for the
  multi-threaded `*-parallel` variants (it'd miss the rayon workers). The throughput
  bench drops those under either counter feature; post 2 needs per-CPU counting.

## 7. The clock is not constant — which is *why* you count cycles

- The headline reason to measure rather than derive: on the Fasv7 the clock isn't
  fixed. A single busy core boosts to **~4.45 GHz** (not the ~4.17 all-core figure),
  and it **drifts ~2% lower on the AVX-512 rungs** (more power). Measured live from
  `/proc/cpuinfo` MHz under a pinned load.
- Deriving cyc/byte from GiB/s × one assumed clock was wrong by ~9%: post 1's
  original "~99% of ceiling" was really ~93%. Counting actual cycles sidesteps it
  entirely — *including* the per-rung boost drift a single-clock assumption can't
  capture. This note is itself the cautionary tale.

## 8. Pending: multi-machine PMU results *(needs PMU-capable boxes)*

- **Zen 2 / T4 node** (has a PMU; cheap, do next): `--features perf` → real
  `perf_event` cyc/byte **and instructions/byte** on Zen 2. Confirms APERF ≈ PMU
  where both exist, and gives the cross-generation contrast (post 1's ~70% aside).
- **Azure HB-series** (Zen HPC, PMU + [AMD uProf] documented; more involved —
  cost/provisioning): the *deep* counters — per-port µops to **prove rung 4's
  port-saturation**, plus IPC and instructions/byte. `perf` works if the PMU is
  exposed; uProf is optional (IBS sampling, nicer UI).
- Deliverable: the §4 table filled in with measured cyc/byte (± instructions/byte)
  across Fasv7 (APERF) / Zen 2 / HB, and the rung-4 port-µop proof.

[AMD uProf]: https://techcommunity.microsoft.com/blog/azurehighperformancecomputingblog/profiling-on-hb-series-with-amd-uprof/3707496

---

### Data dependencies before this can be published
- [ ] `--features perf` run on the Zen 2 / T4 node (PMU cyc/byte + instructions/byte).
- [ ] HB-series run (deep counters; the rung-4 port-saturation proof).
- [ ] Fill in the §4 counter-availability table with measured numbers.
