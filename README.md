# Rust-Cuda AES

A [rust-cuda](https://rust-gpu.github.io/rust-cuda) port of Cihangir Tezcan's
table-based AES-128 CUDA implementation, built up one optimization at a time,
with CPU baselines (scalar + VAES) to benchmark the GPU kernels against.

## Layout

Host-side library crates live under `crates/`; the repo root is a virtual
workspace. The two crates that aren't host workspace members sit at the top
level: `benches/` (all benchmarks) and `kernels/` (the GPU device code).

| Crate | Target | Role |
|-------|--------|------|
| `crates/aes-core/` | portable, `#![no_std]` | Shared primitives: constant tables, key schedule, `encrypt_block` (round function) + `encrypt_ctr_block` (CTR) + the FIPS-197 and NIST F.5.1 KAT vectors. Depended on by everything; no deps of its own. |
| `crates/aes-cpu/` | host (std) | CPU backends: `scalar` (portable T-table CTR), `aesni` (AES-NI hardware AES + interleaved/`PSHUFB`/multi-core kernels, x86_64), and `vaes` (wider VAES — 2 blocks/instr via 256-bit, 4 blocks/instr via 512-bit, + multi-core, x86_64). VAES verified on Azure Fasv7 (Zen 5 EPYC 9V45): all KATs pass, byte-identical to the AES-NI reference. |
| `crates/aes-gpu/` | host (std) | GPU backend **library**: builds the PTX (`build.rs`), embeds it, and exposes `AesGpu::encrypt_ctr` to upload inputs and launch the CTR kernel. Round-trip tests against FIPS-197 + NIST F.5.1. |
| `kernels/` | `nvptx64` (GPU) | The GPU kernel (`aes128_ctr`), compiled to PTX. Depends on `aes-core`. Standalone (its own workspace, not a member) so `CudaBuilder` builds it in isolation. |
| `benches/` | host (std) | `aes-bench`: an `Aes`-variant registry (`variants.rs`) plus criterion benches that iterate it — `throughput.rs` (every backend in one comparable group, GPU variants behind the `gpu` feature) and `latency.rs` (GPU single-block; `gpu`-only). |

The same `aes_core::encrypt_ctr_block` (which calls `encrypt_block`) runs on the
GPU, in the CPU scalar backend, and in the tests, so the known-answer tests
exercise the exact code the kernel runs.

All benchmarks live together in the top-level `benches/` crate (`aes-bench`).
Every kept backend is registered once in `variants.rs` as a named `Aes`
implementation; `throughput.rs` iterates that registry into a single `aes128-ctr`
group, so every variant — CPU scalar today, plus VAES and the GPU
kernels as they land — is reported side by side and directly comparable. Adding
an optimization step is a one-line entry in the registry, and the same registry
is what the cross-backend known-answer test walks, so a new variant is benchmarked
and correctness-checked the moment it's listed. `latency.rs` reports a separate
`aes128-latency` group timing the CTR kernel over a single block — the fixed
per-launch round-trip overhead that batching amortizes. The GPU variants are
behind a `gpu` feature, so the CPU comparison builds and runs without CUDA.

## Test & benchmark

Logic tests + CPU benchmark — table generation and the shared encryption logic
against the shared FIPS-197 known answers (host-only, **no GPU/CUDA required**):

```bash
cargo test  -p aes-core
cargo test  -p aes-cpu
cargo test  -p aes-bench              # KAT over every CPU variant in the registry
cargo bench -p aes-bench --bench throughput          # wall-clock: time + GiB/s
```

Cycles/byte instead of wall-clock — swap criterion's timer for a hardware cycle
counter (Linux only; reports **cycles/byte** directly, no clock assumption). Two
backends, chosen by whether the host exposes a PMU (see the
[methods note](blog-posts/0-measuring-cycles-per-byte.md)):

```bash
# perf_event PMU — where a PMU is exposed (bare metal, the Zen 2 / T4 node).
# cycles/byte, or AES_PERF_EVENT=instructions for instructions/byte:
cargo bench -p aes-bench --features perf --bench throughput
AES_PERF_EVENT=instructions cargo bench -p aes-bench --features perf --bench throughput

# APERF MSR — fallback for hosts with no exposed PMU (e.g. the Azure Zen 5 slice).
# Reads actual core cycles via /dev/cpu/N/msr, so it needs the msr module + root.
# Build as your user (to resolve the toolchain), then run the built binary as root:
sudo modprobe msr
BIN=$(cargo bench -p aes-bench --features aperf --bench throughput --no-run \
        --message-format=json 2>/dev/null \
      | jq -r 'select(.executable and .target.name=="throughput") | .executable' | tail -1)
sudo taskset -c 0 "$BIN" --bench          # pins to AES_APERF_CPU (default 0)
```

Both counter backends report **cycles/byte** and drop the multi-threaded
`*-parallel` variants (a per-thread counter can't see the rayon workers). `perf`
and `aperf` are mutually exclusive; if both are set, `aperf` wins.

GPU round-trip test + benchmarks — builds the kernel and runs it on the device,
adding the GPU variants to the same `aes128-ctr` comparison (needs an NVIDIA GPU
+ CUDA toolkit):

```bash
cargo test  -p aes-gpu
cargo test  -p aes-bench --features gpu          # KAT over CPU + GPU variants
cargo bench -p aes-bench --features gpu --bench throughput
cargo bench -p aes-bench --features gpu --bench latency
```

## Status

**Step 2 — CTR mode (current).** Counter mode replaces the ECB baseline: the
cipher is applied to a per-block counter and the keystream is XORed into the
plaintext (`ct[i] = pt[i] ⊕ E(k, counter₀ + i)`, with the counter's low 32-bit
word incremented per block, NIST SP 800-38A convention). A single kernel
`aes128_ctr` subsumes the two former ECB kernels: a runtime `blocks_per_thread`
(`R`) controls how many consecutive counters each thread covers, so `n_blocks = 1`
is the single-block latency path, `R = 1` is one-thread-per-block (the current
benchmark variant, the direct ECB→CTR swap), and `R > 1` (a later step) is the
paper's arithmetic-intensity win. Correctness is gated two ways: the FIPS-197
cipher-core vectors run through CTR via the `counter₀ = PT, pt = 0` trick
(keystream = `E(k, PT)`), and the multi-block NIST SP 800-38A F.5.1 vectors pin
the mode itself. Tables still live in plain global memory.

**Step 1 — naive table-based baseline.** T-tables `T0..T3` for rounds 1–9, S-box
for the last round, all in plain global memory. The S-box, T-tables and `RCON`
are generated at compile time via `const fn` from GF(2⁸) first principles. This
is the round function (`aes_core::encrypt_block`) that CTR now drives.

## TODO / roadmap

Organized by the series post each item serves. Two kinds of work are interleaved:
**optimization rungs** (a new kept variant in `benches/variants.rs` — the story a
post tells) and **harness/measurement** upgrades (orthogonal tooling). Every rung
stays KAT-green throughout.

### Post 1 — single core, the silicon ceiling

Prose done (`blog-posts/1-single-core-aes.md`), with cycles/byte now
APERF-**measured** (not derived; see the methods note). What's left is one
confirmatory sweep:

- [ ] **Interleave-width (`W`) sweep** — `W` is a compile-time const generic, so
      this is *not* a criterion runtime input: add registry variants
      (`cpu/aesni-x{1,2,4,8,16}-pshufb`) or a dedicated tuning bench to confirm the
      Little's-law optimum (≈4 on Skylake, ≈8 on Zen) — the rung-3/4 claim. Reads
      straight off the cyc/byte metric now that the counter harness is in place.

### Post 2 — multi core, the memory wall

- [ ] **Block-size sweep (criterion inputs)** — `throughput.rs` benches a single
      fixed `N_BLOCKS` (`1<<16`); seeing the cache→DRAM *memory wall* currently
      means editing the const and rebuilding. Switch to
      [`bench_with_input`](https://bheisler.github.io/criterion.rs/book/user_guide/benchmarking_with_inputs.html)
      over a range of sizes (e.g. `1<<10 … 1<<22`) with a per-input
      `Throughput::Bytes`, so the report charts throughput vs working-set size
      and the wall shows up directly. (Watch the combinatorics: variants × sizes
      × ~5s each.)
- [ ] **Memory-wall rungs + bandwidth baseline** — the non-temporal-store and
      keystream-only CPU variants (each raises arithmetic intensity), plus a clean
      STREAM `Copy`/`Triad` on the Zen 5 slice for the quotable DRAM number.
      Detailed in `blog-posts/2-multi-core-aes.md`.

### Post 3 — GPU

Each step layers one of the paper's optimizations onto the previous:

- [ ] **GPU `R > 1` sweep** — the `aes128_ctr` kernel already takes
      `blocks_per_thread` (`R`) at runtime; register `gpu/…-range-{4,8,16}`
      variants (one line each) for the paper's arithmetic-intensity win, each
      thread reusing the round keys/tables across `R` consecutive counters.
- [ ] **GPU keystream-only path** — the counter is derived on-device from the
      thread index, so a no-XOR variant needs **no plaintext upload**; that's what
      makes a clean "time the kernel without host transfers" measurement.
- [ ] **Tables → shared memory** — copy the T-tables from global into shared
      memory at kernel start (the paper's ~10× baseline win).
- [ ] **One table + `__byte_perm`** — keep only `T0` in shared memory; derive
      `T1..T3` on the fly via a single byte-rotation instruction.
- [ ] **Bank-conflict-free** — replicate `T0` (and the last-round S-box) across
      all 32 shared-memory banks (`t0S[256][32]`) so each warp lane reads its own
      bank. This is the paper's core contribution.

### Post 0 (methods) + shared harness / infra

- [ ] **Methods note** (`blog-posts/0-measuring-cycles-per-byte.md`, stub written) —
      the "how we measure cyc/byte, and what we can't" reference the series links to:
      cyc/byte vs GiB/s, the ceiling model, PMU vs APERF, the no-vPMU-on-Azure story,
      the criterion custom `Measurement`, and the clock-isn't-constant catch. Pending
      data: the multi-machine PMU runs below.
- [ ] **Multi-machine PMU runs** (feeds the methods note) — `--features perf` on the
      **Zen 2 / T4 node** (has a PMU; cheap) for real `perf_event` cyc/byte **+
      instructions/byte**, then an **Azure HB-series** box (PMU + AMD uProf) for the
      *deep* counters: per-port µops to **prove rung 4's port saturation**, plus IPC.
      APERF (Fasv7) only gives cycles; these give everything APERF can't.
- [ ] **Standalone profiling harness** — `examples/profile.rs` taking
      `<variant> <log2_blocks> <iters>` that loops one kernel, for clean
      `perf stat` / `perf annotate` / `perf record` runs with a controllable size
      (no const edit + rebuild) and zero criterion framework noise. Complements the
      in-criterion counter harness: `perf annotate` here is what *shows* rung-4's
      repack port pressure (post 1); `perf stat` feeds post 2's bandwidth analysis.
- [ ] **Debug info in the bench profile** — `[profile.bench] debug = true` so
      `perf annotate` maps samples back to symbols/source (codegen unchanged).
- [ ] **One clean canonical run** — collect every rung at a fixed `N` in a single
      run for the write-up tables in `blog-posts/`. Post 1's single-core ladder now
      has APERF-**measured** cyc/byte (the `--features aperf` run) alongside the
      wall-clock GiB/s; post 2's multi-core + memory-wall numbers still need the
      block-size sweep to be collected properly.

## References

- C. Tezcan, "Optimization of Advanced Encryption Standard on Graphics
  Processing Units", IEEE Access, 2021. [eprint 2021/646](https://eprint.iacr.org/2021/646)
- Reference CUDA code: https://github.com/cihangirtezcan/CUDA_AES
  (specifically [`128-ctr.cuh`](https://github.com/cihangirtezcan/CUDA_AES/blob/gh-pages/128-ctr.cuh))
- rust-cuda getting started: https://rust-gpu.github.io/rust-cuda/guide/getting_started.html
