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
| `crates/aes-cpu/` | host (std) | CPU backends: `scalar` (T-table, works) and `vaes` (AES-NI/VAES, x86_64 — TODO). |
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
cargo bench -p aes-bench --bench throughput
```

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

## TODO / next steps

Each step layers one of the paper's optimizations onto the previous, keeping the
KAT green throughout:

- [ ] **CPU VAES backend** — implement `crates/aes-cpu/src/vaes.rs` with AES-NI
      (`_mm_aesenc_si128`), then widen to VAES (`_mm256/512_aesenc_epi128`).
- [x] **CTR mode** — done (Step 2 above). One `aes128_ctr` kernel builds the
      keystream `ct[i] = pt[i] ⊕ E(k, counter₀ + i)` (cipher applied to the
      counter; low-word increment, NIST F.5.1 convention) and subsumes both ECB
      kernels via the runtime `blocks_per_thread` (`R`) knob. Gated on FIPS-197
      (cipher core, via the `counter₀ = PT, pt = 0` trick) **and** the NIST SP
      800-38A F.5.1 multi-block vectors (the mode itself). Deferred follow-ups:
    - **`R > 1` sweep** — the kernel already takes `R` at runtime; register
      `gpu/…-range-{4,8,16}` variants (one line each) for the paper's
      arithmetic-intensity win, each thread reusing the round keys/tables across
      `R` consecutive counters.
    - **Keystream-only path** — the counter is derived on-device from the thread
      index, so a no-XOR variant needs **no plaintext upload**; that's what makes
      a clean "time the kernel without host transfers" measurement.
- [ ] **Tables → shared memory** — copy the T-tables from global into shared
      memory at kernel start (the paper's ~10× baseline win).
- [ ] **One table + `__byte_perm`** — keep only `T0` in shared memory; derive
      `T1..T3` on the fly via a single byte-rotation instruction.
- [ ] **Bank-conflict-free** — replicate `T0` (and the last-round S-box) across
      all 32 shared-memory banks (`t0S[256][32]`) so each warp lane reads its own
      bank. This is the paper's core contribution.
- [x] **GPU benchmarking** — `aes128_ctr` at `R = 1` (one thread per block) is
      benched end-to-end alongside the CPU backends. Next: time the kernel
      without the host transfers, and compare throughput (Gbps) to the paper.

Explicitly out of scope (for now): the exhaustive-search / key-recovery kernels.

## References

- C. Tezcan, "Optimization of Advanced Encryption Standard on Graphics
  Processing Units", IEEE Access, 2021. [eprint 2021/646](https://eprint.iacr.org/2021/646)
- Reference CUDA code: https://github.com/cihangirtezcan/CUDA_AES
  (specifically [`128-ctr.cuh`](https://github.com/cihangirtezcan/CUDA_AES/blob/gh-pages/128-ctr.cuh))
- rust-cuda getting started: https://rust-gpu.github.io/rust-cuda/guide/getting_started.html
