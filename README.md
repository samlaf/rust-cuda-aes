# Rust-Cuda AES

A [rust-cuda](https://rust-gpu.github.io/rust-cuda) port of Cihangir Tezcan's
table-based AES-128 CUDA implementation, built up one optimization at a time.

- Host code: `src/main.rs`
- Kernels + tables: `kernels/src/lib.rs`

Build & run (needs an NVIDIA GPU + CUDA toolkit):

```bash
cargo run --release
```

Test (host-only, no GPU/CUDA required — runs the table generation and the
shared encryption logic against FIPS-197 known answers):

```bash
cargo test --manifest-path kernels/Cargo.toml
```

## Status

**Step 1 — naive table-based baseline (current).** A single thread encrypts one
16-byte block: T-tables `T0..T3` for rounds 1–9, S-box for the last round, all
in plain global memory. Verified against the FIPS-197 known-answer test
(`3243f6a8… → 3925841d…`). The S-box, T-tables and `RCON` are generated at
compile time via `const fn` from GF(2⁸) first principles.

## TODO / next steps

Each step layers one of the paper's optimizations onto the previous, keeping the
KAT green throughout:

- [ ] **CTR mode** — replace the single-block kernel with the counter-mode
      structure from `128-ctr.cuh`: each thread encrypts a range of consecutive
      counter values. This is the real workload to benchmark.
- [ ] **Tables → shared memory** — copy the T-tables from global into shared
      memory at kernel start (the paper's ~10× baseline win).
- [ ] **One table + `__byte_perm`** — keep only `T0` in shared memory; derive
      `T1..T3` on the fly via a single byte-rotation instruction.
- [ ] **Bank-conflict-free** — replicate `T0` (and the last-round S-box) across
      all 32 shared-memory banks (`t0S[256][32]`) so each warp lane reads its own
      bank. This is the paper's core contribution.
- [ ] **Benchmarking** — measure throughput (Gbps) and compare against the
      paper's reported numbers.

Explicitly out of scope (for now): the exhaustive-search / key-recovery kernels.

## References

- C. Tezcan, "Optimization of Advanced Encryption Standard on Graphics
  Processing Units", IEEE Access, 2021. [eprint 2021/646](https://eprint.iacr.org/2021/646)
- Reference CUDA code: https://github.com/cihangirtezcan/CUDA_AES
  (specifically [`128-ctr.cuh`](https://github.com/cihangirtezcan/CUDA_AES/blob/gh-pages/128-ctr.cuh))
- rust-cuda getting started: https://rust-gpu.github.io/rust-cuda/guide/getting_started.html
