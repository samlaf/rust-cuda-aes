# CLAUDE.md

Generic notes for working in this repo. For architecture, crate layout, and the
optimization roadmap, read `README.md`. Per-machine dev setup (e.g. where the GPU
actually runs) lives in a git-ignored `CLAUDE.local.md`, imported here:

@CLAUDE.local.md

## Building without CUDA

`crates/aes-gpu`, the `kernels/` PTX, and the `aes-bench` `gpu` feature /
`latency` bench all need a CUDA toolkit and an NVIDIA GPU (`build.rs` runs
`CudaBuilder`, which needs `libnvvm`). Because `aes-gpu` is a workspace member, a
bare `cargo build` / `test` / `clippy` at the root will try to compile it and
fail on a machine without CUDA. **Target the host-only crates explicitly:**

```bash
cargo test   -p aes-core -p aes-cpu -p aes-bench     # logic + CPU variants + KAT
cargo bench  -p aes-bench --bench throughput          # CPU throughput
cargo clippy -p aes-core -p aes-cpu -p aes-bench --benches
```

The `OUT_DIR not set` rust-analyzer error on `crates/aes-gpu/src/lib.rs`
(`include_str!` of the generated PTX) is **expected** when building without CUDA —
the PTX build script never ran. Don't try to "fix" it.

On a CUDA host, the full set also builds:

```bash
cargo test  -p aes-gpu
cargo test  -p aes-bench --features gpu               # KAT over CPU + GPU variants
cargo bench -p aes-bench --features gpu --bench throughput
cargo bench -p aes-bench --features gpu --bench latency
```

## Git workflow

Solo repo with linear history — **commit directly to `main`** (no feature
branches, no PRs). Commit only when asked. Match the existing message style: a
capitalized, sentence-form subject, then a body explaining the *reasoning* for
non-trivial changes.

## Toolchain

Pinned to `nightly-2025-08-04` (see `rust-toolchain.toml`) because rust-cuda
needs that exact nightly plus its components (`rust-src`, `llvm-tools-preview`,
…). Don't bump it casually — it's coupled to the pinned `cuda_std` git rev in
`kernels/Cargo.toml`.

See https://rust-gpu.github.io/rust-cuda/guide/getting_started.html#rust-toolchaintoml

## Where new work goes

This is a **progressive-optimization tutorial**: each step adds one optimization
*and keeps the previous version* so the benchmark delta tells the story.

- Adding an optimization = add a named variant to the `Aes` registry in
  `benches/variants.rs`. It is then automatically benchmarked (`throughput.rs`)
  and correctness-checked against `aes_core::KAT_VECTORS` (the cross-variant KAT
  test). Those are the rails — use them instead of bolting on one-off benches.
- Every variant, CPU or GPU, must pass the shared `aes-core` KATs: the FIPS-197
  `KAT_VECTORS` (cipher core, run through CTR via the `counter₀ = PT, pt = 0`
  trick) and the NIST F.5.1 `CTR_KAT` (the mode itself). Non-negotiable gate.
- Bench files are named by measurement axis (`throughput.rs`, `latency.rs`); the
  cipher mode lives in the criterion group name (currently `aes128-ctr`).
- `kernels/` is intentionally **not** a workspace member — it's its own
  workspace, compiled to PTX for `nvptx64` by `aes-gpu/build.rs`. Don't add it
  to `members`.

## The write-up

The end goal includes a tutorial **article on a Jekyll site — not an mdBook**.
Don't add book/article tooling unless asked. When the time comes, the plan is
plain Jekyll Markdown plus a small extractor that pulls code snippets and
benchmark numbers from the repo so the article can't drift from the code.
