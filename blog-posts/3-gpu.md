# Breaking the memory wall on the GPU

*Working title. Post 3 of the series. **OUTLINE** — written once the GPU kernels
land. The roadmap below mirrors the optimization steps tracked in the repo README.*

> Recap from [post 2](2-multi-core-aes.md): eight Zen 5 cores at the silicon ceiling
> still top out at the slice's DRAM bandwidth — AES-CTR over a large buffer is a
> memory-bandwidth problem, and a CPU has tens of GB/s to offer. A GPU has ~20×
> that. This post ports Tezcan's table-based AES-128 CUDA kernel to Rust + rust-cuda
> and climbs the GPU optimization ladder, one kept variant at a time, KAT-green
> throughout — the same discipline as the CPU posts.

---

## 1. The port

- Tezcan, "Optimization of AES on GPUs", IEEE Access 2021 ([eprint 2021/646]),
  reference CUDA at [cihangirtezcan/CUDA_AES] (`128-ctr.cuh`).
- In Rust via [rust-cuda]: `kernels/` compiles to PTX for `nvptx64`; the host
  `aes-gpu` crate builds/embeds the PTX and launches the CTR kernel. The *same*
  `aes_core::encrypt_ctr_block` runs on the GPU, in the CPU scalar backend, and in
  the tests — so the KATs exercise the exact code the kernel runs.
- Correctness gate unchanged: FIPS-197 + NIST F.5.1, plus byte-for-byte agreement
  with the CPU variants.

[eprint 2021/646]: https://eprint.iacr.org/2021/646
[cihangirtezcan/CUDA_AES]: https://github.com/cihangirtezcan/CUDA_AES
[rust-cuda]: https://rust-gpu.github.io/rust-cuda

## 2. The GPU ladder (each rung = one kept variant)

- **`gpu/0-global-pageable` — the strawman.** T-tables in plain global memory, one
  thread per block (`blocks_per_thread = 1`, the direct ECB→CTR swap), pageable host
  buffers allocated per call. The baseline every later step is measured against.
- **Transfer strategy: pinned vs pageable host memory.** Page-locked buffers for
  faster H2D/D2H; separate the kernel time from the transfer time.
- **`R > 1` arithmetic-intensity sweep.** Each thread covers `R` consecutive
  counters, reusing the round keys and tables across them — the paper's
  arithmetic-intensity win. Register `gpu/…-range-{4,8,16}`.
- **Keystream-only path.** The counter is derived on-device from the thread index,
  so a no-XOR variant needs *no plaintext upload* — the clean "time the kernel
  without host transfers" measurement, and the GPU analog of post 2's keystream-only
  CPU experiment.
- **Tables → shared memory.** Copy the T-tables from global into shared memory at
  kernel start — the paper's ~10× baseline win.
- **One table + `__byte_perm`.** Keep only `T0` in shared memory; derive `T1..T3` on
  the fly via a single byte-rotation instruction.
- **Bank-conflict-free.** Replicate `T0` (and the last-round S-box) across all 32
  shared-memory banks (`t0S[256][32]`) so each warp lane reads its own bank. The
  paper's core contribution.

## 3. Measurement

- **Latency vs throughput:** a separate `aes128-latency` group times a single-block
  launch — the fixed per-launch round-trip overhead that batching amortizes — paired
  with the throughput group.
- The `R` sweep as an arithmetic-intensity story (compute per byte transferred).
- Roofline framing: where each rung sits relative to the GPU's memory bandwidth.

## 4. The payoff vs the CPU

- Drop the CPU posts' wall (tens of GB/s) and the GPU's HBM/GDDR bandwidth on the
  same chart.
- Honest cross-device comparison: cycles/byte doesn't transfer across architectures,
  so compare throughput (and ideally throughput/watt) at a stated, large buffer
  size — the regime where the GPU's bandwidth advantage is the whole point.
- Close the loop on the series: software floor → single-core silicon ceiling (post
  1) → multi-core memory wall (post 2) → GPU bandwidth (here).

---

### Data dependencies before this can be published
- [ ] The GPU kernels themselves (currently: naive global-memory baseline only).
- [ ] Each optimization rung registered + KAT-verified.
- [ ] Throughput + latency runs on the target GPU; the cross-device comparison plot.
