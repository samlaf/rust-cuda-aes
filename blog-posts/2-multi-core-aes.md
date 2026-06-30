# Spending the other cores — and hitting the memory wall

*Working title. Post 2 of the series. **OUTLINE** — the multi-core numbers exist at
1 MiB, but the memory-wall section needs the block-size sweep (a follow-up PR)
before it can be written for real. Markers below flag what's pending data.*

> Recap from [post 1](1-single-core-aes.md): one core, after six optimizations,
> hit 49 GiB/s of AES-128-CTR — ~99% of the silicon ceiling — with the data sitting
> in cache. This post spends the other seven cores. The twist: per-core efficiency
> and core count *don't* simply multiply, and finding out why is the whole point.

---

## 1. CTR is embarrassingly parallel

- Block `i` only needs `counter₀ + i`, so the buffer splits into independent
  chunks: give each chunk a counter pre-advanced to its first block
  (`counter_block(counter₀, chunk_start)`) and run the single-core VAES kernel on
  each chunk on its own core, via rayon. One chunk per worker, rounded up to a
  whole number of interleave groups.
- **Correctness is provable, not hopeful:** the per-chunk local counter maps to the
  same global counter as the serial path by the truncation homomorphism
  `(a + b) mod 2³² = (a mod 2³²) + (b mod 2³²)`. Byte-identical output, gated by the
  `*_parallel_matches_serial` test + the shared KAT.
- **Why it should compose:** per-core efficiency (VAES + pshufb) and core count
  (rayon) are orthogonal axes — in principle they multiply.

## 2. The 1 MiB numbers — and a surprise

At the cache-resident 1 MiB workload (8 cores, Zen 5 Fasv7):

| kernel | 1 core | 8 cores (parallel) | scaling |
|---|---|---|---|
| `aesni-x8-pshufb` | 12.17 | 53.43 GiB/s | 4.4× |
| `vaes256-x8-pshufb` | 23.96 | 71.96 GiB/s | 3.0× |
| `vaes512-x8-pshufb` | 49.08 | 89.72 GiB/s | **1.8×** |

- The parallel speedup **collapses as the per-core kernel gets faster** (4.4× →
  3.0× → 1.8× on the same 8 cores). The better post 1's kernel, the worse multi-core
  looks. Why?
- Two confounds at 1 MiB, neither of which is "real" scaling:
  1. **Rayon fork-join overhead.** `vaes512-parallel` finishes 1 MiB in ~11 µs;
     spawning and joining 8 threads *per iteration* is a meaningful fraction of
     that. The faster the kernel, the smaller the task, the more overhead dominates.
  2. **Shared-L3 bandwidth.** 1 MiB is L3-resident, so eight cores are contending
     for L3, not running free.
- So the 1 MiB parallel numbers *understate* real scaling — and to see the truth we
  have to vary the working-set size.

## 3. The block-size sweep  *(PENDING — needs the harness PR)*

- Switch `throughput.rs` to criterion's `bench_with_input` over a size range
  (`1<<10 … 1<<24`) with per-input `Throughput::Bytes`, charting throughput vs
  working-set size. The cache→DRAM transition then shows up directly.
- Expectation, two regimes:
  - **Cache-resident (small):** compute-bound; multi-core scaling limited mostly by
    fork-join overhead (recovers as the task grows and amortizes it).
  - **Spills L3 (large):** the **memory wall** — aggregate throughput flattens at
    `DRAM bandwidth ÷ traffic-multiplier`, independent of core count.
- [Plot goes here once collected.]

## 4. Roofline: AES-CTR has fixed, low arithmetic intensity

- The right lens for the compute→memory transition is the **roofline model**
  (Williams, Waterman & Patterson, 2009 — invented *for multicore CPUs*, not GPUs).
  Achievable throughput = `min(peak compute, arithmetic_intensity × bandwidth)`.
- **Arithmetic intensity (AI)** = work per byte of memory traffic. AES-128-CTR's is
  fixed and low: ~10 `AESENC` per 16-byte block against ~32–48 B of DRAM traffic
  (plaintext read + ciphertext write + the RFO). The algorithm pins it — you can't
  trade more compute for the same data. *(Axis note: classic roofline is FLOP/byte;
  AES has no FLOPs, so plot AESENC-µops or AES-rounds per byte — same shape, just
  state it.)*
- Two roofs: a flat **compute roof** (peak AES throughput, AI-independent) and a
  *sloped* **memory-bandwidth roof** (`AI × bandwidth`); attainable = the lower of
  the two, and they meet at the **ridge point** (AI = peak_compute ÷ bandwidth).
- The subtlety is that there's a *separate* sloped roof per memory level — L1
  highest, DRAM lowest, since cache bandwidth ≫ DRAM bandwidth (the *hierarchical*
  roofline). AES's AI never moves (fixed vertical line, far left); **which sloped
  roof applies** is set by where the working set lives:
  - **Cache-resident (L2/L3):** the cache roof is so high that even AES's low AI
    clears the ridge and hits the flat **compute** roof → compute-bound. Post 1's
    49 GiB/s `vaes512` ceiling.
  - **DRAM-resident (large buffer):** the DRAM roof is far lower, so AES's *same*
    low-AI line now lands on the sloped **DRAM** roof, below the compute roof →
    memory-bound. The wall.
  - The block-size sweep walks the working set down the hierarchy, so throughput
    descends from the cache roof onto the DRAM roof — the roofline made visible.
- **The unifying payoff:** short of buying more bandwidth, the only way past the
  wall is to **raise AI** — which is exactly what §5's experiments do (keystream-only
  removes the plaintext-read stream; NT-stores remove the RFO), and the same lever
  as the GPU's `R > 1` (amortizing table/key traffic) in post 3. Roofline ties the
  wall, both CPU experiments, and the headline GPU optimization into one picture.
- **CPU caveat:** on the AES-NI/VAES path the round keys are in registers and there
  are no table loads (only the scalar variant touches tables), so CPU AI is pinned
  by plaintext/ciphertext streaming — there's no `R`-style *reuse* knob here. You
  raise AI by changing *what data moves*, not by reusing loads; reuse is the GPU's
  lever.

## 5. The memory wall  *(PARTIAL — bandwidth measured, AES-at-large-buffer pending)*

- Measured Zen 5 slice DRAM bandwidth (sysbench, 1 GiB block ≫ L3, 4 threads):
  read ≈ 112 GiB/s; write noisy (cloud + first-touch), needs a clean STREAM run.
  *(TODO: STREAM `Copy`/`Triad` at 8 threads for the quotable number.)*
- The traffic multiplier (the sharp insight to keep): AES-CTR with **normal stores**
  reads plaintext, writes ciphertext, *and* the write triggers a read-for-ownership
  (RFO) — so ~3× the encrypted bytes hit DRAM. On the Zen 2 box this matched
  precisely: the 10.87 GiB/s wall × 3 ≈ 32.6 GiB/s ≈ the measured read-bandwidth
  ceiling. Re-confirm the multiplier on Zen 5.
- **Two wall-raising experiments** the analysis predicts (both worth their own rung):
  - **Non-temporal stores** (`_mm_stream_si128`): kill the RFO → 3× drops to 2× →
    the wall should rise by ~1.5×.
  - **Keystream-only** (no plaintext load — counter derived in-register): drops a
    whole read stream; its large-buffer throughput should jump if we're truly
    bandwidth-bound. (Also the clean analog of the GPU's no-upload path.)
- **The kicker:** because post 1's kernel is so fast, a *single* `vaes512` core
  already demands ~100 GB/s of DRAM traffic — likely more than this slice delivers.
  So on Zen 5 the wall arrives at one or two cores, not eight. The faster the
  kernel, the fewer cores it takes to turn AES into a memory problem.

## 6. The lesson

- Once the per-core kernel is compute-optimal, **adding cores converts a compute
  problem into a memory-bandwidth problem.** "Throughput" is meaningless without
  stating the working-set size — the *same binary* swings ~2× between cache-resident
  and DRAM-resident.
- Roofline says this is structural, not a tuning miss: AES-CTR's arithmetic
  intensity is fixed and low, so once the data falls out of cache there's no knob
  (short of changing what data moves) that keeps you on the compute roof.
- This recontextualizes the paper comparison from post 1: cycles/byte was the fair
  compute metric; aggregate Gbps is a memory-bandwidth (and core-count, and
  buffer-size) measurement wearing an AES costume.

## 7. Cliffhanger → the GPU

- A CPU — even eight Zen 5 cores at the silicon ceiling — tops out at *tens* of GB/s
  of DRAM bandwidth on this slice. An RTX-class GPU has on the order of **20× the
  memory bandwidth**. When the bottleneck is memory, not compute, that's the lever
  that matters.
- **Next:** [breaking the memory wall on the GPU →](3-gpu.md)

---

### Data dependencies before this can be published
- [ ] Block-size sweep harness (`bench_with_input`) — the throughput-vs-size curves.
- [ ] Clean STREAM `Copy`/`Triad` on the Zen 5 slice (8 threads).
- [ ] AES throughput at large buffers (16/64/256 MiB) on Zen 5 — the actual wall.
- [ ] Optional rungs: non-temporal-store variant; keystream-only variant.
