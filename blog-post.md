# Blog post outline — Climbing the AES-NI ladder on the CPU

> Working title ideas: *"Four cores done wrong, or one done right: an AES-NI
> optimization ladder"* / *"From 0.26 to 20 GiB/s: optimizing AES-128-CTR on the
> CPU one step at a time"*

Scope note: this is the **CPU half** of the project — the baselines we build
before the GPU port. It ends where the CPU runs out of road (the memory wall);
that's the hook into the GPU posts. VAES is explicitly deferred (the dev box's
Zen 2 EPYC has no AVX-512/`vaes`).

All code lives in `crates/aes-cpu/src/aesni.rs`; variants are registered in
`benches/variants.rs` and benchmarked by `benches/throughput.rs`.

---

## 1. Hook & motivation

- We're porting Tezcan's table-based AES-128 CUDA paper ([eprint 2021/646]) to
  Rust + rust-cuda. Before the GPU, build **honest CPU baselines** — otherwise
  "the GPU is fast" is meaningless.
- This is a **progressive-optimization tutorial**: every step is a *kept* bench
  variant, so the delta between rungs tells the story. Nothing is thrown away.
- Headline arc: **0.26 → 19.7 GiB/s, a 75× climb on one 4-core CPU**, and every
  rung teaches one microarchitectural lesson. The last rung hits a wall that
  *only* a GPU can break — which is the whole point.

## 2. The setup (keep brief, link to repo)

- **AES-128-CTR.** Why CTR: the cipher becomes a keystream generator —
  `ct[i] = pt[i] ⊕ E(k, counter₀ + i)`. Each block is independent → trivially
  parallelizable, which is exactly what we'll exploit twice (instruction-level,
  then thread-level).
- **The correctness gate (non-negotiable).** Every variant must pass the shared
  `aes-core` KATs: FIPS-197 cipher-core vectors (run through CTR via the
  `counter₀ = PT, pt = 0` trick) and the NIST SP 800-38A F.5.1 multi-block CTR
  vector. Plus a cross-variant "all widths/kernels agree byte-for-byte" test.
  *Optimization without a correctness gate is just fast wrongness.*
- **The harness.** An `Aes` trait + variant registry; `throughput.rs` iterates it
  into one criterion `aes128-ctr` group so every rung is directly comparable.
  Adding a rung = one line in the registry.
- **Test machine:** 4-vCPU AMD Zen 2 EPYC, **3.2 GHz sustained all-core** (no
  AVX/AES all-core throttling — verified via `/proc/cpuinfo` during the run, so
  no clock confound). Default workload `N_BLOCKS = 1<<16` = 1 MiB unless noted.

## 3. The optimization ladder (the heart of the post)

> Each subsection: what changed vs the previous rung, *why* it's faster, the
> code, and the measured delta. Frame every rung as "the previous rung was
> bottlenecked on X; this removes X."

### 3.1 `cpu/scalar` — the software baseline

- Portable T-table AES (the classic 4×256-entry tables collapsing
  SubBytes+ShiftRows+MixColumns into table lookups + XORs). Runs everywhere, no
  hardware AES.
- Purpose: the "honest software" floor. **0.26 GiB/s.**

### 3.2 `cpu/aesni` — hardware AES, one block at a time

- **Change:** replace the entire software round with `_mm_aesenc_si128` — one
  instruction does SubBytes+ShiftRows+MixColumns+AddRoundKey. A block = initial
  XOR + 9× `AESENC` + 1× `AESENCLAST`.
- **Why faster:** dedicated AES silicon vs table lookups. 4.2× over scalar.
- **The catch (sets up the next rung):** it's a *single dependency chain* — each
  `AESENC` waits on the previous one. Bound by `AESENC` **latency** (~4 cyc on
  Zen 2), so the two AES units sit mostly idle. **Latency-bound.**
- Result: **1.10 GiB/s.** (Later: this is 2.70 cyc/byte ≈ only 12% of the AES-NI
  ceiling.)

### 3.3 `cpu/aesni-x8` — interleave 8 blocks to hide latency

- **Change:** process `W = 8` independent counter blocks in lockstep — issue all
  8 `AESENC`s for round *r* before advancing to round *r+1* (`encrypt_ctr<W>`).
  CTR makes the 8 blocks independent for free.
- **Why faster (the key idea):** the loop *nesting order* presents 8 independent
  chains to the scheduler, so the AES units always have ready work — **throughput-
  bound** instead of latency-bound.
- **Callout — why 8, via Little's law:** to saturate 2 AES units with 4-cycle
  latency you need `2 × 4 = 8` AESENCs in flight = 8 independent blocks. 8 is
  also the most that fits without spilling the 16 XMM registers; it's why
  OpenSSL/BoringSSL interleave 8. (W stays a knob — optimum is ~4 on Skylake.)
- **Callout — "can't the CPU reorder these itself?"** Yes, partially: out-of-
  order execution *does* overlap independent work. But it's bounded by the
  finite reorder-buffer/scheduler window and an *in-order* front-end, so it can't
  reliably look 8 iterations ahead. Manual interleaving *guarantees* the
  parallelism is dense and visible instead of hoping the window reaches it.
- Result: **1.93 GiB/s** — but only **1.75×** over x1, *not* the big jump the
  latency-hiding model predicts. That anomaly is the lead-in to the real story.

### 3.4 `cpu/aesni-x8-pshufb` — get out of the scalar lane

- **The diagnosis (the post's "aha").** x8 at 1.93 GiB/s is **1.55 cyc/byte —
  only ~20% of the AES-NI throughput ceiling (0.3125 cyc/byte).** The AES units
  are ~75% idle. The bottleneck isn't `AESENC`; it's the per-block **scalar
  byte-repack** in `load_words`/`store_words`.
- **Why the repack is the bottleneck (the mental-model fix):** throughput on a
  superscalar core is set by *which execution port is busiest*, not instruction
  count. `AESENC` and the repack run on *different* ports. The repack does:
  - the project's "big-endian within each 32-bit word" byte convention forces a
    `to_be_bytes`/`from_be_bytes` swap per word, **and**
  - a crossing of the **integer ↔ XMM register-file boundary** (`movd`/`pinsrd`/
    `pextrd`, or a store-forwarded stack round-trip).
  Those land on a narrow transfer port (and the load/store units) and *can't*
  issue AES work. ~16 such µops/block vs 10 AESENCs → the transfer/LS ports
  saturate first; AES units starve. (Tie to the Zen register-file diagram:
  separate Integer and FP/SIMD register files, with a cross-domain penalty.)
- **Callout — store-to-load forwarding gotcha:** the `[u8;16]` stack path is 4
  narrow stores feeding one 16-byte load — a width mismatch the hardware *can't*
  fast-forward, so it stalls until the stores drain to L1d (~10–15 extra cyc).
- **The fix — keep the whole per-block dataflow in the SIMD domain**
  (`encrypt_ctr_pshufb<W>`):
  1. **Counters in-register:** hold the counter as one `__m128i`, bump the low
     word with `_mm_add_epi32` (per-lane wrap == the project's `wrapping_add`),
     instead of building a `[u32;4]` and packing it.
  2. **One `PSHUFB` for the byte swap:** the BE-within-word ↔ AES-state reorder
     is a single `_mm_shuffle_epi8` against a constant, self-inverse mask — in
     the FP domain, replacing all the scalar swaps + register-file crossings.
  3. **`__m128i` plaintext I/O:** `blocks`/`out` are contiguous 16-byte blocks,
     so `_mm_loadu_si128` → `_mm_xor_si128` → `_mm_storeu_si128` directly. No
     `store_words`, no scalar word-wise XOR.
- **Note:** the repack was partly an artifact of *our* `[u32;4]` byte
  convention; the lesson generalizes — *data marshalling around a SIMD kernel is
  the cost beginners under-weight because it isn't named "AES."*
- **Design aside (why one variant, not two):** the "PSHUFB" and "in-register
  counter / `__m128i` I/O" changes aren't independent — PSHUFB only pays off once
  the data already lives in XMM, which is what the in-register/`__m128i` change
  establishes. They're one transformation ("move the dataflow into SIMD"), so
  they're one rung.
- Result: **6.64 GiB/s — 3.44× over x8**, landing at **0.449 cyc/byte ≈ 70% of
  the hardware ceiling.** The AES units are finally the bottleneck.

### 3.5 `cpu/aesni-x8-pshufb-parallel` — spend the other cores

- **Change:** CTR is embarrassingly parallel, so split the buffer into chunks,
  give each chunk a counter pre-advanced to its first block
  (`counter_block(counter₀, chunk_start)`), and run the pshufb kernel on each
  core via rayon (`encrypt_ctr_pshufb_parallel<W>`). One chunk per worker,
  rounded up to a multiple of W.
- **Correctness:** byte-identical to the serial kernel — provable via the
  truncation homomorphism `(a+b) mod 2³² = (a mod 2³²)+(b mod 2³²)`, so the
  per-chunk local counter maps to the same global counter as the serial path.
  Gated by `parallel_matches_serial` + the shared KAT.
- **Why it composes:** per-core efficiency (PSHUFB) and core count (rayon) are
  **orthogonal** axes — they multiply.
- Result (1 MiB): **19.66 GiB/s, 2.96× over single-core pshufb** (~74% of a
  perfect 4×; the rest is rayon fork-join overhead on a 50 µs task).
- **Design note for the post:** we *retired* the earlier "parallel-on-the-slow-
  kernel" variant. Stacking rayon on the un-optimized x8 also reached ~6.7 GiB/s
  — i.e. **one core done right ≈ four cores done wrong.** That comparison is the
  punchline; the ladder itself should compose (parallel sits on top of pshufb).

## 4. The memory wall (the twist)

- Re-run at **`1<<20` = 16 MiB** (working set 32 MiB, spills the L3):

| buffer | `x8-pshufb` (1 core) | `x8-pshufb-parallel` (4 cores) | parallel scaling |
|---|---|---|---|
| 1 MiB (L3-resident) | 6.64 GiB/s | 19.66 GiB/s | **2.96×** |
| 16 MiB (spills L3) | 6.40 GiB/s | 10.87 GiB/s | **1.70×** |

- **What happened:** single-core barely moves (one core can't saturate DRAM), but
  4 cores collapse from ~3× to 1.7× — they're now contending for shared memory
  bandwidth. The wall ≈ `10.87 × 2 ≈ 21.7 GiB/s ≈ 23 GB/s` of DRAM traffic on
  this 4-vCPU slice.
- **The lesson:** once the per-core kernel is compute-optimal, *adding cores
  turns a compute problem into a memory-bandwidth problem.* "Throughput" is
  meaningless without stating the working-set size — same binary, 1.8× swing.
- This is **the motivation for the GPU**: a CPU tops out at tens of GB/s; an
  RTX-class GPU has ~20× the memory bandwidth.

## 5. How good is 6.64 GiB/s, really? (cycles/byte)

- Gbps/GiB-s hides the truth; **cycles/byte** is the clock-normalized, fair
  metric. Ceiling = `10 AESENC × 0.5 recip-throughput ÷ 16 B = 0.3125 cyc/byte`.
- At 3.2 GHz, single core, 1 MiB:

| kernel | GiB/s | cyc/byte | % of AES-NI ceiling |
|---|---|---|---|
| `cpu/aesni` (x1, naive) | 1.10 | 2.70 | 12% |
| `cpu/aesni-x8` (scalar repack) | 1.93 | 1.55 | 20% |
| **`cpu/aesni-x8-pshufb`** | **6.64** | **0.449** | **70%** |
| AES-NI hardware ceiling | — | 0.3125 | 100% |

- The remaining 30% gap (0.449 → 0.3125): the 2 PSHUFBs/block contend with
  AESENC on the FP pipes, plus loads/stores + loop overhead. Closing it (1
  PSHUFB/block) is fiddly and not worth it — 70% of a hardware ceiling is a good
  place to stop.

## 6. Comparison with the paper (the part that surprised us)

- The paper (a GPU paper) reports CPU AES-NI references: **i7-6700K 90.6 Gbps**
  (4 cores), **i7-10700F 134.7 Gbps** (8 cores), i7-980X 102.4 Gbps.
- Our numbers in Gbps (×8.59 from GiB/s): **169 Gbps @ 1 MiB**, **93 Gbps @
  16 MiB** (4 cores). So at face value we "beat" their 8-core desktop. *How?*
- **Untangling it honestly — three separable reasons:**
  1. **Per-core, we genuinely are ~3× more efficient.** The paper states their
     kernel runs at **1.30 cyc/byte = ~24% of the AES-NI ceiling** ("unchanged
     across six CPU generations") — i.e. an **under-optimized, under-interleaved
     reference**, sitting right between our naive x1 (12%) and our scalar-repack
     x8 (20%). Ours is 0.449 cyc/byte (70%). This is the *fair* comparison
     (cycles/byte, both compute-bound, clock divided out) — and it's the real
     win. ✅
  2. **Our 1 MiB number is cache-inflated.** At a realistic 16 MiB it's 93 Gbps,
     basically tied with the i7-6700K. ✅
  3. **Aggregate Gbps is confounded** by core count, clock, SMT — *and* the paper
     **never states the CPU payload/buffer size** (we read Section III; only the
     i7-980X *reference* is described as "large buffers"). So we can't even pin
     their numbers to a regime. ❌
- **Regime decides the "win":**
  - *Cache-resident (compute-bound):* our 4 efficient cores (ceiling 4 × 6.64 =
    26.6 GiB/s ≈ 228 Gbps; measured 169) **beat** their 8-core 134.7 — because
    `efficiency × cores` favors us despite fewer cores.
  - *Memory-bound (large buffers):* our 4-vCPU slice (~93 Gbps, ~23 GB/s) **loses**
    to their desktop's dual-channel DDR4 (~34 GB/s → 134.7). We're comparing RAM,
    not AES.
- **Correction worth being explicit about in the post:** I initially assumed
  their 134.7 was memory-bound; with the paper in hand that's *under-determined*.
  Their single-core 1.30 cyc/byte is compute-bound/cache-resident; the i7-980X
  "large buffers" figure equals its DDR3 bandwidth (memory-bound); the 6700K/
  10700F could be either. **A throughput figure without a stated payload size
  isn't comparable** — that's itself the methodological takeaway.

## 7. Full results table (canonical)

1 MiB (`1<<16`), 4-core Zen 2 EPYC @ 3.2 GHz, criterion median:

| variant | what it adds | time | throughput | vs. prev | vs. scalar |
|---|---|---|---|---|---|
| `cpu/scalar` | software T-tables | 3.76 ms | 0.26 GiB/s | — | 1× |
| `cpu/aesni` | hardware AES (x1) | 885 µs | 1.10 GiB/s | 4.2× | 4.2× |
| `cpu/aesni-x8` | interleave 8 (hide latency) | 507 µs | 1.93 GiB/s | 1.75× | 7.4× |
| `cpu/aesni-x8-pshufb` | SIMD dataflow (kill repack) | 147 µs | 6.64 GiB/s | 3.44× | 25.5× |
| `cpu/aesni-x8-pshufb-parallel` | + all 4 cores | 49.7 µs | 19.66 GiB/s | 2.96× | 75.6× |

## 8. Conclusions & what's next

- **The CPU ladder, in one breath:** hardware AES (4×) → interleave to hide
  latency (1.75×) → kill the scalar repack to become AES-bound (3.4×) → fan
  across cores (≈3×). Net ~75×, ending at 70% of the per-core hardware ceiling.
- **Two transferable lessons:** (1) the named "real work" (AESENC) is cheap and
  pipelined; the *data marshalling* around it is the actual cost. (2) Make each
  core efficient *before* adding cores — then watch the bottleneck move to
  memory.
- **Deferred:** `cpu/vaes` (2/4 blocks per *instruction* via AVX-512+VAES) — the
  dev box's Zen 2 lacks it; revisit on capable hardware.
- **Next:** the memory wall is why we go to the GPU. Forward-reference the GPU
  posts (tables in shared memory, `__byte_perm`, bank-conflict-free, the `R>1`
  arithmetic-intensity sweep).

---

### Loose ends / TODO before publishing
- [ ] Re-collect all five rungs in a *single* bench run at a fixed N for a clean
      table (current numbers are stitched from several runs; throughput is stable
      but do it properly).
- [ ] Add per-core cyc/byte for the parallel rung? (aggregate cyc/byte is less
      meaningful once memory-bound.)
- [ ] Optional: a keystream-only variant (no plaintext load → half the memory
      traffic) as an explicit memory-wall proof — its `1<<20` throughput should
      jump if we're truly bandwidth-bound.
- [ ] Confirm Zen 2 AESENC recip-throughput (0.5) and latency (~4) against
      uops.info before asserting the 0.3125 ceiling in print.
- [ ] Pull code snippets + numbers from the repo via the extractor (per the
      Jekyll plan) so the post can't drift from the code.
- [ ] Sanity-check the `×8.59` GiB/s→Gbps conversions in the final draft.
