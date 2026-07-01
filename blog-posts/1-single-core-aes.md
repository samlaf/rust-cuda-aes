# From 0.47 to 49 GiB/s: maxing out AES-128 on a single core

*Working titles: "One core, 100×" / "Climbing the AES-NI ladder, then going wider with VAES"*

> This is the first post in a series. We're porting Cihangir Tezcan's table-based
> AES-128 CUDA implementation ([eprint 2021/646]) to Rust + [rust-cuda], and before
> we can honestly say "the GPU is fast" we need honest CPU baselines to measure it
> against. This post builds those baselines on **one core** — from a portable
> software implementation to the silicon ceiling, a ~104× climb in six
> steps, each teaching one microarchitectural lesson. The next post spends the
> other cores and runs into the memory wall; the one after that is the GPU.

[eprint 2021/646]: https://eprint.iacr.org/2021/646
[rust-cuda]: https://rust-gpu.github.io/rust-cuda

---

## Why this is a ladder, not a number

This is a progressive-optimization tutorial. Every rung is a *kept* benchmark
variant — nothing is thrown away — so the delta between consecutive rungs is the
whole story. Adding an optimization is one line in a registry; that same registry
is what the throughput benchmark iterates and what the correctness tests walk, so
a new variant is benchmarked *and* known-answer-tested the moment it's listed.

All the code lives in [`crates/aes-cpu/src/aesni.rs`][aesni] and
[`vaes.rs`][vaes]; the variants are registered in `benches/variants.rs` and timed
by `benches/throughput.rs`.

[aesni]: ../crates/aes-cpu/src/aesni.rs
[vaes]: ../crates/aes-cpu/src/vaes.rs

## The setup (briefly)

**AES-128-CTR.** Counter mode turns the block cipher into a keystream generator:

```
ct[i] = pt[i] ⊕ E(k, counter₀ + i)
```

The counter's low 32-bit word is incremented per block (NIST SP 800-38A). The key
property for us: **every block is independent** — block `i` only needs
`counter₀ + i`. That independence is the parallelism we'll exploit twice in this
post (instruction-level, by interleaving and then by widening the instruction) and
again in the next post (thread-level, across cores).

**The correctness gate (non-negotiable).** Every variant must pass the shared
known-answer tests: the FIPS-197 cipher-core vectors (run through CTR via the
`counter₀ = PT, plaintext = 0` trick, so the keystream *is* `E(k, PT)`) and the
NIST SP 800-38A F.5.1 multi-block CTR vector. Plus a cross-variant test that every
kernel agrees byte-for-byte with every other. *Optimization without a correctness
gate is just fast wrongness.*

**The machine.** An Azure `Standard_F8as_v7`: **AMD EPYC 9V45** ("Turin", Zen 5),
**8 physical cores (no SMT)**, single NUMA node. The clock is *not* fixed: a single
busy core boosts to ~4.45 GHz, and even drifts ~2% lower on the AVX-512 rungs
(they draw more power). We don't have to pin that number down, though — **we
measure cycles/byte directly, by counting actual core cycles**, so the clock
divides itself out. (This box exposes no PMU to `perf`, so we read cycles via the
APERF MSR; see the [methods note](0-measuring-cycles-per-byte.md) for how and why.)
The default workload is `N_BLOCKS = 1 << 16` = 1 MiB.

One deliberate choice: **1 MiB is cache-resident.** Input + output (2 MiB) lives in
L3, so a single core is never waiting on DRAM. This post is about *per-core
compute*, so we keep the working set in cache and let cycles/byte tell the truth.
Memory effects are the entire subject of the next post.

## The ladder

> Each rung: what changed, *why* it's faster, and the measured delta. The frame is
> always "the previous rung was bottlenecked on X; this removes X."

### 1. `cpu/scalar` — the software floor

Portable T-table AES: the classic four 256-entry tables that collapse SubBytes +
ShiftRows + MixColumns into table lookups and XORs, plus the S-box for the final
round. No hardware AES; runs anywhere.

**0.474 GiB/s — 8.2 cycles/byte.** This is the "honest software" floor everything
else is measured against.

### 2. `cpu/aesni` — hardware AES, one block at a time

x86 has had dedicated AES silicon since 2010. `_mm_aesenc_si128` does an entire AES
round — SubBytes + ShiftRows + MixColumns + AddRoundKey — in *one instruction*. A
block is an initial XOR, nine `AESENC`s, and one `AESENCLAST`:

```rust
let mut s = _mm_xor_si128(load_words(block), keys[0]);
for &k in &keys[1..10] {
    s = _mm_aesenc_si128(s, k);
}
s = _mm_aesenclast_si128(s, keys[10]);
```

**2.12 GiB/s — 4.5× over scalar.** Dedicated silicon beats table lookups, no
contest. But there's a catch that sets up the next rung: those ten `AESENC`s form a
*single dependency chain* — each waits on the previous one's result. `AESENC` has a
~4-cycle latency, so the core is bound by *latency*, and the two AES units sit
mostly idle. At **1.96 cyc/byte this is only ~16% of the AES-NI throughput
ceiling** (more on that ceiling below).

### 3. `cpu/aesni-x8` — interleave 8 blocks to hide the latency

Latency-bound? Then give the scheduler independent work to fill the gaps. Process
`W = 8` independent counter blocks in lockstep: issue all eight round-`r` `AESENC`s
*before* advancing to round `r+1`. CTR makes the eight blocks independent for free.

```rust
for r in 1..10 {
    for j in 0..W {              // W = 8 independent chains in flight
        s[j] = _mm_aesenc_si128(s[j], keys[r]);
    }
}
```

Why eight? Little's law: to saturate 2 AES units with 4-cycle latency you need
`2 × 4 = 8` `AESENC`s in flight — and eight states plus the round keys is about all
that fits in the 16 XMM registers. (It's why OpenSSL and BoringSSL also interleave
eight.)

The model says this should be a big jump. It isn't: **3.60 GiB/s, only 1.70× over
the single-block version — still just 27% of the ceiling.** That anomaly is the
lead-in to the real story.

### 4. `cpu/aesni-x8-pshufb` — get out of the scalar lane

If interleaving eight independent chains barely helped, then `AESENC` latency was
*not* the bottleneck. So what is?

Throughput on a superscalar core is set by whichever execution port is busiest, not
by instruction count — and `AESENC` is not the busiest port here. The culprit is
the per-block **byte repack** hiding in `load_words`/`store_words`. Our `[u32; 4]`
representation uses a "big-endian within each word" convention, so every block does
four `bswap`s to reorder bytes *and* crosses the integer↔XMM register-file boundary
(`movd`/`pinsrd`, or a store-forwarded stack round-trip). That's ~16 µops per block
landing on the narrow transfer and load/store ports — versus ten `AESENC`s — so
those ports saturate first and the AES units starve.

The fix is to keep the entire per-block dataflow in the SIMD domain:

- **Counter in a register.** Hold it as one `__m128i` and bump the low word with
  `_mm_add_epi32` (a per-lane add is exactly our `wrapping_add`, no carry), instead
  of building a `[u32; 4]` and packing it.
- **One `PSHUFB` for the byte swap.** The "big-endian within word" ↔ AES-state
  reorder is a single `_mm_shuffle_epi8` against a constant, self-inverse mask — in
  the FP domain, replacing all the scalar swaps and register-file crossings.
- **`__m128i` plaintext I/O.** Blocks are contiguous 16 bytes, so load / XOR /
  store directly with `_mm_loadu_si128` / `_mm_xor_si128` / `_mm_storeu_si128`.

```rust
let bswap = _mm_setr_epi8(3,2,1,0, 7,6,5,4, 11,10,9,8, 15,14,13,12);
// initial AddRoundKey, advancing the in-register counter:
*sj = _mm_xor_si128(_mm_shuffle_epi8(ctr, bswap), keys[0]);
ctr = _mm_add_epi32(ctr, step1);
// ... rounds ...
// XOR keystream straight into the contiguous plaintext and store:
let pt = _mm_loadu_si128(blocks.as_ptr().add(i + j) as *const __m128i);
let ks = _mm_shuffle_epi8(*sj, bswap);
_mm_storeu_si128(out.as_mut_ptr().add(i + j) as *mut __m128i, _mm_xor_si128(pt, ks));
```

**12.17 GiB/s — 3.4× over `x8`, landing at 0.341 cyc/byte ≈ 92% of the AES-NI
ceiling.** The AES units are *finally* the bottleneck. The repack was partly an
artifact of our own byte convention, but the lesson generalizes: **the data
marshalling around a SIMD kernel is the cost beginners under-weight, because it
isn't named "AES."**

### 5. `cpu/vaes256-x8-pshufb` — two blocks per instruction

We're at the 128-bit ceiling. The only way up is to make each instruction do more
work — which is exactly what VAES (vectorized AES) is: the same AES-round
instructions, but applied to *every* 128-bit lane of a wider register at once.
`_mm256_aesenc_epi128` runs one AES round on **two** independent blocks.

This is the same kernel as the last rung — same 8-way interleave, same `PSHUFB`
dataflow (now `_mm256_shuffle_epi8`, the byte swap broadcast into both lanes) —
just lifted into 256-bit registers. The AES round is the only thing that widened:

```rust
for r in 1..10 {
    for sj in s.iter_mut() {                 // each __m256i = 2 blocks
        *sj = _mm256_aesenc_epi128(*sj, keys2[r]);
    }
}
```

**23.96 GiB/s — 1.97× over the 128-bit path, 0.173 cyc/byte ≈ 90% of the
(now-halved) 256-bit ceiling.** Doubling the instruction width doubled throughput.

### 6. `cpu/vaes512-x8-pshufb` — four blocks per instruction

`_mm512_aesenc_epi128` runs one AES round on **four** blocks. Same kernel again, now
in 512-bit registers (`_mm512_shuffle_epi8` for the swap):

```rust
*sj = _mm512_aesenc_epi128(*sj, keys4[r]);   // each __m512i = 4 blocks
```

**49.08 GiB/s — 2.05× over the 256-bit path, 4.0× over 128-bit, and 0.0841 cyc/byte
≈ 93% of the 512-bit ceiling.** This is the per-core apex.

The clean *doubling at every width* — 12.2 → 24.0 → 49.1 GiB/s — is the headline.
It only happens if a 512-bit `VAESENC` retires *as fast* as a 128-bit one while
doing 4× the work, which requires a **native 512-bit datapath**. That's the whole
reason this ran on Zen 5 and not Zen 4: Zen 4's AVX-512 is "double-pumped" (the
physical units are 256-bit, so a 512-bit op decodes into two), and the 256→512 step
there would have been a flat line. Zen 5 widened the datapath to a true 512 bits.

## How good is this, really? Cycles/byte

GiB/s hides the clock; **cycles/byte** is the fair, clock-normalized metric — and
we measure it *directly*, counting actual core cycles (via the APERF MSR, since this
VM exposes no PMU; [see the methods note](0-measuring-cycles-per-byte.md)), so it
isn't even inferred from a clock. The AES-NI throughput ceiling is set by the AES
execution units:

```
ceiling = 10 rounds × 0.5 (reciprocal throughput) ÷ bytes-per-instruction
```

For 128-bit (16 B/instr) that's **0.3125 cyc/byte**; VAES halves it at each width.

| rung | throughput | cyc/byte | % of width's ceiling |
|---|---|---|---|
| `cpu/scalar` | 0.474 GiB/s | 8.90 | — |
| `cpu/aesni` (x1) | 2.12 GiB/s | 1.96 | 16% |
| `cpu/aesni-x8` | 3.60 GiB/s | 1.15 | 27% |
| `cpu/aesni-x8-pshufb` (128-bit) | 12.17 GiB/s | 0.341 | **92%** |
| `cpu/vaes256-x8-pshufb` (256-bit) | 23.96 GiB/s | 0.173 | **90%** |
| `cpu/vaes512-x8-pshufb` (512-bit) | 49.08 GiB/s | 0.0841 | **93%** |

*(AES-NI rungs vs the 0.3125 128-bit ceiling; VAES vs the width-adjusted 0.15625 /
0.078125.)*

This isn't hand-waving about reciprocal throughput — [uops.info] confirms it for
Zen 5: `VAESENC` at **every** width (XMM/YMM/ZMM) is **TP 0.50, latency 4, one
µop, ports FP0/1**. Two ports × one per cycle = the "2 AES units" the ceiling
assumes, and TP staying 0.50 even for ZMM is the native-datapath proof straight
from the silicon tables. (Latency 4 × 2 ports = 8 ops to saturate — which is
exactly our interleave width.)

[uops.info]: https://uops.info

**A cross-generation aside.** On an older Zen 2, the `pshufb` rung only reached
~70% of the 128-bit ceiling — Zen 5's wider ports and better store-forwarding take
the same code to ~92%. On a modern core, once you remove the repack you're within a
hair of the silicon limit; there's no big slack left to chase.

## Comparison with the paper

The Tezcan paper is a GPU paper, but it reports CPU AES-NI references and states
its CPU kernel runs at **1.30 cyc/byte ≈ 24% of the AES-NI ceiling**, "unchanged
across six CPU generations" — i.e. an under-interleaved reference. At the *same*
128-bit instruction width, our `pshufb` rung is 0.341 cyc/byte (92%): **~3.8× more
efficient per core**, before VAES even enters the picture. With VAES-512 it's
0.084.

Cycles/byte is the fair comparison here — the clock is divided out and both sides
are compute-bound and cache-resident. *Aggregate* Gbps is a different story,
confounded by core count, clock, SMT, and (crucially) the buffer size, which the
paper never states. We'll untangle that in the next post, where buffer size is the
whole point.

## Where this leaves us

The single-core climb, in one breath: software T-tables → hardware AES → interleave
to hide latency → **kill the scalar repack** (the real win) → widen the instruction
with VAES, twice → the silicon ceiling. Net **~104×**, ending at **~93% of the
per-core hardware maximum**.

The transferable lesson is rung 4: the named "real work" (`AESENC`) is cheap and
fully pipelined; the *data marshalling around it* is the actual cost, and it's the
part that doesn't show up in a naive instruction count.

But notice what we carefully avoided: this is **one core, with the data sitting in
cache**. 49 GiB/s on one of eight cores. The obvious next move is to spend the other
seven — CTR is embarrassingly parallel, so it should be nearly free. It is not. The
moment the working set spills the cache, a completely different bottleneck appears,
one that no amount of compute can fix.

**Next:** [spending the other cores, and hitting the memory wall →](2-multi-core-aes.md)
