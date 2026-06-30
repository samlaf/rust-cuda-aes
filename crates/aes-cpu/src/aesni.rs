//! AES-128-CTR via AES-NI — x86_64 hardware AES.
//!
//! `_mm_aesenc_si128` performs a whole AES round (ShiftRows + SubBytes +
//! MixColumns + AddRoundKey) in one instruction. The single entry point
//! [`encrypt_ctr`] is generic over an **interleave width** `W`: it encrypts `W`
//! independent CTR counter blocks in lockstep (all `W` AESENCs for a round, then
//! the next round).
//!
//! - `W = 1` is the naive, one-block-at-a-time path: a single dependency chain,
//!   so it's bound by the ~4-cycle `AESENC` *latency* and leaves the AES unit(s)
//!   mostly idle.
//! - `W = 8` keeps 8 independent chains in flight, hiding that latency and making
//!   it *throughput*-bound. 8 saturates Zen 2's two AES units (latency 4 ×
//!   2 ops/cycle = 8 in flight) and is the most that fits without spilling the 16
//!   xmm registers (8 states + round keys) — which is why production libraries
//!   (OpenSSL, BoringSSL) interleave 8 in their AES-NI path. The optimum is
//!   microarch-dependent (~4 on Skylake, ~8 on Zen), so `W` is a knob, not a
//!   constant.
//!
//! CTR mode like [`crate::scalar`]: the cipher is applied to the counter and the
//! keystream is XORed into the plaintext, reusing [`aes_core::key_expansion`] and
//! the project's "big-endian within each u32 word" byte convention, so the shared
//! NIST/FIPS KATs pass.
//!
//! [`encrypt_ctr_pshufb`] is the same kernel with the per-block byte handling
//! moved off the scalar/integer ports: counters live in an `__m128i`, the
//! byte-order swap is one `PSHUFB`, and plaintext is XORed as `__m128i`. That
//! removes the [`load_words`]/[`store_words`] register-file crossings that
//! otherwise bottleneck [`encrypt_ctr`] well below the `AESENC` ceiling.
//!
//! [`encrypt_ctr_pshufb_parallel`] is the multi-core wrapper: CTR is
//! embarrassingly parallel (block `i` only needs `counter0 + i`), so it splits
//! the buffer into chunks, hands each chunk a pre-advanced counter, and runs the
//! repack-free `PSHUFB` kernel on every core via rayon. Per-core throughput
//! (`PSHUFB`) and cross-core scaling (rayon) compose — it's the pshufb kernel ×
//! N cores.
//!
//! No VAES here (2/4 blocks per *instruction* via AVX-512) — the dev box's Zen 2
//! lacks it; see [`crate::vaes`].

use core::arch::x86_64::*;
use rayon::prelude::*;

/// Pack 4 big-endian-within-word u32s into a 16-byte AES state. The 16 bytes land
/// in natural order (`w[0]`'s most-significant byte first), which is exactly the
/// state byte order AES-NI operates on.
#[inline]
#[target_feature(enable = "aes")]
unsafe fn load_words(w: [u32; 4]) -> __m128i {
    let mut bytes = [0u8; 16];
    bytes[0..4].copy_from_slice(&w[0].to_be_bytes());
    bytes[4..8].copy_from_slice(&w[1].to_be_bytes());
    bytes[8..12].copy_from_slice(&w[2].to_be_bytes());
    bytes[12..16].copy_from_slice(&w[3].to_be_bytes());
    unsafe { _mm_loadu_si128(bytes.as_ptr() as *const __m128i) }
}

/// Inverse of [`load_words`]: read a 16-byte state back out as 4 big-endian words.
#[inline]
#[target_feature(enable = "aes")]
unsafe fn store_words(v: __m128i) -> [u32; 4] {
    let mut bytes = [0u8; 16];
    unsafe { _mm_storeu_si128(bytes.as_mut_ptr() as *mut __m128i, v) };
    [
        u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
        u32::from_be_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
        u32::from_be_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]),
        u32::from_be_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]),
    ]
}

/// Pack the 44-word schedule from [`aes_core::key_expansion`] into 11 round keys.
///
/// `pub(crate)` so the VAES kernels ([`crate::vaes`]) can broadcast these same
/// `__m128i` round keys into their 256-/512-bit lanes — one byte convention,
/// shared.
#[inline]
#[target_feature(enable = "aes")]
pub(crate) unsafe fn round_keys(rk: &[u32]) -> [__m128i; 11] {
    unsafe {
        let mut k = [_mm_setzero_si128(); 11];
        for (r, kr) in k.iter_mut().enumerate() {
            *kr = load_words([rk[4 * r], rk[4 * r + 1], rk[4 * r + 2], rk[4 * r + 3]]);
        }
        k
    }
}

/// Encrypt one block (the cipher core / keystream generator): initial
/// AddRoundKey, 9 full rounds, then the last round (no MixColumns). Used for the
/// `< W` remainder of [`encrypt_ctr`] and the VAES kernels' scalar tail.
#[inline]
#[target_feature(enable = "aes")]
pub(crate) unsafe fn encrypt_one(rk: &[__m128i; 11], block: [u32; 4]) -> [u32; 4] {
    unsafe {
        let mut s = _mm_xor_si128(load_words(block), rk[0]);
        for &k in &rk[1..10] {
            s = _mm_aesenc_si128(s, k);
        }
        s = _mm_aesenclast_si128(s, rk[10]);
        store_words(s)
    }
}

/// The counter block for position `idx`: `counter0` with its low 32-bit word
/// advanced by `idx` (NIST SP 800-38A low-word increment; matches
/// [`aes_core::encrypt_ctr_block`]). `pub(crate)` so [`crate::vaes`] derives its
/// per-chunk and scalar-tail counters through the same path.
#[inline]
pub(crate) fn counter_block(counter0: [u32; 4], idx: u32) -> [u32; 4] {
    [counter0[0], counter0[1], counter0[2], counter0[3].wrapping_add(idx)]
}

/// Encrypt `blocks` in CTR mode with AES-NI, **interleaving `W` independent
/// blocks** per iteration: `out[i] = blocks[i] ⊕ E(rk, counter0 + i)`.
///
/// `W = 1` is the naive, latency-bound path (one dependency chain); `W = 8` keeps
/// 8 chains in flight to hide the ~4-cycle `AESENC` latency and saturate the AES
/// units (see the module docs for why 8). Result and semantics are identical for
/// every `W`. Mirrors [`crate::scalar::encrypt_ctr`]; `rk` is the 44-word expanded
/// key from [`aes_core::key_expansion`] and `out` must be as long as `blocks`.
///
/// Panics if the CPU lacks AES-NI (the `aes` feature) — every x86_64 server CPU
/// since ~2011 has it; the runtime check just upholds the intrinsics' contract.
pub fn encrypt_ctr<const W: usize>(
    rk: &[u32],
    counter0: [u32; 4],
    blocks: &[[u32; 4]],
    out: &mut [[u32; 4]],
) {
    assert!(
        std::is_x86_feature_detected!("aes"),
        "aes-cpu::aesni requires AES-NI (the `aes` CPU feature)"
    );
    // SAFETY: the runtime check above upholds the intrinsics' `aes` precondition.
    unsafe { encrypt_ctr_impl::<W>(rk, counter0, blocks, out) }
}

#[target_feature(enable = "aes")]
unsafe fn encrypt_ctr_impl<const W: usize>(
    rk: &[u32],
    counter0: [u32; 4],
    blocks: &[[u32; 4]],
    out: &mut [[u32; 4]],
) {
    unsafe {
        let keys = round_keys(rk);
        let n = blocks.len();
        let mut i = 0;
        // Main loop: W independent counter blocks at a time. Issuing all W AESENCs
        // for a round before advancing keeps W independent chains in flight, so
        // the AESENC latency is hidden behind throughput.
        while i + W <= n {
            let mut s = [_mm_setzero_si128(); W];
            for j in 0..W {
                s[j] = _mm_xor_si128(load_words(counter_block(counter0, (i + j) as u32)), keys[0]);
            }
            for r in 1..10 {
                for j in 0..W {
                    s[j] = _mm_aesenc_si128(s[j], keys[r]);
                }
            }
            for j in 0..W {
                s[j] = _mm_aesenclast_si128(s[j], keys[10]);
            }
            for j in 0..W {
                let ks = store_words(s[j]);
                let pt = blocks[i + j];
                out[i + j] = [pt[0] ^ ks[0], pt[1] ^ ks[1], pt[2] ^ ks[2], pt[3] ^ ks[3]];
            }
            i += W;
        }
        // Remainder (fewer than W blocks left): one at a time.
        while i < n {
            let ks = encrypt_one(&keys, counter_block(counter0, i as u32));
            let pt = blocks[i];
            out[i] = [pt[0] ^ ks[0], pt[1] ^ ks[1], pt[2] ^ ks[2], pt[3] ^ ks[3]];
            i += 1;
        }
    }
}

/// Same CTR result as [`encrypt_ctr`], but with the per-block dataflow kept
/// entirely in the SIMD domain — no scalar repack.
///
/// [`encrypt_ctr`] is throughput-bound not on `AESENC` but on the scalar
/// byte-order shuffling in [`load_words`]/[`store_words`]: each block crosses the
/// integer↔XMM register-file boundary (`movd`/`pinsrd`, or a store-forwarded
/// stack round-trip) and runs four `bswap`s, all on ports that then can't issue
/// AES work — so the AES units sit ~75% idle. This kernel removes that:
///
/// - **Counters in-register**: hold the counter as one `__m128i` in native lane
///   order and bump the low word with `_mm_add_epi32` (per-lane wrap = the
///   project's `wrapping_add`, no carry), instead of building a `[u32; 4]` and
///   packing it with [`load_words`].
/// - **One `PSHUFB` for the byte swap**: the "big-endian within each word" ↔ AES
///   state-order reorder is a single `_mm_shuffle_epi8` against a constant mask
///   (which is its own inverse), in the FP domain, replacing the 4×`to_be_bytes`
///   / `from_be_bytes` + register-file crossing.
/// - **`__m128i` plaintext I/O**: `blocks`/`out` are contiguous 16-byte blocks,
///   so load/XOR/store directly with `_mm_loadu_si128`/`_mm_xor_si128`/
///   `_mm_storeu_si128` — no [`store_words`], no scalar word-wise XOR.
///
/// Byte-identical to [`encrypt_ctr`] for every `W` (same KAT gate). Needs SSSE3
/// for `PSHUFB` on top of AES-NI; every AES-NI CPU has it, the runtime check just
/// upholds the intrinsics' contract.
pub fn encrypt_ctr_pshufb<const W: usize>(
    rk: &[u32],
    counter0: [u32; 4],
    blocks: &[[u32; 4]],
    out: &mut [[u32; 4]],
) {
    assert!(
        std::is_x86_feature_detected!("aes") && std::is_x86_feature_detected!("ssse3"),
        "aes-cpu::aesni pshufb path requires AES-NI + SSSE3"
    );
    // The kernel writes `out` via raw `storeu` (no bounds check), so this length
    // guard is what keeps those stores in bounds.
    assert_eq!(blocks.len(), out.len(), "out must be as long as blocks");
    // SAFETY: the runtime check above upholds the `aes` + `ssse3` preconditions.
    unsafe { encrypt_ctr_pshufb_impl::<W>(rk, counter0, blocks, out) }
}

#[target_feature(enable = "aes,ssse3")]
unsafe fn encrypt_ctr_pshufb_impl<const W: usize>(
    rk: &[u32],
    counter0: [u32; 4],
    blocks: &[[u32; 4]],
    out: &mut [[u32; 4]],
) {
    unsafe {
        let keys = round_keys(rk);
        let n = blocks.len();
        // Reverse the 4 bytes within each 32-bit lane: native (little-endian word)
        // order ↔ AES state order. Self-inverse, so the same mask does both ways.
        let bswap = _mm_setr_epi8(3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12);
        // +1 in word index 3 (the low CTR word, per `counter_block`); per-lane add.
        let step1 = _mm_set_epi32(1, 0, 0, 0);
        // The counter for block i, in native lane order (lane k = counter0[k]).
        let mut ctr = _mm_set_epi32(
            counter0[3] as i32,
            counter0[2] as i32,
            counter0[1] as i32,
            counter0[0] as i32,
        );
        let mut i = 0;
        while i + W <= n {
            // Initial AddRoundKey for W counters, advancing the in-register counter.
            let mut s = [_mm_setzero_si128(); W];
            for sj in s.iter_mut() {
                *sj = _mm_xor_si128(_mm_shuffle_epi8(ctr, bswap), keys[0]);
                ctr = _mm_add_epi32(ctr, step1);
            }
            for r in 1..10 {
                for sj in s.iter_mut() {
                    *sj = _mm_aesenc_si128(*sj, keys[r]);
                }
            }
            for sj in s.iter_mut() {
                *sj = _mm_aesenclast_si128(*sj, keys[10]);
            }
            // XOR each keystream (swapped back to native order) straight into the
            // contiguous plaintext block and store it — no scalar word handling.
            for (j, sj) in s.iter().enumerate() {
                let pt = _mm_loadu_si128(blocks.as_ptr().add(i + j) as *const __m128i);
                let ks = _mm_shuffle_epi8(*sj, bswap);
                _mm_storeu_si128(out.as_mut_ptr().add(i + j) as *mut __m128i, _mm_xor_si128(pt, ks));
            }
            i += W;
        }
        // Remainder (fewer than W blocks left): the cold scalar path is fine here.
        while i < n {
            let ks = encrypt_one(&keys, counter_block(counter0, i as u32));
            let pt = blocks[i];
            out[i] = [pt[0] ^ ks[0], pt[1] ^ ks[1], pt[2] ^ ks[2], pt[3] ^ ks[3]];
            i += 1;
        }
    }
}

/// Like [`encrypt_ctr_pshufb`] but fans the work across CPU cores with rayon —
/// the per-core (`PSHUFB`) and cross-core (rayon) wins composed. CTR is
/// embarrassingly parallel: block `i` always uses `counter0 + i`, so the chunk
/// starting at block `start` is just an independent [`encrypt_ctr_pshufb`] whose
/// counter is pre-advanced to `counter_block(counter0, start)`. Each core runs
/// the exact same repack-free `W`-wide kernel, so the result is byte-identical to
/// `encrypt_ctr::<W>` — only faster by ~the core count.
///
/// Chunks are sized to one per worker thread, rounded up to a whole number of
/// `W`-wide groups so the main interleaved loop stays full-width in every chunk
/// and only the final chunk can carry a `< W` remainder.
///
/// Panics if the CPU lacks AES-NI / SSSE3, or if `out.len() != blocks.len()`.
pub fn encrypt_ctr_pshufb_parallel<const W: usize>(
    rk: &[u32],
    counter0: [u32; 4],
    blocks: &[[u32; 4]],
    out: &mut [[u32; 4]],
) {
    assert!(
        std::is_x86_feature_detected!("aes") && std::is_x86_feature_detected!("ssse3"),
        "aes-cpu::aesni pshufb path requires AES-NI + SSSE3"
    );
    assert_eq!(blocks.len(), out.len(), "out must be as long as blocks");
    let n = blocks.len();
    if n == 0 {
        return;
    }
    // One chunk per worker thread, rounded up to a multiple of W so each chunk
    // (bar the last) is a whole number of W-wide groups. `counter_block` advances
    // the counter by the chunk's start index — see the homomorphism note: the
    // per-chunk local index j maps to the same global counter as the serial path.
    let threads = rayon::current_num_threads().max(1);
    let chunk = n.div_ceil(threads).next_multiple_of(W);
    out.par_chunks_mut(chunk)
        .zip(blocks.par_chunks(chunk))
        .enumerate()
        .for_each(|(c, (out_c, blk_c))| {
            let c0 = counter_block(counter0, (c * chunk) as u32);
            // SAFETY: `aes` + `ssse3` are checked above and hold on every worker
            // (same CPU); the chunks from `par_chunks_mut` are disjoint.
            unsafe { encrypt_ctr_pshufb_impl::<W>(rk, c0, blk_c, out_c) };
        });
}

#[cfg(test)]
mod tests {
    use super::{encrypt_ctr, encrypt_ctr_pshufb, encrypt_ctr_pshufb_parallel};
    use aes_core::{key_expansion, CTR_KAT, KAT_VECTORS};

    /// The shared correctness gate, run at a given interleave width: the NIST
    /// F.5.1 multi-block vector (the mode itself) and the FIPS-197 cipher-core
    /// vectors via the `counter0 = PT, pt = 0` trick.
    fn check_kats<const W: usize>() {
        let rk = key_expansion(CTR_KAT.key);
        let mut out = [[0u32; 4]; 4];
        encrypt_ctr::<W>(&rk, CTR_KAT.counter0, &CTR_KAT.plaintext, &mut out);
        assert_eq!(out, CTR_KAT.ciphertext);
        for &(pt, key, expected) in KAT_VECTORS {
            let rk = key_expansion(key);
            let mut out = [[0u32; 4]];
            encrypt_ctr::<W>(&rk, pt, &[[0u32; 4]], &mut out);
            assert_eq!(out[0], expected);
        }
    }

    #[test]
    fn ctr_known_answers() {
        if !std::is_x86_feature_detected!("aes") {
            eprintln!("skipping ctr_known_answers: no AES-NI on this CPU");
            return;
        }
        check_kats::<1>(); // naive, one block at a time
        check_kats::<8>(); // interleaved
    }

    #[test]
    fn widths_agree() {
        if !std::is_x86_feature_detected!("aes") {
            eprintln!("skipping widths_agree: no AES-NI on this CPU");
            return;
        }
        // Every width must produce identical output, across sizes that exercise
        // full W-chunks *and* the (< W) remainder.
        let rk = key_expansion(CTR_KAT.key);
        let counter0 = CTR_KAT.counter0;
        for n in [0usize, 1, 7, 8, 9, 16, 17, 100] {
            let blocks: Vec<[u32; 4]> = (0..n)
                .map(|k| [k as u32, 0xDEAD_BEEF, 0x0BAD_F00D, 0x1234_5678])
                .collect();
            let mut a = vec![[0u32; 4]; n];
            let mut b = vec![[0u32; 4]; n];
            encrypt_ctr::<1>(&rk, counter0, &blocks, &mut a);
            encrypt_ctr::<8>(&rk, counter0, &blocks, &mut b);
            assert_eq!(a, b, "width 1 vs 8 disagree at n={n}");
        }
    }

    #[test]
    fn parallel_matches_serial() {
        if !std::is_x86_feature_detected!("aes") || !std::is_x86_feature_detected!("ssse3") {
            eprintln!("skipping parallel_matches_serial: no AES-NI/SSSE3 on this CPU");
            return;
        }
        // The multi-core path must be byte-identical to the single-core kernel.
        // Sizes span the empty case, sub-chunk, exact W multiples, the (< W)
        // remainder, and n large enough to split across several worker threads.
        let rk = key_expansion(CTR_KAT.key);
        let counter0 = CTR_KAT.counter0;
        for n in [0usize, 1, 7, 8, 9, 16, 17, 100, 1000, 65536] {
            let blocks: Vec<[u32; 4]> = (0..n)
                .map(|k| [k as u32, 0xDEAD_BEEF, 0x0BAD_F00D, 0x1234_5678])
                .collect();
            let mut serial = vec![[0u32; 4]; n];
            let mut par = vec![[0u32; 4]; n];
            encrypt_ctr::<8>(&rk, counter0, &blocks, &mut serial);
            encrypt_ctr_pshufb_parallel::<8>(&rk, counter0, &blocks, &mut par);
            assert_eq!(serial, par, "pshufb-parallel vs serial disagree at n={n}");
        }
    }

    #[test]
    fn pshufb_matches_serial() {
        if !std::is_x86_feature_detected!("aes") || !std::is_x86_feature_detected!("ssse3") {
            eprintln!("skipping pshufb_matches_serial: no AES-NI/SSSE3 on this CPU");
            return;
        }
        // The all-SIMD kernel must be byte-identical to the scalar-repack one, at
        // both the single-block width and the interleaved width, across sizes that
        // span the empty case, exact W multiples, and the (< W) remainder tail.
        let rk = key_expansion(CTR_KAT.key);
        let counter0 = CTR_KAT.counter0;
        for n in [0usize, 1, 7, 8, 9, 16, 17, 100, 1000] {
            let blocks: Vec<[u32; 4]> = (0..n)
                .map(|k| [k as u32, 0xDEAD_BEEF, 0x0BAD_F00D, 0x1234_5678])
                .collect();
            let mut reference = vec![[0u32; 4]; n];
            let mut w1 = vec![[0u32; 4]; n];
            let mut w8 = vec![[0u32; 4]; n];
            encrypt_ctr::<8>(&rk, counter0, &blocks, &mut reference);
            encrypt_ctr_pshufb::<1>(&rk, counter0, &blocks, &mut w1);
            encrypt_ctr_pshufb::<8>(&rk, counter0, &blocks, &mut w8);
            assert_eq!(reference, w1, "pshufb W=1 disagrees at n={n}");
            assert_eq!(reference, w8, "pshufb W=8 disagrees at n={n}");
        }
    }
}
