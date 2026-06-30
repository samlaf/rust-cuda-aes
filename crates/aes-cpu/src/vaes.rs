//! AES-128-CTR via **VAES** — wider hardware AES on x86_64.
//!
//! VAES is the vectorized form of the AES-NI round instructions: a single
//! `VAESENC` applies one AES round to *every* 128-bit lane of a wide register at
//! once. So where [`crate::aesni`] does one block per `_mm_aesenc_si128`, this
//! module does:
//!
//! - **2 blocks per instruction** with `_mm256_aesenc_epi128` (`__m256i`, needs
//!   `vaes` + `avx2`), and
//! - **4 blocks per instruction** with `_mm512_aesenc_epi128` (`__m512i`, needs
//!   `vaes` + `avx512f`/`avx512bw`).
//!
//! Both kernels are the [`crate::aesni::encrypt_ctr_pshufb`] design lifted into
//! wide registers: the per-block dataflow stays entirely in the SIMD domain (no
//! scalar repack), counters are bumped in-register with a per-lane
//! `_mm_add_epi32`-style step, and the "big-endian within each word" ↔ AES-state
//! byte swap is one `PSHUFB` (here `_mm256/512_shuffle_epi8`, which shuffle
//! *within each 128-bit lane*, so the same 16-byte mask broadcast to every lane
//! reverses each block's words independently).
//!
//! Each kernel is generic over an **interleave width** `W` = the number of *wide
//! registers* kept in flight per iteration, so it processes `W * LANES` blocks at
//! a time (`LANES` = 2 for the 256-bit path, 4 for the 512-bit path) — the same
//! latency-hiding trick as `aesni`, one level up: VAES widens the *instruction*,
//! `W` keeps enough independent VAES ops in flight to saturate the units. Round
//! keys are broadcast into every lane once and held in an array so each `VAESENC`
//! folds the broadcast key as a memory operand (no per-round broadcast, no
//! register pressure from 11 live key registers).
//!
//! Byte-identical to [`crate::aesni::encrypt_ctr`] for every `W` — same shared KAT
//! gate, and the round keys / scalar-tail / per-chunk counters all go through the
//! exact `aesni` helpers ([`round_keys`], [`encrypt_one`], [`counter_block`]), so
//! there is a single byte convention across the AES-NI and VAES paths.
//!
//! **Hardware:** VAES needs Intel Ice Lake+ / Tiger Lake+ or AMD Zen 4+ (the
//! 256-bit path runs on AMD Zen 3 too; the 512-bit path needs AVX-512, so Zen 4+
//! / Ice Lake+). The dev box's Zen 2 EPYC has neither, so these variants register
//! and run only where `is_x86_feature_detected!` reports the features — they're
//! silently absent on Zen 2 (same pattern as the `aes` guard in `aesni`).

use core::arch::x86_64::*;
use rayon::prelude::*;

use crate::aesni::{counter_block, encrypt_one, round_keys};

/// The byte-reverse-within-each-32-bit-word mask (native little-endian word order
/// ↔ AES big-endian-within-word state order). Identical to the `aesni` pshufb
/// kernel's mask; broadcast into every 128-bit lane below. Self-inverse.
#[inline]
#[target_feature(enable = "sse2")]
unsafe fn bswap_mask() -> __m128i {
    _mm_setr_epi8(3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12)
}

/// The block-0 counter as one `__m128i` in native lane order (element `k` =
/// `counter0[k]`), the seed every wide counter vector is built from.
#[inline]
#[target_feature(enable = "sse2")]
unsafe fn counter_seed(counter0: [u32; 4]) -> __m128i {
    _mm_set_epi32(
        counter0[3] as i32,
        counter0[2] as i32,
        counter0[1] as i32,
        counter0[0] as i32,
    )
}

// ---------------------------------------------------------------------------
// 256-bit VAES — 2 blocks per instruction.
// ---------------------------------------------------------------------------

/// Encrypt `blocks` in CTR mode with **256-bit VAES** (2 blocks per `VAESENC`),
/// interleaving `W` `__m256i` registers (`2 * W` blocks per iteration).
///
/// Same result as [`crate::aesni::encrypt_ctr`] for every `W`. `rk` is the
/// 44-word expanded key from [`aes_core::key_expansion`]; `out.len()` must equal
/// `blocks.len()`.
///
/// Panics if the CPU lacks VAES + AVX2 — register the variant behind a runtime
/// `is_x86_feature_detected!` check (the bench registry does) so it's simply
/// absent on CPUs without it.
pub fn encrypt_ctr_vaes256<const W: usize>(
    rk: &[u32],
    counter0: [u32; 4],
    blocks: &[[u32; 4]],
    out: &mut [[u32; 4]],
) {
    assert!(
        std::is_x86_feature_detected!("aes")
            && std::is_x86_feature_detected!("vaes")
            && std::is_x86_feature_detected!("avx2"),
        "aes-cpu::vaes 256-bit path requires VAES + AVX2"
    );
    assert_eq!(blocks.len(), out.len(), "out must be as long as blocks");
    // SAFETY: the runtime checks above uphold the intrinsics' feature contract.
    unsafe { encrypt_ctr_vaes256_impl::<W>(rk, counter0, blocks, out) }
}

#[target_feature(enable = "aes,vaes,avx2")]
unsafe fn encrypt_ctr_vaes256_impl<const W: usize>(
    rk: &[u32],
    counter0: [u32; 4],
    blocks: &[[u32; 4]],
    out: &mut [[u32; 4]],
) {
    const LANES: usize = 2;
    unsafe {
        let keys = round_keys(rk);
        // Broadcast each round key into both 128-bit lanes. Held in an array so
        // every VAESENC folds the key as a 256-bit memory operand.
        let mut keys2 = [_mm256_setzero_si256(); 11];
        for (k2, &k) in keys2.iter_mut().zip(keys.iter()) {
            *k2 = _mm256_broadcastsi128_si256(k);
        }
        let bswap = _mm256_broadcastsi128_si256(bswap_mask());
        let bc = _mm256_broadcastsi128_si256(counter_seed(counter0));

        // ctr[j] holds the (native-order) counters for blocks (2j, 2j+1); the low
        // lane is the earlier block, matching the contiguous `__m256i` load below.
        // Per-lane add (no carry) on the low CTR word == the project's wrapping_add.
        let mut ctr = [_mm256_setzero_si256(); W];
        for (j, c) in ctr.iter_mut().enumerate() {
            let off = _mm256_set_epi32((2 * j + 1) as i32, 0, 0, 0, (2 * j) as i32, 0, 0, 0);
            *c = _mm256_add_epi32(bc, off);
        }
        let step = _mm256_set_epi32((2 * W) as i32, 0, 0, 0, (2 * W) as i32, 0, 0, 0);

        let n = blocks.len();
        let mut i = 0;
        while i + W * LANES <= n {
            let mut s = [_mm256_setzero_si256(); W];
            for (sj, &cj) in s.iter_mut().zip(ctr.iter()) {
                *sj = _mm256_xor_si256(_mm256_shuffle_epi8(cj, bswap), keys2[0]);
            }
            for r in 1..10 {
                for sj in s.iter_mut() {
                    *sj = _mm256_aesenc_epi128(*sj, keys2[r]);
                }
            }
            for sj in s.iter_mut() {
                *sj = _mm256_aesenclast_epi128(*sj, keys2[10]);
            }
            for (j, sj) in s.iter().enumerate() {
                let pt = _mm256_loadu_si256(blocks.as_ptr().add(i + j * LANES) as *const __m256i);
                let ks = _mm256_shuffle_epi8(*sj, bswap);
                _mm256_storeu_si256(
                    out.as_mut_ptr().add(i + j * LANES) as *mut __m256i,
                    _mm256_xor_si256(pt, ks),
                );
            }
            for cj in ctr.iter_mut() {
                *cj = _mm256_add_epi32(*cj, step);
            }
            i += W * LANES;
        }
        // Remainder (< W*LANES blocks): the cold scalar-repack path is fine here.
        while i < n {
            let ks = encrypt_one(&keys, counter_block(counter0, i as u32));
            let pt = blocks[i];
            out[i] = [pt[0] ^ ks[0], pt[1] ^ ks[1], pt[2] ^ ks[2], pt[3] ^ ks[3]];
            i += 1;
        }
    }
}

// ---------------------------------------------------------------------------
// 512-bit VAES — 4 blocks per instruction.
// ---------------------------------------------------------------------------

/// Encrypt `blocks` in CTR mode with **512-bit VAES** (4 blocks per `VAESENC`),
/// interleaving `W` `__m512i` registers (`4 * W` blocks per iteration). On a true
/// 512-bit datapath (AMD Zen 5, Intel Ice Lake+/Sapphire Rapids) this doubles the
/// 256-bit path again; on Zen 4's double-pumped AVX-512 it does not.
///
/// Same result as [`crate::aesni::encrypt_ctr`] for every `W`. Panics if the CPU
/// lacks VAES + AVX-512 (`avx512f` + `avx512bw`).
pub fn encrypt_ctr_vaes512<const W: usize>(
    rk: &[u32],
    counter0: [u32; 4],
    blocks: &[[u32; 4]],
    out: &mut [[u32; 4]],
) {
    assert!(
        std::is_x86_feature_detected!("aes")
            && std::is_x86_feature_detected!("vaes")
            && std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512bw"),
        "aes-cpu::vaes 512-bit path requires VAES + AVX-512 (avx512f + avx512bw)"
    );
    assert_eq!(blocks.len(), out.len(), "out must be as long as blocks");
    // SAFETY: the runtime checks above uphold the intrinsics' feature contract.
    unsafe { encrypt_ctr_vaes512_impl::<W>(rk, counter0, blocks, out) }
}

#[target_feature(enable = "aes,vaes,avx512f,avx512bw")]
unsafe fn encrypt_ctr_vaes512_impl<const W: usize>(
    rk: &[u32],
    counter0: [u32; 4],
    blocks: &[[u32; 4]],
    out: &mut [[u32; 4]],
) {
    const LANES: usize = 4;
    unsafe {
        let keys = round_keys(rk);
        // Broadcast each round key into all four 128-bit lanes (memory operand).
        let mut keys4 = [_mm512_setzero_si512(); 11];
        for (k4, &k) in keys4.iter_mut().zip(keys.iter()) {
            *k4 = _mm512_broadcast_i32x4(k);
        }
        let bswap = _mm512_broadcast_i32x4(bswap_mask());
        let bc = _mm512_broadcast_i32x4(counter_seed(counter0));

        // ctr[j] holds the counters for blocks (4j .. 4j+3), lane k = block 4j+k,
        // matching the contiguous `__m512i` load. The CTR low word is element 3 of
        // each 128-bit lane → global elements 3, 7, 11, 15.
        let mut ctr = [_mm512_setzero_si512(); W];
        for (j, c) in ctr.iter_mut().enumerate() {
            let off = _mm512_set_epi32(
                (4 * j + 3) as i32,
                0,
                0,
                0,
                (4 * j + 2) as i32,
                0,
                0,
                0,
                (4 * j + 1) as i32,
                0,
                0,
                0,
                (4 * j) as i32,
                0,
                0,
                0,
            );
            *c = _mm512_add_epi32(bc, off);
        }
        let w4 = (4 * W) as i32;
        let step = _mm512_set_epi32(w4, 0, 0, 0, w4, 0, 0, 0, w4, 0, 0, 0, w4, 0, 0, 0);

        let n = blocks.len();
        let mut i = 0;
        while i + W * LANES <= n {
            let mut s = [_mm512_setzero_si512(); W];
            for (sj, &cj) in s.iter_mut().zip(ctr.iter()) {
                *sj = _mm512_xor_si512(_mm512_shuffle_epi8(cj, bswap), keys4[0]);
            }
            for r in 1..10 {
                for sj in s.iter_mut() {
                    *sj = _mm512_aesenc_epi128(*sj, keys4[r]);
                }
            }
            for sj in s.iter_mut() {
                *sj = _mm512_aesenclast_epi128(*sj, keys4[10]);
            }
            for (j, sj) in s.iter().enumerate() {
                let pt = _mm512_loadu_si512(blocks.as_ptr().add(i + j * LANES) as *const __m512i);
                let ks = _mm512_shuffle_epi8(*sj, bswap);
                _mm512_storeu_si512(
                    out.as_mut_ptr().add(i + j * LANES) as *mut __m512i,
                    _mm512_xor_si512(pt, ks),
                );
            }
            for cj in ctr.iter_mut() {
                *cj = _mm512_add_epi32(*cj, step);
            }
            i += W * LANES;
        }
        // Remainder (< W*LANES blocks): scalar-repack tail.
        while i < n {
            let ks = encrypt_one(&keys, counter_block(counter0, i as u32));
            let pt = blocks[i];
            out[i] = [pt[0] ^ ks[0], pt[1] ^ ks[1], pt[2] ^ ks[2], pt[3] ^ ks[3]];
            i += 1;
        }
    }
}

// ---------------------------------------------------------------------------
// Multi-core wrappers — VAES kernel × N cores, same rayon split as `aesni`.
// ---------------------------------------------------------------------------

/// One chunk per worker thread, rounded up to a whole number of `group`-block
/// interleave groups so each chunk (bar the last) is full-width and only the last
/// can carry a `< group` remainder. `group = W * LANES`.
#[inline]
fn par_chunk(n: usize, group: usize) -> usize {
    n.div_ceil(rayon::current_num_threads().max(1))
        .next_multiple_of(group)
}

/// [`encrypt_ctr_vaes256`] fanned across CPU cores with rayon. Each chunk is an
/// independent CTR run whose counter is pre-advanced to its first block via
/// [`counter_block`], so the result is byte-identical to the serial kernel.
pub fn encrypt_ctr_vaes256_parallel<const W: usize>(
    rk: &[u32],
    counter0: [u32; 4],
    blocks: &[[u32; 4]],
    out: &mut [[u32; 4]],
) {
    assert!(
        std::is_x86_feature_detected!("aes")
            && std::is_x86_feature_detected!("vaes")
            && std::is_x86_feature_detected!("avx2"),
        "aes-cpu::vaes 256-bit path requires VAES + AVX2"
    );
    assert_eq!(blocks.len(), out.len(), "out must be as long as blocks");
    let n = blocks.len();
    if n == 0 {
        return;
    }
    let chunk = par_chunk(n, W * 2);
    out.par_chunks_mut(chunk)
        .zip(blocks.par_chunks(chunk))
        .enumerate()
        .for_each(|(c, (out_c, blk_c))| {
            let c0 = counter_block(counter0, (c * chunk) as u32);
            // SAFETY: features checked above hold on every worker (same CPU); the
            // chunks from par_chunks_mut are disjoint.
            unsafe { encrypt_ctr_vaes256_impl::<W>(rk, c0, blk_c, out_c) };
        });
}

/// [`encrypt_ctr_vaes512`] fanned across CPU cores with rayon (see
/// [`encrypt_ctr_vaes256_parallel`]).
pub fn encrypt_ctr_vaes512_parallel<const W: usize>(
    rk: &[u32],
    counter0: [u32; 4],
    blocks: &[[u32; 4]],
    out: &mut [[u32; 4]],
) {
    assert!(
        std::is_x86_feature_detected!("aes")
            && std::is_x86_feature_detected!("vaes")
            && std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512bw"),
        "aes-cpu::vaes 512-bit path requires VAES + AVX-512 (avx512f + avx512bw)"
    );
    assert_eq!(blocks.len(), out.len(), "out must be as long as blocks");
    let n = blocks.len();
    if n == 0 {
        return;
    }
    let chunk = par_chunk(n, W * 4);
    out.par_chunks_mut(chunk)
        .zip(blocks.par_chunks(chunk))
        .enumerate()
        .for_each(|(c, (out_c, blk_c))| {
            let c0 = counter_block(counter0, (c * chunk) as u32);
            // SAFETY: see encrypt_ctr_vaes256_parallel.
            unsafe { encrypt_ctr_vaes512_impl::<W>(rk, c0, blk_c, out_c) };
        });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aesni::encrypt_ctr;
    use aes_core::{key_expansion, CTR_KAT, KAT_VECTORS};

    fn have_vaes256() -> bool {
        std::is_x86_feature_detected!("vaes") && std::is_x86_feature_detected!("avx2")
    }
    fn have_vaes512() -> bool {
        std::is_x86_feature_detected!("vaes")
            && std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512bw")
    }

    /// The shared KAT gate (FIPS-197 cipher core via `counter0 = PT, pt = 0`, plus
    /// NIST F.5.1 multi-block CTR), run through one of the VAES kernels.
    fn check_kats(enc: impl Fn(&[u32], [u32; 4], &[[u32; 4]], &mut [[u32; 4]])) {
        let rk = key_expansion(CTR_KAT.key);
        let mut out = [[0u32; 4]; 4];
        enc(&rk, CTR_KAT.counter0, &CTR_KAT.plaintext, &mut out);
        assert_eq!(out, CTR_KAT.ciphertext);
        for &(pt, key, expected) in KAT_VECTORS {
            let rk = key_expansion(key);
            let mut out = [[0u32; 4]];
            enc(&rk, pt, &[[0u32; 4]], &mut out);
            assert_eq!(out[0], expected);
        }
    }

    /// Every VAES kernel must be byte-identical to the AES-NI reference across
    /// sizes that exercise full W*LANES groups *and* the (< group) remainder tail.
    fn agrees_with_aesni(enc: impl Fn(&[u32], [u32; 4], &[[u32; 4]], &mut [[u32; 4]])) {
        let rk = key_expansion(CTR_KAT.key);
        let counter0 = CTR_KAT.counter0;
        for n in [0usize, 1, 3, 4, 7, 8, 9, 16, 17, 32, 33, 100, 1000, 65536] {
            let blocks: Vec<[u32; 4]> = (0..n)
                .map(|k| [k as u32, 0xDEAD_BEEF, 0x0BAD_F00D, 0x1234_5678])
                .collect();
            let mut reference = vec![[0u32; 4]; n];
            let mut got = vec![[0u32; 4]; n];
            encrypt_ctr::<8>(&rk, counter0, &blocks, &mut reference);
            enc(&rk, counter0, &blocks, &mut got);
            assert_eq!(reference, got, "VAES disagrees with AES-NI at n={n}");
        }
    }

    #[test]
    fn vaes256_known_answers() {
        if !have_vaes256() {
            eprintln!("skipping vaes256_known_answers: no VAES/AVX2 on this CPU");
            return;
        }
        check_kats(encrypt_ctr_vaes256::<1>);
        check_kats(encrypt_ctr_vaes256::<8>);
        agrees_with_aesni(encrypt_ctr_vaes256::<1>);
        agrees_with_aesni(encrypt_ctr_vaes256::<8>);
        agrees_with_aesni(encrypt_ctr_vaes256_parallel::<8>);
    }

    #[test]
    fn vaes512_known_answers() {
        if !have_vaes512() {
            eprintln!("skipping vaes512_known_answers: no VAES/AVX-512 on this CPU");
            return;
        }
        check_kats(encrypt_ctr_vaes512::<1>);
        check_kats(encrypt_ctr_vaes512::<8>);
        agrees_with_aesni(encrypt_ctr_vaes512::<1>);
        agrees_with_aesni(encrypt_ctr_vaes512::<8>);
        agrees_with_aesni(encrypt_ctr_vaes512_parallel::<8>);
    }
}
