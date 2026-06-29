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
//! [`encrypt_ctr_parallel`] is the multi-core wrapper: CTR is embarrassingly
//! parallel (block `i` only needs `counter0 + i`), so it splits the buffer into
//! chunks, hands each chunk a pre-advanced counter, and runs the same `W`-wide
//! kernel on every core via rayon. Per-core latency hiding (`W`) and cross-core
//! scaling (rayon) compose — the parallel variant is the x8 kernel × N cores.
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
#[inline]
#[target_feature(enable = "aes")]
unsafe fn round_keys(rk: &[u32]) -> [__m128i; 11] {
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
/// `< W` remainder of [`encrypt_ctr`].
#[inline]
#[target_feature(enable = "aes")]
unsafe fn encrypt_one(rk: &[__m128i; 11], block: [u32; 4]) -> [u32; 4] {
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
/// [`aes_core::encrypt_ctr_block`]).
#[inline]
fn counter_block(counter0: [u32; 4], idx: u32) -> [u32; 4] {
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

/// Like [`encrypt_ctr`] but fans the `W`-interleaved work across CPU cores with
/// rayon. CTR is embarrassingly parallel: block `i` always uses `counter0 + i`,
/// so the chunk starting at block `start` is just an independent [`encrypt_ctr`]
/// whose counter is pre-advanced to `counter_block(counter0, start)`. Each core
/// then runs the exact same latency-hiding `W`-wide kernel, so the result is
/// byte-identical to `encrypt_ctr::<W>` — only faster by ~the core count.
///
/// Chunks are sized to one per worker thread, rounded up to a whole number of
/// `W`-wide groups so the main interleaved loop stays full-width in every chunk
/// and only the final chunk can carry a `< W` remainder.
///
/// Panics if the CPU lacks AES-NI, or if `out.len() != blocks.len()`.
pub fn encrypt_ctr_parallel<const W: usize>(
    rk: &[u32],
    counter0: [u32; 4],
    blocks: &[[u32; 4]],
    out: &mut [[u32; 4]],
) {
    assert!(
        std::is_x86_feature_detected!("aes"),
        "aes-cpu::aesni requires AES-NI (the `aes` CPU feature)"
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
            // SAFETY: the `aes` feature is checked above and holds on every
            // worker (same CPU); the chunks from `par_chunks_mut` are disjoint.
            unsafe { encrypt_ctr_impl::<W>(rk, c0, blk_c, out_c) };
        });
}

#[cfg(test)]
mod tests {
    use super::{encrypt_ctr, encrypt_ctr_parallel};
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
        if !std::is_x86_feature_detected!("aes") {
            eprintln!("skipping parallel_matches_serial: no AES-NI on this CPU");
            return;
        }
        // The multi-core path must be byte-identical to the single-core x8 kernel.
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
            encrypt_ctr_parallel::<8>(&rk, counter0, &blocks, &mut par);
            assert_eq!(serial, par, "parallel vs serial x8 disagree at n={n}");
        }
    }
}
