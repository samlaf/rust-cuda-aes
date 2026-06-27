//! AES-128 GPU kernels, ported from Tezcan's CUDA_AES.
//!
//! References:
//! - C. Tezcan, "Optimization of Advanced Encryption Standard on Graphics
//!   Processing Units", IEEE Access 2021 (eprint 2021/646).
//! - https://github.com/cihangirtezcan/CUDA_AES/blob/gh-pages/128-ctr.cuh
//!
//! Step 1 (this file): the *naive* table-based baseline the paper starts from.
//! A single thread encrypts one 16-byte block using the four T-tables
//! (T0..T3) for rounds 1..9 and the S-box for the last round. Everything lives
//! in plain global memory; the shared-memory / `__byte_perm` / bank-conflict
//! optimizations come in later steps.
//!
//! The constant tables live in [`tables`]; the round logic in [`encrypt_block`]
//! is shared between the GPU kernel and the host tests, so the test exercises
//! the exact same code the kernel runs.

#![allow(clippy::needless_range_loop)]

use cuda_std::prelude::*;

mod tables;
pub use tables::{RCON, SBOX, SBOX_U32, T0, T1, T2, T3};

// ---------------------------------------------------------------------------
// AES-128 key schedule (host-side; round keys are uploaded to the GPU).
// ---------------------------------------------------------------------------

/// Expand the 4 key words into 44 round-key words.
pub fn key_expansion(key: [u32; 4]) -> [u32; 44] {
    fn rot_word(w: u32) -> u32 {
        w.rotate_left(8)
    }
    fn sub_word(w: u32) -> u32 {
        u32::from_be_bytes(w.to_be_bytes().map(|b| SBOX[b as usize]))
    }

    let mut w = [0u32; 44];
    w[..4].copy_from_slice(&key);
    for i in 4..44 {
        let mut temp = w[i - 1];
        if i % 4 == 0 {
            temp = sub_word(rot_word(temp)) ^ ((RCON[i / 4 - 1] as u32) << 24);
        }
        w[i] = w[i - 4] ^ temp;
    }
    w
}

// ---------------------------------------------------------------------------
// Core single-block encryption, shared by the kernel and the tests.
// ---------------------------------------------------------------------------

/// Encrypt one AES-128 block.
///
/// - `pt`:   4 words of plaintext (big-endian within each word).
/// - `rk`:   44 expanded round-key words.
/// - `t0..t3`: the four T-tables (256 words each).
/// - `sbox`: the S-box (256 words) for the last round.
///
/// Returns the 4 ciphertext words.
#[inline]
pub fn encrypt_block(
    pt: [u32; 4],
    rk: &[u32],
    t0: &[u32],
    t1: &[u32],
    t2: &[u32],
    t3: &[u32],
    sbox: &[u32],
) -> [u32; 4] {
    // Initial AddRoundKey.
    let mut s0 = pt[0] ^ rk[0];
    let mut s1 = pt[1] ^ rk[1];
    let mut s2 = pt[2] ^ rk[2];
    let mut s3 = pt[3] ^ rk[3];

    // Rounds 1..=9: table-based SubBytes+ShiftRows+MixColumns+AddRoundKey.
    let mut round = 1usize;
    while round < 10 {
        let k = round * 4;
        let n0 = t0[(s0 >> 24) as usize]
            ^ t1[((s1 >> 16) & 0xff) as usize]
            ^ t2[((s2 >> 8) & 0xff) as usize]
            ^ t3[(s3 & 0xff) as usize]
            ^ rk[k];
        let n1 = t0[(s1 >> 24) as usize]
            ^ t1[((s2 >> 16) & 0xff) as usize]
            ^ t2[((s3 >> 8) & 0xff) as usize]
            ^ t3[(s0 & 0xff) as usize]
            ^ rk[k + 1];
        let n2 = t0[(s2 >> 24) as usize]
            ^ t1[((s3 >> 16) & 0xff) as usize]
            ^ t2[((s0 >> 8) & 0xff) as usize]
            ^ t3[(s1 & 0xff) as usize]
            ^ rk[k + 2];
        let n3 = t0[(s3 >> 24) as usize]
            ^ t1[((s0 >> 16) & 0xff) as usize]
            ^ t2[((s1 >> 8) & 0xff) as usize]
            ^ t3[(s2 & 0xff) as usize]
            ^ rk[k + 3];
        s0 = n0;
        s1 = n1;
        s2 = n2;
        s3 = n3;
        round += 1;
    }

    // Last round: SubBytes + ShiftRows + AddRoundKey (no MixColumns).
    let o0 = (sbox[(s0 >> 24) as usize] << 24)
        | (sbox[((s1 >> 16) & 0xff) as usize] << 16)
        | (sbox[((s2 >> 8) & 0xff) as usize] << 8)
        | sbox[(s3 & 0xff) as usize];
    let o1 = (sbox[(s1 >> 24) as usize] << 24)
        | (sbox[((s2 >> 16) & 0xff) as usize] << 16)
        | (sbox[((s3 >> 8) & 0xff) as usize] << 8)
        | sbox[(s0 & 0xff) as usize];
    let o2 = (sbox[(s2 >> 24) as usize] << 24)
        | (sbox[((s3 >> 16) & 0xff) as usize] << 16)
        | (sbox[((s0 >> 8) & 0xff) as usize] << 8)
        | sbox[(s1 & 0xff) as usize];
    let o3 = (sbox[(s3 >> 24) as usize] << 24)
        | (sbox[((s0 >> 16) & 0xff) as usize] << 16)
        | (sbox[((s1 >> 8) & 0xff) as usize] << 8)
        | sbox[(s2 & 0xff) as usize];

    [o0 ^ rk[40], o1 ^ rk[41], o2 ^ rk[42], o3 ^ rk[43]]
}

// ---------------------------------------------------------------------------
// Kernel: encrypt a single 16-byte block with one thread.
// ---------------------------------------------------------------------------

/// Single-block AES-128 encryption. See [`encrypt_block`] for the parameters.
#[kernel]
#[allow(improper_ctypes_definitions)]
pub unsafe fn aes128_encrypt_block(
    pt: &[u32],
    rk: &[u32],
    t0: &[u32],
    t1: &[u32],
    t2: &[u32],
    t3: &[u32],
    sbox: &[u32],
    ct: *mut u32,
) {
    // Only one thread does the work for this single-block baseline.
    if thread::index_1d() != 0 {
        return;
    }

    let out = encrypt_block([pt[0], pt[1], pt[2], pt[3]], rk, t0, t1, t2, t3, sbox);

    unsafe {
        *ct.add(0) = out[0];
        *ct.add(1) = out[1];
        *ct.add(2) = out[2];
        *ct.add(3) = out[3];
    }
}

#[cfg(test)]
mod tests {
    // Import items explicitly rather than `use super::*` so we don't pull in
    // cuda_std's device-side `assert_eq!` macro, which conflicts with std's.
    use super::{encrypt_block, key_expansion, SBOX_U32, T0, T1, T2, T3};

    /// Run the shared `encrypt_block` against a FIPS-197 known-answer test.
    fn kat(pt: [u32; 4], key: [u32; 4]) -> [u32; 4] {
        let rk = key_expansion(key);
        encrypt_block(pt, &rk, &T0, &T1, &T2, &T3, &SBOX_U32)
    }

    #[test]
    fn key_expansion_intermediate_values() {
        // FIPS-197 Appendix A.1 expanded-key words for this key.
        let rk = key_expansion([0x2B7E1516, 0x28AED2A6, 0xABF71588, 0x09CF4F3C]);
        assert_eq!(rk[4], 0xa0fafe17); // first generated word
        assert_eq!(rk[43], 0xb6630ca6); // last word
    }

    #[test]
    fn fips197_appendix_b() {
        let ct = kat(
            [0x3243F6A8, 0x885A308D, 0x313198A2, 0xE0370734],
            [0x2B7E1516, 0x28AED2A6, 0xABF71588, 0x09CF4F3C],
        );
        assert_eq!(ct, [0x3925841D, 0x02DC09FB, 0xDC118597, 0x196A0B32]);
    }

    #[test]
    fn fips197_appendix_c1() {
        let ct = kat(
            [0x00112233, 0x44556677, 0x8899AABB, 0xCCDDEEFF],
            [0x00010203, 0x04050607, 0x08090A0B, 0x0C0D0E0F],
        );
        assert_eq!(ct, [0x69C4E0D8, 0x6A7B0430, 0xD8CDB780, 0x70B4C55A]);
    }
}
