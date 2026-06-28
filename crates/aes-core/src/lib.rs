//! Portable AES-128 primitives, shared by the GPU kernels and the CPU code.
//!
//! References:
//! - C. Tezcan, "Optimization of Advanced Encryption Standard on Graphics
//!   Processing Units", IEEE Access 2021 (eprint 2021/646).
//! - https://github.com/cihangirtezcan/CUDA_AES/blob/gh-pages/128-ctr.cuh
//!
//! This crate holds the *naive* table-based baseline the paper starts from:
//! [`encrypt_block`] encrypts one 16-byte block using the four T-tables
//! (`T0..T3`) for rounds 1..9 and the S-box for the last round. The same
//! function is called by the GPU kernel (see the `kernels` crate), by the
//! scalar CPU backend (`aes-cpu`), and by the tests below, so the known-answer
//! tests exercise the exact code the kernel runs.
//!
//! The constant tables live in [`tables`]; they are generated at compile time
//! from GF(2^8) first principles (no hand-transcribed magic numbers).
//!
//! `no_std` (except under `cargo test`) so the crate can be compiled for the
//! `nvptx64` device target as a dependency of `kernels`.

#![cfg_attr(not(test), no_std)]
#![allow(clippy::needless_range_loop)]

mod tables;
pub use tables::{RCON, SBOX, SBOX_U32, T0, T1, T2, T3};

/// FIPS-197 known-answer vectors: `(plaintext, key, ciphertext)`, as big-endian
/// words. Single source of truth shared by every backend's tests (CPU, GPU) and
/// the benchmark variant registry, so they all check the exact same answers.
pub const KAT_VECTORS: &[([u32; 4], [u32; 4], [u32; 4])] = &[
    // Appendix B.
    (
        [0x3243F6A8, 0x885A308D, 0x313198A2, 0xE0370734],
        [0x2B7E1516, 0x28AED2A6, 0xABF71588, 0x09CF4F3C],
        [0x3925841D, 0x02DC09FB, 0xDC118597, 0x196A0B32],
    ),
    // Appendix C.1.
    (
        [0x00112233, 0x44556677, 0x8899AABB, 0xCCDDEEFF],
        [0x00010203, 0x04050607, 0x08090A0B, 0x0C0D0E0F],
        [0x69C4E0D8, 0x6A7B0430, 0xD8CDB780, 0x70B4C55A],
    ),
];

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
// Core single-block encryption, shared by the kernel, the CPU code, and tests.
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

#[cfg(test)]
mod tests {
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
    fn fips197_known_answers() {
        // Every vector in the shared FIPS-197 set round-trips through the same
        // table-based round function the kernel and CPU backends use.
        for &(pt, key, expected) in crate::KAT_VECTORS {
            assert_eq!(kat(pt, key), expected);
        }
    }
}
