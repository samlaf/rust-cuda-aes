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

/// A multi-block CTR-mode known-answer vector: the cipher-core [`KAT_VECTORS`]
/// only exercise `encrypt_block`, so they say nothing about the *mode* (counter
/// increment + keystream XOR). This pins the mode itself.
pub struct CtrKat {
    /// 4 key words (big-endian within each word).
    pub key: [u32; 4],
    /// Initial counter block; block `i` encrypts `counter0` with its low 32-bit
    /// word increased by `i` (see [`encrypt_ctr_block`]).
    pub counter0: [u32; 4],
    /// Plaintext blocks, in order.
    pub plaintext: [[u32; 4]; 4],
    /// Expected ciphertext blocks, in order.
    pub ciphertext: [[u32; 4]; 4],
}

/// NIST SP 800-38A Appendix F.5.1 (CTR-AES128.Encrypt). The single source of
/// truth for CTR-mode correctness, shared by every backend's tests. The key is
/// the same FIPS-197 Appendix B key used in [`KAT_VECTORS`]; the four-block
/// counter sequence `…fcfdfeff → …fcfdff00 → …ff01 → …ff02` is what fixes the
/// low-word increment convention.
pub const CTR_KAT: CtrKat = CtrKat {
    key: [0x2B7E1516, 0x28AED2A6, 0xABF71588, 0x09CF4F3C],
    counter0: [0xF0F1F2F3, 0xF4F5F6F7, 0xF8F9FAFB, 0xFCFDFEFF],
    plaintext: [
        [0x6BC1BEE2, 0x2E409F96, 0xE93D7E11, 0x7393172A],
        [0xAE2D8A57, 0x1E03AC9C, 0x9EB76FAC, 0x45AF8E51],
        [0x30C81C46, 0xA35CE411, 0xE5FBC119, 0x1A0A52EF],
        [0xF69F2445, 0xDF4F9B17, 0xAD2B417B, 0xE66C3710],
    ],
    ciphertext: [
        [0x874D6191, 0xB620E326, 0x1BEF6864, 0x990DB6CE],
        [0x9806F66B, 0x7970FDFF, 0x8617187B, 0xB9FFFDFF],
        [0x5AE4DF3E, 0xDBD5D35E, 0x5B4F0902, 0x0DB03EAB],
        [0x1E031DDA, 0x2FBE03D1, 0x792170A0, 0xF3009CEE],
    ],
};

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

// ---------------------------------------------------------------------------
// CTR mode: apply the cipher to a counter, XOR the keystream into the plaintext.
// ---------------------------------------------------------------------------

/// Encrypt one CTR-mode block: `ct = pt ⊕ E(k, counter)`.
///
/// The counter for block `index` is `counter0` with its **low 32-bit word**
/// increased by `index` (`counter0[3].wrapping_add(index)`), matching NIST SP
/// 800-38A F.5: the increment stays in the low word and never carries (good for
/// up to 2^32 blocks per nonce). The cipher is applied to the counter, not to
/// `pt`, so this is the exact building block both the GPU kernel and the CPU
/// backend loop over — keeping the keystream identical across backends.
///
/// Parameters mirror [`encrypt_block`]; `index` is the block's position in the
/// counter sequence.
#[inline]
#[allow(clippy::too_many_arguments)] // tables are passed explicitly (no globals on the GPU)
pub fn encrypt_ctr_block(
    counter0: [u32; 4],
    index: u32,
    pt: [u32; 4],
    rk: &[u32],
    t0: &[u32],
    t1: &[u32],
    t2: &[u32],
    t3: &[u32],
    sbox: &[u32],
) -> [u32; 4] {
    let counter = [
        counter0[0],
        counter0[1],
        counter0[2],
        counter0[3].wrapping_add(index),
    ];
    let ks = encrypt_block(counter, rk, t0, t1, t2, t3, sbox);
    [pt[0] ^ ks[0], pt[1] ^ ks[1], pt[2] ^ ks[2], pt[3] ^ ks[3]]
}

#[cfg(test)]
mod tests {
    use super::{
        encrypt_block, encrypt_ctr_block, key_expansion, CTR_KAT, SBOX_U32, T0, T1, T2, T3,
    };

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

    #[test]
    fn nist_f51_ctr() {
        // The CTR-mode vector (NIST SP 800-38A F.5.1) exercises the counter
        // increment and the keystream XOR, which the cipher-core KATs don't.
        let rk = key_expansion(CTR_KAT.key);
        for (i, (&pt, &expected)) in CTR_KAT
            .plaintext
            .iter()
            .zip(CTR_KAT.ciphertext.iter())
            .enumerate()
        {
            let ct = encrypt_ctr_block(
                CTR_KAT.counter0,
                i as u32,
                pt,
                &rk,
                &T0,
                &T1,
                &T2,
                &T3,
                &SBOX_U32,
            );
            assert_eq!(ct, expected, "CTR block {i} mismatch");
        }
    }
}
