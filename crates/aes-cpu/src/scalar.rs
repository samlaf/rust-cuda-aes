//! Portable scalar AES-128 over a batch of blocks — the CPU baseline.
//!
//! Reuses the shared [`aes_core::encrypt_block`] (the table-based round
//! function) in a plain loop. This is the implementation every other backend
//! (VAES, GPU) is measured against.

use aes_core::{encrypt_block, SBOX_U32, T0, T1, T2, T3};

/// Encrypt each block independently (ECB): `out[i] = AES-128(rk, blocks[i])`.
///
/// `rk` is the 44-word expanded key from [`aes_core::key_expansion`]. `out` must
/// be at least as long as `blocks`.
pub fn encrypt_blocks(rk: &[u32], blocks: &[[u32; 4]], out: &mut [[u32; 4]]) {
    for (dst, &pt) in out.iter_mut().zip(blocks.iter()) {
        *dst = encrypt_block(pt, rk, &T0, &T1, &T2, &T3, &SBOX_U32);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aes_core::key_expansion;

    #[test]
    fn batch_matches_fips197() {
        // Each shared FIPS-197 vector, replicated across a batch, must come back
        // identical for every block.
        for &(pt, key, expected) in aes_core::KAT_VECTORS {
            let rk = key_expansion(key);
            let blocks = vec![pt; 64];
            let mut out = vec![[0u32; 4]; 64];
            encrypt_blocks(&rk, &blocks, &mut out);
            assert!(out.iter().all(|&ct| ct == expected));
        }
    }
}
