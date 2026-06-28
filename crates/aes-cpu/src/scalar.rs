//! Portable scalar AES-128-CTR over a batch of blocks — the CPU baseline.
//!
//! Reuses the shared [`aes_core::encrypt_ctr_block`] (cipher applied to the
//! counter, keystream XORed into the plaintext) in a plain loop. This is the
//! implementation every other backend (VAES, GPU) is measured against.

use aes_core::{encrypt_ctr_block, SBOX_U32, T0, T1, T2, T3};

/// Encrypt `blocks` in CTR mode: `out[i] = blocks[i] ⊕ E(rk, counter0 + i)`.
///
/// `rk` is the 44-word expanded key from [`aes_core::key_expansion`]; `counter0`
/// is the initial counter block, incremented per block in its low 32-bit word
/// (see [`aes_core::encrypt_ctr_block`]). `out` must be at least as long as
/// `blocks`.
pub fn encrypt_ctr(rk: &[u32], counter0: [u32; 4], blocks: &[[u32; 4]], out: &mut [[u32; 4]]) {
    for (i, (dst, &pt)) in out.iter_mut().zip(blocks.iter()).enumerate() {
        *dst = encrypt_ctr_block(counter0, i as u32, pt, rk, &T0, &T1, &T2, &T3, &SBOX_U32);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aes_core::{key_expansion, CTR_KAT, KAT_VECTORS};

    #[test]
    fn ctr_matches_nist_f51() {
        // The full multi-block CTR vector: counter increment + keystream XOR.
        let rk = key_expansion(CTR_KAT.key);
        let mut out = [[0u32; 4]; 4];
        encrypt_ctr(&rk, CTR_KAT.counter0, &CTR_KAT.plaintext, &mut out);
        assert_eq!(out, CTR_KAT.ciphertext);
    }

    #[test]
    fn ctr_recovers_cipher_core() {
        // With counter0 = PT and zero plaintext, the keystream *is* the cipher
        // output: out[0] = 0 ⊕ E(k, PT) = E(k, PT). This runs the FIPS-197
        // cipher-core vectors through the CTR path.
        for &(pt, key, expected) in KAT_VECTORS {
            let rk = key_expansion(key);
            let mut out = [[0u32; 4]];
            encrypt_ctr(&rk, pt, &[[0u32; 4]], &mut out);
            assert_eq!(out[0], expected);
        }
    }
}
