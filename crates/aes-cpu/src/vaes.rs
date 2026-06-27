//! AES-NI / VAES hardware-accelerated AES-128 (x86_64 only).
//!
//! TODO: implement and benchmark on the x86_64 `gpu` host (cannot be compiled
//! on arm64 macOS, hence the `#[cfg(target_arch = "x86_64")]` gate in `lib.rs`).
//!
//! Planned approach:
//!   1. Single-block AES-NI: load round keys as `__m128i`, then
//!      `_mm_xor_si128` + 9×`_mm_aesenc_si128` + `_mm_aesenclast_si128`.
//!   2. Widen to VAES for throughput: process 2/4 blocks per instruction with
//!      `_mm256_aesenc_epi128` / `_mm512_aesenc_epi128` over CTR-mode counters.
//!
//! Correctness gate: match the byte order of [`aes_core::encrypt_block`] and
//! verify against the FIPS-197 Appendix B known answer before benchmarking.
//!
//! Expose a `pub fn encrypt_blocks(rk: &[u32], blocks: &[[u32; 4]], out: &mut [[u32; 4]])`
//! mirroring [`crate::scalar::encrypt_blocks`] so the bench harness can swap
//! backends behind one signature.
