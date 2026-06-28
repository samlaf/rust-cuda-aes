//! Shared helpers for the AES benchmarks, so every backend (CPU scalar, VAES,
//! and — eventually — the GPU kernels) is measured against the exact same
//! workload.

mod variants;
pub use variants::*;

use aes_core::key_expansion;

/// Default workload size, in 16-byte blocks (`1 << 16` = 64Ki blocks = 1 MiB).
pub const N_BLOCKS: usize = 1 << 16;

/// The FIPS-197 Appendix B key, expanded to 44 round-key words.
pub fn demo_round_keys() -> [u32; 44] {
    key_expansion([0x2B7E1516, 0x28AED2A6, 0xABF71588, 0x09CF4F3C])
}

/// The FIPS-197 Appendix B plaintext block.
pub fn demo_block() -> [u32; 4] {
    [0x3243F6A8, 0x885A308D, 0x313198A2, 0xE0370734]
}

/// A batch of `n` identical plaintext blocks (the FIPS-197 Appendix B vector).
pub fn demo_blocks(n: usize) -> Vec<[u32; 4]> {
    vec![demo_block(); n]
}
