//! CPU AES-128 implementations, used as the benchmark baseline for the GPU
//! kernels.
//!
//! - [`scalar`]: portable software AES-CTR using the shared T-tables
//!   (`aes_core`). The "simple for-loop" reference that runs everywhere.
//! - [`aesni`]: AES-NI hardware AES (x86_64), one 128-bit block per instruction,
//!   then interleaved / `PSHUFB`-dataflow / multi-core kernels on top.
//! - [`vaes`]: wider VAES path (x86_64) — 2 blocks per instruction (256-bit,
//!   `vaes`+`avx2`) and 4 blocks per instruction (512-bit, `vaes`+AVX-512), plus
//!   multi-core wrappers. Registered/run only where the CPU reports the features.
//!
//! Everything here is host-only (it needs `std` and, for `aesni`/`vaes`, x86 SIMD
//! intrinsics), which is why it lives outside the `kernels` device crate. The
//! shared algorithm/tables live in `aes_core` so all backends agree on the same
//! byte order and can be checked against the same FIPS-197 known answers.

pub mod scalar;

#[cfg(target_arch = "x86_64")]
pub mod aesni;

#[cfg(target_arch = "x86_64")]
pub mod vaes;
