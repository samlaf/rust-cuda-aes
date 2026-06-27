//! CPU AES-128 implementations, used as the benchmark baseline for the GPU
//! kernels.
//!
//! - [`scalar`]: portable software AES using the shared T-tables (`aes_core`).
//!   This is the "simple for-loop" reference that runs everywhere.
//! - [`vaes`]: AES-NI / VAES hardware-accelerated path (x86_64 only).
//!
//! Everything here is host-only (it needs `std` and, for `vaes`, x86 SIMD
//! intrinsics), which is why it lives outside the `kernels` device crate. The
//! shared algorithm/tables live in `aes_core` so all backends agree on the same
//! byte order and can be checked against the same FIPS-197 known answers.

pub mod scalar;

#[cfg(target_arch = "x86_64")]
pub mod vaes;
