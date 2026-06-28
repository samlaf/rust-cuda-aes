//! The benchmark variant registry — one entry per tutorial step.
//!
//! Every kept version of AES (CPU scalar, AES-NI, VAES, and each GPU kernel /
//! transfer strategy) is exposed here as a named [`Aes`] implementation. The
//! throughput benchmark (`throughput.rs`) and the known-answer test below both
//! iterate these registries, so adding an optimization step is a one-line change
//! here and it automatically shows up — side by side with every earlier step —
//! in the `aes128-ctr` benchmark report and in the correctness test.
//!
//! The library crates (`aes-cpu`, `aes-gpu`) hold the actual implementations; the
//! thin wrappers here just give each one a stable name behind a single interface
//! so the numbers are directly comparable.

/// One AES-128-CTR "encrypt a batch of blocks" strategy.
///
/// `encrypt_ctr` is infallible by contract: a backend that can fail (a CUDA
/// error on the GPU) panics rather than returning a `Result`, because in this
/// measurement harness a backend that can't run is a hard error, not a value to
/// thread through every call site.
pub trait Aes {
    /// Short label used as the criterion benchmark id and in test output. Prefix
    /// with the tutorial step order (e.g. `"gpu/2-pinned"`) so reports sort in
    /// narrative order.
    fn name(&self) -> &str;

    /// Encrypt `blocks` into `out` in CTR mode (`out[i] = blocks[i] ⊕ E(rk,
    /// counter0 + i)`); `out.len()` must equal `blocks.len()`. GPU backends pin
    /// blocks-per-thread at 1 for now (one thread per block).
    fn encrypt_ctr(&self, rk: &[u32], counter0: [u32; 4], blocks: &[[u32; 4]], out: &mut [[u32; 4]]);
}

// ---------------------------------------------------------------------------
// CPU variants (available everywhere; no GPU required).
// ---------------------------------------------------------------------------

/// Portable scalar T-table backend (`aes_cpu::scalar`).
pub struct CpuScalar;

impl Aes for CpuScalar {
    fn name(&self) -> &str {
        "cpu/scalar"
    }
    fn encrypt_ctr(&self, rk: &[u32], counter0: [u32; 4], blocks: &[[u32; 4]], out: &mut [[u32; 4]]) {
        aes_cpu::scalar::encrypt_ctr(rk, counter0, blocks, out);
    }
}

/// Every CPU variant, in tutorial order.
pub fn cpu_variants() -> Vec<Box<dyn Aes>> {
    vec![Box::new(CpuScalar)]
}

// ---------------------------------------------------------------------------
// GPU variants (only built with the `gpu` feature, which needs CUDA).
// ---------------------------------------------------------------------------

/// The naive baseline GPU path: tables in global memory, pageable host buffers
/// allocated per call (`aes_gpu::AesGpu::encrypt_ctr`). This is the strawman
/// the later transfer/kernel optimizations are measured against.
#[cfg(feature = "gpu")]
pub struct GpuGlobalPageable(pub aes_gpu::AesGpu);

#[cfg(feature = "gpu")]
impl Aes for GpuGlobalPageable {
    fn name(&self) -> &str {
        "gpu/0-global-pageable"
    }
    fn encrypt_ctr(&self, rk: &[u32], counter0: [u32; 4], blocks: &[[u32; 4]], out: &mut [[u32; 4]]) {
        // One thread per block (blocks_per_thread = 1) — the direct ECB→CTR swap.
        self.0
            .encrypt_ctr(rk, counter0, blocks, out, 1)
            .expect("GPU encrypt_ctr failed");
    }
}

/// Every GPU variant, in tutorial order. Initializes CUDA (panics without an
/// NVIDIA GPU). For now each variant owns its own context; once there are several
/// they can share one.
#[cfg(feature = "gpu")]
pub fn gpu_variants() -> Vec<Box<dyn Aes>> {
    let gpu = aes_gpu::AesGpu::new().expect("CUDA init failed (needs an NVIDIA GPU)");
    vec![Box::new(GpuGlobalPageable(gpu))]
}

#[cfg(test)]
mod tests {
    use super::*;
    use aes_core::{key_expansion, CTR_KAT, KAT_VECTORS};

    /// The shared correctness gate every kept variant must pass, in CTR terms:
    ///
    /// 1. Cipher core — with `counter0 = PT` and zero plaintext, `out[0] = 0 ⊕
    ///    E(k, PT) = E(k, PT)`, so the FIPS-197 vectors check the round function
    ///    through the CTR path.
    /// 2. CTR mode itself (counter increment + keystream XOR) — the multi-block
    ///    NIST SP 800-38A F.5.1 vector.
    fn check_kat(variant: &dyn Aes) {
        for &(pt, key, expected) in KAT_VECTORS {
            let rk = key_expansion(key);
            let mut out = [[0u32; 4]];
            variant.encrypt_ctr(&rk, pt, &[[0u32; 4]], &mut out);
            assert_eq!(
                out[0],
                expected,
                "variant {} failed FIPS-197 cipher-core KAT",
                variant.name()
            );
        }

        let rk = key_expansion(CTR_KAT.key);
        let mut out = [[0u32; 4]; 4];
        variant.encrypt_ctr(&rk, CTR_KAT.counter0, &CTR_KAT.plaintext, &mut out);
        assert_eq!(
            out,
            CTR_KAT.ciphertext,
            "variant {} failed NIST F.5.1 CTR KAT",
            variant.name()
        );
    }

    #[test]
    fn cpu_variants_match_fips197() {
        for v in cpu_variants() {
            check_kat(v.as_ref());
        }
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn gpu_variants_match_fips197() {
        for v in gpu_variants() {
            check_kat(v.as_ref());
        }
    }
}
