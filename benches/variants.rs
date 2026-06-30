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

/// AES-NI, one block at a time (`encrypt_ctr::<1>`), x86_64 only. The
/// latency-bound baseline the interleaved `cpu/aesni-x8` is measured against.
#[cfg(target_arch = "x86_64")]
pub struct CpuAesNi;

#[cfg(target_arch = "x86_64")]
impl Aes for CpuAesNi {
    fn name(&self) -> &str {
        "cpu/aesni"
    }
    fn encrypt_ctr(&self, rk: &[u32], counter0: [u32; 4], blocks: &[[u32; 4]], out: &mut [[u32; 4]]) {
        aes_cpu::aesni::encrypt_ctr::<1>(rk, counter0, blocks, out);
    }
}

/// AES-NI with 8-way block interleaving (`encrypt_ctr::<8>`), x86_64 only. Hides
/// the ~4-cycle AESENC latency that the one-block `cpu/aesni` is bound by, so it
/// should be several times faster on the same core.
#[cfg(target_arch = "x86_64")]
pub struct CpuAesNiX8;

#[cfg(target_arch = "x86_64")]
impl Aes for CpuAesNiX8 {
    fn name(&self) -> &str {
        "cpu/aesni-x8"
    }
    fn encrypt_ctr(&self, rk: &[u32], counter0: [u32; 4], blocks: &[[u32; 4]], out: &mut [[u32; 4]]) {
        aes_cpu::aesni::encrypt_ctr::<8>(rk, counter0, blocks, out);
    }
}

/// AES-NI x8 with the per-block byte handling kept in the SIMD domain
/// (`encrypt_ctr_pshufb::<8>`), x86_64 only. Same interleaving as `cpu/aesni-x8`,
/// but counters live in `__m128i`, the byte swap is one `PSHUFB`, and plaintext
/// is XORed as `__m128i` — removing the scalar repack that pins `cpu/aesni-x8`
/// well below the `AESENC` ceiling.
#[cfg(target_arch = "x86_64")]
pub struct CpuAesNiX8Pshufb;

#[cfg(target_arch = "x86_64")]
impl Aes for CpuAesNiX8Pshufb {
    fn name(&self) -> &str {
        "cpu/aesni-x8-pshufb"
    }
    fn encrypt_ctr(&self, rk: &[u32], counter0: [u32; 4], blocks: &[[u32; 4]], out: &mut [[u32; 4]]) {
        aes_cpu::aesni::encrypt_ctr_pshufb::<8>(rk, counter0, blocks, out);
    }
}

/// The per-core `cpu/aesni-x8-pshufb` kernel fanned across CPU cores with rayon
/// (`encrypt_ctr_pshufb_parallel::<8>`), x86_64 only. The two wins compose: each
/// core runs the repack-free SIMD kernel over an independent slice of counter
/// blocks, so an N-core box stacks ~N× on top of the per-core pshufb throughput.
#[cfg(target_arch = "x86_64")]
pub struct CpuAesNiX8PshufbParallel;

#[cfg(target_arch = "x86_64")]
impl Aes for CpuAesNiX8PshufbParallel {
    fn name(&self) -> &str {
        "cpu/aesni-x8-pshufb-parallel"
    }
    fn encrypt_ctr(&self, rk: &[u32], counter0: [u32; 4], blocks: &[[u32; 4]], out: &mut [[u32; 4]]) {
        aes_cpu::aesni::encrypt_ctr_pshufb_parallel::<8>(rk, counter0, blocks, out);
    }
}

/// 256-bit VAES: 2 blocks per `VAESENC`, 8 wide registers interleaved
/// (`encrypt_ctr_vaes256::<8>`), x86_64 + `vaes`/`avx2` only. The next rung above
/// `cpu/aesni-x8-pshufb` — and it keeps the *same* `-x8-pshufb` machinery (8-way
/// interleave + the `PSHUFB` SIMD-dataflow byte swap, now `_mm256_shuffle_epi8`);
/// only the AES round widens, to 2 independent blocks per instruction. The suffix
/// is carried so the report can't be misread as "VAES dropped pshufb."
#[cfg(target_arch = "x86_64")]
pub struct CpuVaes256X8Pshufb;

#[cfg(target_arch = "x86_64")]
impl Aes for CpuVaes256X8Pshufb {
    fn name(&self) -> &str {
        "cpu/vaes256-x8-pshufb"
    }
    fn encrypt_ctr(&self, rk: &[u32], counter0: [u32; 4], blocks: &[[u32; 4]], out: &mut [[u32; 4]]) {
        aes_cpu::vaes::encrypt_ctr_vaes256::<8>(rk, counter0, blocks, out);
    }
}

/// 512-bit VAES: 4 blocks per `VAESENC`, 8 wide registers interleaved
/// (`encrypt_ctr_vaes512::<8>`), x86_64 + `vaes`/AVX-512 only. Same `-x8-pshufb`
/// machinery as the rest (the byte swap is `_mm512_shuffle_epi8` here); doubles
/// the 256-bit path again on a native 512-bit datapath (Zen 5, Ice Lake+).
#[cfg(target_arch = "x86_64")]
pub struct CpuVaes512X8Pshufb;

#[cfg(target_arch = "x86_64")]
impl Aes for CpuVaes512X8Pshufb {
    fn name(&self) -> &str {
        "cpu/vaes512-x8-pshufb"
    }
    fn encrypt_ctr(&self, rk: &[u32], counter0: [u32; 4], blocks: &[[u32; 4]], out: &mut [[u32; 4]]) {
        aes_cpu::vaes::encrypt_ctr_vaes512::<8>(rk, counter0, blocks, out);
    }
}

/// `cpu/vaes256-x8-pshufb` fanned across all cores with rayon
/// (`encrypt_ctr_vaes256_parallel::<8>`), x86_64 + `vaes`/`avx2` only.
#[cfg(target_arch = "x86_64")]
pub struct CpuVaes256X8PshufbParallel;

#[cfg(target_arch = "x86_64")]
impl Aes for CpuVaes256X8PshufbParallel {
    fn name(&self) -> &str {
        "cpu/vaes256-x8-pshufb-parallel"
    }
    fn encrypt_ctr(&self, rk: &[u32], counter0: [u32; 4], blocks: &[[u32; 4]], out: &mut [[u32; 4]]) {
        aes_cpu::vaes::encrypt_ctr_vaes256_parallel::<8>(rk, counter0, blocks, out);
    }
}

/// `cpu/vaes512-x8-pshufb` fanned across all cores with rayon
/// (`encrypt_ctr_vaes512_parallel::<8>`), x86_64 + `vaes`/AVX-512 only. The top
/// CPU rung: widest instruction × every core.
#[cfg(target_arch = "x86_64")]
pub struct CpuVaes512X8PshufbParallel;

#[cfg(target_arch = "x86_64")]
impl Aes for CpuVaes512X8PshufbParallel {
    fn name(&self) -> &str {
        "cpu/vaes512-x8-pshufb-parallel"
    }
    fn encrypt_ctr(&self, rk: &[u32], counter0: [u32; 4], blocks: &[[u32; 4]], out: &mut [[u32; 4]]) {
        aes_cpu::vaes::encrypt_ctr_vaes512_parallel::<8>(rk, counter0, blocks, out);
    }
}

/// Every CPU variant available on this target, in tutorial order. The AES-NI
/// variants are added only on an x86_64 CPU that has AES-NI; the VAES variants
/// only where the CPU also reports `vaes` (+ AVX2 for 256-bit, AVX-512 for
/// 512-bit), so they're simply absent on a Zen 2 box and present on a Zen 4+ /
/// Ice Lake+ one — same auto-skip pattern as the `aes` guard.
pub fn cpu_variants() -> Vec<Box<dyn Aes>> {
    #[allow(unused_mut)]
    let mut v: Vec<Box<dyn Aes>> = vec![Box::new(CpuScalar)];
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("aes") {
        v.push(Box::new(CpuAesNi));
        v.push(Box::new(CpuAesNiX8));
        v.push(Box::new(CpuAesNiX8Pshufb));
        v.push(Box::new(CpuAesNiX8PshufbParallel));
    }
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("vaes") && std::is_x86_feature_detected!("avx2") {
        v.push(Box::new(CpuVaes256X8Pshufb));
        v.push(Box::new(CpuVaes256X8PshufbParallel));
    }
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("vaes")
        && std::is_x86_feature_detected!("avx512f")
        && std::is_x86_feature_detected!("avx512bw")
    {
        v.push(Box::new(CpuVaes512X8Pshufb));
        v.push(Box::new(CpuVaes512X8PshufbParallel));
    }
    v
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
