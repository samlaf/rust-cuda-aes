//! GPU-backed AES-128, built on [rust-cuda](https://rust-gpu.github.io/rust-cuda).
//!
//! The `kernels` crate is compiled to PTX at build time (see `build.rs`); this
//! crate embeds that PTX, loads it onto the device, and exposes [`AesGpu`] — a
//! handle that keeps the CUDA context live and the constant T-tables / S-box
//! resident, so each encryption only has to upload its inputs and launch.
//!
//! This is the GPU counterpart to the CPU backends in `aes-cpu`; both encrypt
//! through the shared `aes_core` round function.

use aes_core::{SBOX_U32, T0, T1, T2, T3};
use cust::error::CudaResult;
use cust::prelude::*;

/// PTX generated from the `kernels` crate by `build.rs`.
static PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernels.ptx"));

/// A live CUDA context with the AES kernels loaded and the constant tables
/// uploaded once. Reuse a single handle across many encryptions.
pub struct AesGpu {
    module: Module,
    stream: Stream,
    t0: DeviceBuffer<u32>,
    t1: DeviceBuffer<u32>,
    t2: DeviceBuffer<u32>,
    t3: DeviceBuffer<u32>,
    sbox: DeviceBuffer<u32>,
    // Declared last so it drops last: the module, stream, and buffers above all
    // require the context to still be alive when they are freed.
    _ctx: Context,
}

impl AesGpu {
    /// Initialize CUDA, load the kernels, and upload the constant tables.
    pub fn new() -> CudaResult<Self> {
        let _ctx = cust::quick_init()?;
        let module = Module::from_ptx(PTX, &[])?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
        Ok(Self {
            module,
            stream,
            t0: T0.as_dbuf()?,
            t1: T1.as_dbuf()?,
            t2: T2.as_dbuf()?,
            t3: T3.as_dbuf()?,
            sbox: SBOX_U32.as_dbuf()?,
            _ctx,
        })
    }

    /// Encrypt many 16-byte blocks in CTR mode: `out[i] = blocks[i] ⊕ E(rk,
    /// counter0 + i)`.
    ///
    /// `rk` is the 44-word expanded key; `counter0` is the initial counter block
    /// (incremented per block in its low 32-bit word, see
    /// [`aes_core::encrypt_ctr_block`]); `out` must be the same length as
    /// `blocks`. `blocks_per_thread` (`R`) is how many consecutive blocks each
    /// GPU thread covers — `R = 1` is one-thread-per-block, `R > 1` trades
    /// threads for per-thread arithmetic intensity; `n_blocks = 1` is the
    /// single-block latency path.
    ///
    /// End-to-end: uploads the inputs, launches the kernel, and copies the
    /// ciphertext back — that whole cost is what the GPU benchmark measures.
    pub fn encrypt_ctr(
        &self,
        rk: &[u32],
        counter0: [u32; 4],
        blocks: &[[u32; 4]],
        out: &mut [[u32; 4]],
        blocks_per_thread: usize,
    ) -> CudaResult<()> {
        let n = blocks.len();
        if n == 0 {
            return Ok(());
        }
        debug_assert_eq!(out.len(), n, "output length must match the input");
        let r = blocks_per_thread.max(1);

        let counter0_gpu = counter0.as_dbuf()?;
        let pt_gpu = blocks.as_flattened().as_dbuf()?;
        let rk_gpu = rk.as_dbuf()?;
        let ct_gpu = DeviceBuffer::<u32>::zeroed(n * 4)?;

        let kernel = self.module.get_function("aes128_ctr")?;
        // `launch!` wants the stream as a plain identifier.
        let stream = &self.stream;
        // Each thread covers `r` blocks; round the thread count up to that, then
        // round the grid up to whole thread-blocks.
        const BLOCK: u32 = 256;
        let threads = (n as u32).div_ceil(r as u32);
        let grid = threads.div_ceil(BLOCK);
        unsafe {
            launch!(
                kernel<<<grid, BLOCK, 0, stream>>>(
                    counter0_gpu.as_device_ptr(), counter0_gpu.len(),
                    pt_gpu.as_device_ptr(), pt_gpu.len(),
                    rk_gpu.as_device_ptr(), rk_gpu.len(),
                    self.t0.as_device_ptr(), self.t0.len(),
                    self.t1.as_device_ptr(), self.t1.len(),
                    self.t2.as_device_ptr(), self.t2.len(),
                    self.t3.as_device_ptr(), self.t3.len(),
                    self.sbox.as_device_ptr(), self.sbox.len(),
                    ct_gpu.as_device_ptr(),
                    n,
                    r,
                )
            )?;
        }
        self.stream.synchronize()?;
        ct_gpu.copy_to(out.as_flattened_mut())?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::AesGpu;
    use aes_core::{key_expansion, CTR_KAT, KAT_VECTORS};

    /// Round-trips the shared known answers through the real CTR kernel (upload,
    /// launch, download). Requires an NVIDIA GPU + CUDA toolkit, so it only runs
    /// on a CUDA host — the crate doesn't build without one. One `AesGpu` is
    /// shared across both checks so the CUDA context is initialized exactly once.
    #[test]
    fn gpu_matches_known_answers() {
        let gpu = AesGpu::new().expect("CUDA init failed (needs an NVIDIA GPU)");

        // Cipher core: with counter0 = PT and zero plaintext, out[0] = E(k, PT),
        // so the FIPS-197 vectors check the round function through the CTR path.
        for &(pt, key, expected) in KAT_VECTORS {
            let rk = key_expansion(key);
            let mut out = [[0u32; 4]];
            gpu.encrypt_ctr(&rk, pt, &[[0u32; 4]], &mut out, 1)
                .expect("kernel launch failed");
            assert_eq!(out[0], expected, "GPU cipher-core mismatch for pt={pt:08x?}");
        }

        // CTR mode itself (counter increment + keystream XOR): NIST F.5.1.
        let rk = key_expansion(CTR_KAT.key);
        let mut out = [[0u32; 4]; 4];
        gpu.encrypt_ctr(&rk, CTR_KAT.counter0, &CTR_KAT.plaintext, &mut out, 1)
            .expect("kernel launch failed");
        assert_eq!(out, CTR_KAT.ciphertext, "GPU CTR-mode mismatch (NIST F.5.1)");
    }
}
