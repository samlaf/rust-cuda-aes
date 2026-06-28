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

    /// Encrypt a single 16-byte block on the GPU.
    ///
    /// `pt` is 4 plaintext words (big-endian within each word) and `rk` is the
    /// 44-word expanded key from [`aes_core::key_expansion`]. Returns the 4
    /// ciphertext words.
    pub fn encrypt_block(&self, pt: [u32; 4], rk: &[u32]) -> CudaResult<[u32; 4]> {
        let pt_gpu = pt.as_dbuf()?;
        let rk_gpu = rk.as_dbuf()?;
        let mut ct = [0u32; 4];
        let ct_gpu = ct.as_dbuf()?;

        let kernel = self.module.get_function("aes128_encrypt_block")?;
        // `launch!` wants the stream as a plain identifier.
        let stream = &self.stream;
        // One block, one thread (the naive baseline kernel).
        unsafe {
            launch!(
                kernel<<<1, 1, 0, stream>>>(
                    pt_gpu.as_device_ptr(), pt_gpu.len(),
                    rk_gpu.as_device_ptr(), rk_gpu.len(),
                    self.t0.as_device_ptr(), self.t0.len(),
                    self.t1.as_device_ptr(), self.t1.len(),
                    self.t2.as_device_ptr(), self.t2.len(),
                    self.t3.as_device_ptr(), self.t3.len(),
                    self.sbox.as_device_ptr(), self.sbox.len(),
                    ct_gpu.as_device_ptr(),
                )
            )?;
        }
        self.stream.synchronize()?;
        ct_gpu.copy_to(&mut ct)?;
        Ok(ct)
    }

    /// Encrypt many 16-byte blocks (ECB), one GPU thread per block.
    ///
    /// `rk` is the 44-word expanded key; `out` must be the same length as
    /// `blocks`. End-to-end: uploads the inputs, launches the batch kernel, and
    /// copies the ciphertext back — that whole cost is what the GPU benchmark
    /// measures.
    pub fn encrypt_blocks(
        &self,
        rk: &[u32],
        blocks: &[[u32; 4]],
        out: &mut [[u32; 4]],
    ) -> CudaResult<()> {
        let n = blocks.len();
        if n == 0 {
            return Ok(());
        }
        debug_assert_eq!(out.len(), n, "output length must match the input");

        let pt_gpu = blocks.as_flattened().as_dbuf()?;
        let rk_gpu = rk.as_dbuf()?;
        let ct_gpu = DeviceBuffer::<u32>::zeroed(n * 4)?;

        let kernel = self.module.get_function("aes128_encrypt_blocks")?;
        let stream = &self.stream;
        // One thread per block; round the grid up to a whole block of threads.
        const BLOCK: u32 = 256;
        let grid = (n as u32).div_ceil(BLOCK);
        unsafe {
            launch!(
                kernel<<<grid, BLOCK, 0, stream>>>(
                    pt_gpu.as_device_ptr(), pt_gpu.len(),
                    rk_gpu.as_device_ptr(), rk_gpu.len(),
                    self.t0.as_device_ptr(), self.t0.len(),
                    self.t1.as_device_ptr(), self.t1.len(),
                    self.t2.as_device_ptr(), self.t2.len(),
                    self.t3.as_device_ptr(), self.t3.len(),
                    self.sbox.as_device_ptr(), self.sbox.len(),
                    ct_gpu.as_device_ptr(),
                    n,
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
    use aes_core::{key_expansion, KAT_VECTORS};

    /// Round-trips the shared FIPS-197 vectors through the real kernel (upload,
    /// launch, download). Requires an NVIDIA GPU + CUDA toolkit, so it only runs
    /// on a CUDA host — the crate doesn't build without one. A single `AesGpu` is
    /// shared so the CUDA context is initialized exactly once.
    #[test]
    fn gpu_matches_fips197() {
        let gpu = AesGpu::new().expect("CUDA init failed (needs an NVIDIA GPU)");
        for &(pt, key, expected) in KAT_VECTORS {
            let rk = key_expansion(key);
            let ct = gpu.encrypt_block(pt, &rk).expect("kernel launch failed");
            assert_eq!(ct, expected, "GPU ciphertext mismatch for pt={pt:08x?}");
        }
    }
}
