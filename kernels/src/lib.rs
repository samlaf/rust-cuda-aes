//! AES-128 GPU kernels, ported from Tezcan's CUDA_AES.
//!
//! References:
//! - C. Tezcan, "Optimization of Advanced Encryption Standard on Graphics
//!   Processing Units", IEEE Access 2021 (eprint 2021/646).
//! - https://github.com/cihangirtezcan/CUDA_AES/blob/gh-pages/128-ctr.cuh
//!
//! Step 2 (this file): counter (CTR) mode. A single kernel [`aes128_ctr`]
//! replaces the earlier pair of ECB kernels — it applies the cipher to a
//! per-block counter and XORs the keystream into the plaintext
//! (`ct[i] = pt[i] ⊕ E(k, counter₀ + i)`). The portable round logic and the
//! constant tables live in the [`aes_core`] crate, so the exact same
//! `encrypt_ctr_block` runs here on the GPU and in the CPU tests/benches.
//!
//! A runtime `blocks_per_thread` (`R`) controls how many consecutive counters
//! each thread covers, so one kernel subsumes every launch shape: `n_blocks = 1`
//! is the single-block latency path, `R = 1` is one-thread-per-block (the direct
//! ECB→CTR swap), and `R > 1` is the paper's arithmetic-intensity win (each
//! thread reuses the loaded round keys/tables across `R` counters). Everything
//! still lives in plain global memory; the shared-memory / `__byte_perm` /
//! bank-conflict optimizations come in later steps.

use aes_core::encrypt_ctr_block;
use cuda_std::prelude::*;

// ---------------------------------------------------------------------------
// Kernel: CTR-mode encryption of a batch of blocks.
// ---------------------------------------------------------------------------

/// Batch AES-128-CTR encryption: thread `t` encrypts the `R` consecutive blocks
/// `t·R … t·R + R - 1`, where `R = blocks_per_thread`.
///
/// - `counter0`: the 4-word initial counter block; block `i` uses `counter0`
///   with its low 32-bit word increased by `i` (see [`encrypt_ctr_block`]).
/// - `pt`/`ct`: `n_blocks` consecutive blocks of 4 words each (plaintext in,
///   ciphertext out).
/// - `rk`:   44 expanded round-key words.
/// - `t0..t3`: the four T-tables (256 words each).
/// - `sbox`: the S-box (256 words) for the last round.
/// - `n_blocks`: number of real blocks; indices past it are skipped (the launch
///   is rounded up to whole threads/blocks).
///
/// This reads the T-tables from global memory (the shared-memory optimizations
/// come later). See [`encrypt_ctr_block`] for the per-block parameters.
#[kernel]
#[allow(improper_ctypes_definitions)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn aes128_ctr(
    counter0: &[u32],
    pt: &[u32],
    rk: &[u32],
    t0: &[u32],
    t1: &[u32],
    t2: &[u32],
    t3: &[u32],
    sbox: &[u32],
    ct: *mut u32,
    n_blocks: usize,
    blocks_per_thread: usize,
) {
    let counter0 = [counter0[0], counter0[1], counter0[2], counter0[3]];
    let start = thread::index_1d() as usize * blocks_per_thread;

    for j in 0..blocks_per_thread {
        let i = start + j;
        if i >= n_blocks {
            return;
        }

        let base = i * 4;
        let out = encrypt_ctr_block(
            counter0,
            i as u32,
            [pt[base], pt[base + 1], pt[base + 2], pt[base + 3]],
            rk,
            t0,
            t1,
            t2,
            t3,
            sbox,
        );

        unsafe {
            *ct.add(base) = out[0];
            *ct.add(base + 1) = out[1];
            *ct.add(base + 2) = out[2];
            *ct.add(base + 3) = out[3];
        }
    }
}
