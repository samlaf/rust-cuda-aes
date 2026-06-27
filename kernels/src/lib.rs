//! AES-128 GPU kernels, ported from Tezcan's CUDA_AES.
//!
//! References:
//! - C. Tezcan, "Optimization of Advanced Encryption Standard on Graphics
//!   Processing Units", IEEE Access 2021 (eprint 2021/646).
//! - https://github.com/cihangirtezcan/CUDA_AES/blob/gh-pages/128-ctr.cuh
//!
//! Step 1 (this file): the *naive* table-based baseline the paper starts from —
//! `aes128_encrypt_block` (one thread, one block; used by the round-trip test)
//! and `aes128_encrypt_blocks` (one thread per block, the benchmark workload).
//! The portable round logic and the constant tables live in the [`aes_core`]
//! crate, so the exact same `encrypt_block` runs here on the GPU and in the CPU
//! tests/benches. Everything lives in plain global memory; the shared-memory /
//! `__byte_perm` / bank-conflict optimizations come in later steps.

use aes_core::encrypt_block;
use cuda_std::prelude::*;

// ---------------------------------------------------------------------------
// Kernel: encrypt a single 16-byte block with one thread.
// ---------------------------------------------------------------------------

/// Single-block AES-128 encryption.
///
/// - `pt`:   4 words of plaintext (big-endian within each word).
/// - `rk`:   44 expanded round-key words.
/// - `t0..t3`: the four T-tables (256 words each).
/// - `sbox`: the S-box (256 words) for the last round.
/// - `ct`:   output pointer for the 4 ciphertext words.
#[kernel]
#[allow(improper_ctypes_definitions)]
pub unsafe fn aes128_encrypt_block(
    pt: &[u32],
    rk: &[u32],
    t0: &[u32],
    t1: &[u32],
    t2: &[u32],
    t3: &[u32],
    sbox: &[u32],
    ct: *mut u32,
) {
    // Only one thread does the work for this single-block baseline.
    if thread::index_1d() != 0 {
        return;
    }

    let out = encrypt_block([pt[0], pt[1], pt[2], pt[3]], rk, t0, t1, t2, t3, sbox);

    unsafe {
        *ct.add(0) = out[0];
        *ct.add(1) = out[1];
        *ct.add(2) = out[2];
        *ct.add(3) = out[3];
    }
}

// ---------------------------------------------------------------------------
// Kernel: encrypt a batch of blocks, one thread per 16-byte block.
// ---------------------------------------------------------------------------

/// Batch AES-128 encryption (ECB): thread `i` encrypts block `i`.
///
/// `pt` and `ct` hold `n_blocks` consecutive blocks of 4 words each. Threads
/// past `n_blocks` (the launch is rounded up to a whole block size) return
/// without doing anything. This is the parallel workload the benchmarks use; it
/// still reads the T-tables from global memory (the shared-memory optimizations
/// come later). See [`encrypt_block`] for the per-block parameters.
#[kernel]
#[allow(improper_ctypes_definitions)]
pub unsafe fn aes128_encrypt_blocks(
    pt: &[u32],
    rk: &[u32],
    t0: &[u32],
    t1: &[u32],
    t2: &[u32],
    t3: &[u32],
    sbox: &[u32],
    ct: *mut u32,
    n_blocks: usize,
) {
    let i = thread::index_1d() as usize;
    if i >= n_blocks {
        return;
    }

    let base = i * 4;
    let out = encrypt_block(
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
