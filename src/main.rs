use cust::prelude::*;
use kernels::{key_expansion, SBOX_U32, T0, T1, T2, T3};
use std::error::Error;

// Embed the PTX code as a static string.
static PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernels.ptx"));

fn main() -> Result<(), Box<dyn Error>> {
    // FIPS-197 Appendix B known-answer test.
    let pt: [u32; 4] = [0x3243F6A8, 0x885A308D, 0x313198A2, 0xE0370734];
    let key: [u32; 4] = [0x2B7E1516, 0x28AED2A6, 0xABF71588, 0x09CF4F3C];
    let expected: [u32; 4] = [0x3925841D, 0x02DC09FB, 0xDC118597, 0x196A0B32];

    let rk = key_expansion(key);

    // Initialize the CUDA Driver API. `_ctx` must be kept alive until the end.
    let _ctx = cust::quick_init()?;
    let module = Module::from_ptx(PTX, &[])?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    // Upload everything the kernel needs.
    let pt_gpu = pt.as_dbuf()?;
    let rk_gpu = rk.as_dbuf()?;
    let t0_gpu = T0.as_dbuf()?;
    let t1_gpu = T1.as_dbuf()?;
    let t2_gpu = T2.as_dbuf()?;
    let t3_gpu = T3.as_dbuf()?;
    let sbox_gpu = SBOX_U32.as_dbuf()?;

    let mut ct = vec![0u32; 4];
    let ct_gpu = ct.as_slice().as_dbuf()?;

    // One block, one thread.
    let kernel = module.get_function("aes128_encrypt_block")?;
    unsafe {
        launch!(
            kernel<<<1, 1, 0, stream>>>(
                pt_gpu.as_device_ptr(),
                pt_gpu.len(),
                rk_gpu.as_device_ptr(),
                rk_gpu.len(),
                t0_gpu.as_device_ptr(),
                t0_gpu.len(),
                t1_gpu.as_device_ptr(),
                t1_gpu.len(),
                t2_gpu.as_device_ptr(),
                t2_gpu.len(),
                t3_gpu.as_device_ptr(),
                t3_gpu.len(),
                sbox_gpu.as_device_ptr(),
                sbox_gpu.len(),
                ct_gpu.as_device_ptr(),
            )
        )?;
    }

    stream.synchronize()?;
    ct_gpu.copy_to(&mut ct)?;

    println!(
        "plaintext  : {:08x} {:08x} {:08x} {:08x}",
        pt[0], pt[1], pt[2], pt[3]
    );
    println!(
        "key        : {:08x} {:08x} {:08x} {:08x}",
        key[0], key[1], key[2], key[3]
    );
    println!(
        "ciphertext : {:08x} {:08x} {:08x} {:08x}",
        ct[0], ct[1], ct[2], ct[3]
    );
    println!(
        "expected   : {:08x} {:08x} {:08x} {:08x}",
        expected[0], expected[1], expected[2], expected[3]
    );

    if ct == expected {
        println!("KAT: PASS");
        Ok(())
    } else {
        println!("KAT: FAIL");
        Err("ciphertext does not match FIPS-197 known answer".into())
    }
}
