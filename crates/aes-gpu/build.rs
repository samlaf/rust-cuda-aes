use std::env;
use std::path;

use cuda_builder::CudaBuilder;

fn main() {
    // The shared core crate is a sibling under crates/; the GPU kernel crate is
    // a standalone (non-member) crate at the workspace root.
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=../../kernels");
    // The kernels crate depends on aes-core, so its sources affect the PTX too.
    println!("cargo::rerun-if-changed=../aes-core");

    let out_dir = path::PathBuf::from(env::var("OUT_DIR").unwrap());
    let manifest_dir = path::PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    // crates/aes-gpu -> workspace root.
    let workspace_root = manifest_dir.parent().unwrap().parent().unwrap();

    // Compile the `kernels` crate to `$OUT_DIR/kernels.ptx`.
    CudaBuilder::new(workspace_root.join("kernels"))
        .copy_to(out_dir.join("kernels.ptx"))
        .build()
        .unwrap();
}
