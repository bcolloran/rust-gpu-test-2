//! Build script for compiling kernels to SPIR-V and CUDA PTX
fn main() {
    println!("cargo:rerun-if-changed=kernel/src");
    // Only build kernels when the appropriate features are enabled
    #[cfg(any(feature = "vulkan", feature = "wgpu"))]
    build_spirv_kernel();
}

#[cfg(any(feature = "vulkan", feature = "wgpu"))]
fn build_spirv_kernel() {
    use spirv_builder::SpirvBuilder;
    use std::path::PathBuf;

    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let crate_path = PathBuf::from(manifest_dir).join("kernel");

    let result = SpirvBuilder::new(crate_path, "spirv-unknown-vulkan1.2")
        .print_metadata(spirv_builder::MetadataPrintout::Full)
        .build()
        .unwrap();

    println!(
        "cargo:warning=building SPIRV to: {}",
        result.module.unwrap_single().display()
    );
    // Export the kernel path for the runtime to use
    println!(
        "cargo:rustc-env=BITONIC_KERNEL_SPV_PATH={}",
        result.module.unwrap_single().display()
    );

    // Use the first entry point
    println!(
        "cargo:rustc-env=BITONIC_KERNEL_SPV_ENTRY={}",
        result.entry_points.first().unwrap()
    );

    println!(
        "cargo:rustc-env=OTHER_SHADERS_SPV_PATH=/data/code_projects/rust/rust-gpu-sparkler/target/spirv-builder/spirv-unknown-vulkan1.2/release/deps/operators.spv"
    );

    // Use the first entry point
    println!("cargo:rustc-env=OTHER_SHADERS_ENTRY_ADDER=adder",);
    println!("cargo:rustc-env=OTHER_SHADERS_ENTRY_MULT=mult::mult",);
}
