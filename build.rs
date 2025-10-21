//! Build script for compiling kernels to SPIR-V and CUDA PTX
fn main() {
    // println!("cargo:rerun-if-changed=kernel/src");
    // Only build kernels when the appropriate features are enabled
    #[cfg(any(feature = "vulkan"))]
    build_spirv_kernel();
}

#[cfg(any(feature = "vulkan"))]
fn build_spirv_kernel() {
    use spirv_builder::SpirvBuilder;
    use std::path::PathBuf;

    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let kernels_path = PathBuf::from(manifest_dir).join("shaders");

    let result = SpirvBuilder::new(kernels_path, "spirv-unknown-vulkan1.4")
        .scalar_block_layout(true)
        .print_metadata(spirv_builder::MetadataPrintout::Full)
        .build()
        .unwrap();

    println!(
        "cargo:warning=building SPIRV to: {}",
        result.module.unwrap_single().display()
    );
    println!(
        "cargo:warning=SPIRV entry points: {}",
        result.entry_points.join(", ")
    );
    // Export the kernel path for the runtime to use
    // println!(
    //     "cargo:rustc-env=BITONIC_KERNEL_SPV_PATH={}",
    //     result.module.unwrap_single().display()
    // );

    // // Use the first entry point
    // println!(
    //     "cargo:rustc-env=BITONIC_KERNEL_SPV_ENTRY={}",
    //     result.entry_points.first().unwrap()
    // );

    println!(
        "cargo:rustc-env=SHADERS_SPV_PATH={}",
        result.module.unwrap_single().display()
    );

    // Use the first entry point
    println!("cargo:rustc-env=SHADERS_ENTRY_ADDER=adder",);
}
