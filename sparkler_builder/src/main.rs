use spirv_builder::{MetadataPrintout, SpirvBuilder};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // print working directory for debugging
    println!(
        "cargo:warning=Current working directory: {}",
        std::env::current_dir()?.display()
    );

    let build_result =
        SpirvBuilder::new("../sparkler_gen_kernels_sketch", "spirv-unknown-vulkan1.2")
            .print_metadata(MetadataPrintout::Full)
            .build();

    if let Ok(result) = build_result {
        println!(
            "cargo:warning=SPIR-V module built to: {}",
            result.module.unwrap_single().display()
        );
        // Export the kernel path for the runtime to use
        println!("kernel path: {}", result.module.unwrap_single().display());

        // Use the first entry point
        println!("entry points: {:?}", result.entry_points);
    } else {
        println!(
            "cargo:warning=Failed to build SPIR-V module: {:?}",
            build_result
        );
        return Err("Failed to build SPIR-V module".into());
    }

    Ok(())
}
