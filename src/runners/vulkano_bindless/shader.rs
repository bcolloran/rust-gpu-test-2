use crate::error::{ChimeraError, CrateResult};

use std::sync::Arc;

use vulkano::{
    device::Device,
    shader::{EntryPoint, ShaderModule},
};

/// Create a shader module containing the bindless shader variants
///
/// This uses the same SPIR-V binary as the regular shaders, but we'll
/// reference the "bindless::" prefixed entry points (bindless::adder, bindless::step_particles, etc.)
pub fn shader_module(device: Arc<Device>) -> CrateResult<Arc<ShaderModule>> {
    // Create shader module from embedded SPIR-V
    let kernel_bytes = crate::OTHER_SHADERS_SPIRV;
    // Convert SPIR-V bytes to words then create shader module
    let words = vulkano::shader::spirv::bytes_to_words(kernel_bytes)?;
    let shader_module = unsafe {
        match ShaderModule::new(
            device.clone(),
            vulkano::shader::ShaderModuleCreateInfo::new(&words),
        ) {
            Ok(m) => m,
            Err(e) => {
                // Provide more detailed diagnostics using Debug formatting
                return Err(ChimeraError::Other(format!(
                    "Failed to create shader module: {e:?}"
                )));
            }
        }
    };
    Ok(shader_module)
}

pub fn shader_entry_point(
    shader_module: Arc<ShaderModule>,
    entry_point_name: &str,
) -> CrateResult<EntryPoint> {
    // For bindless shaders, we prepend "bindless::" to the entry point name
    let bindless_entry_name = format!("bindless::{}", entry_point_name);

    Ok(shader_module
        .entry_point(&bindless_entry_name)
        .ok_or_else(|| {
            ChimeraError::Other(format!(
                "Entry point '{}' not found in SPIR-V",
                bindless_entry_name
            ))
        })?)
}
