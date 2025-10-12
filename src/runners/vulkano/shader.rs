use crate::error::{ChimeraError, CrateResult};

use std::sync::Arc;

use vulkano::{
    device::Device,
    shader::{EntryPoint, ShaderModule},
};

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
    Ok(shader_module.entry_point(entry_point_name).ok_or_else(|| {
        ChimeraError::Other(format!(
            "Entry point '{entry_point_name}' not found in SPIR-V"
        ))
    })?)
}
