use crate::{error::Result, runners::vulkano::shader_buffer_mapping::BufNameToBinding};
use std::sync::Arc;

use std::collections::BTreeMap;
use vulkano::{
    descriptor_set::layout::{
        DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo,
        DescriptorType,
    },
    device::Device,
    shader::ShaderStages,
};

pub fn build_descriptor_set(
    device: Arc<Device>,
    global_buf_to_binding: BufNameToBinding,
) -> Result<Arc<DescriptorSetLayout>> {
    let mut bindings = BTreeMap::new();

    for (_buf_name, binding) in global_buf_to_binding.0.iter() {
        let mut binding_desc =
            DescriptorSetLayoutBinding::descriptor_type(DescriptorType::StorageBuffer);
        binding_desc.stages = ShaderStages::COMPUTE;
        binding_desc.descriptor_count = 1;

        bindings.insert(*binding, binding_desc);
    }

    let layout = DescriptorSetLayout::new(
        device.clone(),
        DescriptorSetLayoutCreateInfo {
            bindings,
            ..Default::default()
        },
    )?;

    Ok(layout)
}
