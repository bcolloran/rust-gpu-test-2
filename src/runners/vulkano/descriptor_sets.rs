use crate::runners::vulkano::buffer::BufNameToBinding;
use crate::{error::Result, runners::vulkano::shader_buffer_mapping::ShaderPipelineInfosWithEntry};
use std::{collections::HashMap, sync::Arc};

use std::collections::BTreeMap;
use vulkano::{
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator,
        layout::{
            DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo,
            DescriptorType,
        },
        DescriptorSet, WriteDescriptorSet,
    },
    device::Device,
    shader::ShaderStages,
};

/// Build the DescriptorSetLayout needed for the given set of (shaders, input buf requirements)
// pub fn build_abstract_descriptor_set_layouts(
//     device: Arc<Device>,
//     shader_bufs_and_entries: ShaderPipelineInfosWithEntry,
//     global_buf_to_binding: BufNameToBinding,
// ) -> Result<HashMap<String, Arc<DescriptorSetLayout>>> {
//     let mut layouts = HashMap::new();

//     for pipeline_info in shader_bufs_and_entries.pipelines.iter() {
//         let mut bindings = BTreeMap::new();

//         for buf_name in pipeline_info.buf_names.clone() {
//             let mut binding_desc =
//                 DescriptorSetLayoutBinding::descriptor_type(DescriptorType::StorageBuffer);
//             binding_desc.stages = ShaderStages::COMPUTE;
//             binding_desc.descriptor_count = 1;

//             let binding = global_buf_to_binding[&buf_name];

//             bindings.insert(binding, binding_desc);
//         }

//         let layout = DescriptorSetLayout::new(
//             device.clone(),
//             DescriptorSetLayoutCreateInfo {
//                 bindings,
//                 ..Default::default()
//             },
//         )?;
//         layouts.insert(pipeline_info.entry_point_name.clone(), layout);
//     }
//     Ok(layouts)
// }

pub fn build_concrete_descriptor_set(
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    layout: Arc<DescriptorSetLayout>,
    write_descriptor_sets: Vec<WriteDescriptorSet>,
) -> Result<Arc<DescriptorSet>> {
    let set = DescriptorSet::new(
        descriptor_set_allocator.clone(),
        layout,
        write_descriptor_sets,
        [],
    )?;
    Ok(set)
}
