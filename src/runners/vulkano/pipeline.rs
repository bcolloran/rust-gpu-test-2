use crate::error::Result;
use std::sync::Arc;

// Vulkano imports (version 0.35 API)
use std::collections::BTreeMap;
use vulkano::{
    descriptor_set::layout::{
        DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo,
        DescriptorType,
    },
    device::Device,
    pipeline::{
        compute::{ComputePipeline, ComputePipelineCreateInfo},
        layout::{PipelineLayout, PipelineLayoutCreateInfo},
        PipelineShaderStageCreateInfo,
    },
    shader::{EntryPoint, ShaderStages},
};

pub fn build_adder_pipeline(
    device: Arc<Device>,
    entry_point: EntryPoint,
) -> Result<Arc<ComputePipeline>> {
    // Descriptor set layout (binding 0: storage buffer)
    let mut binding0 = DescriptorSetLayoutBinding::descriptor_type(DescriptorType::StorageBuffer);
    binding0.stages = ShaderStages::COMPUTE;
    binding0.descriptor_count = 1;

    let mut binding1 = DescriptorSetLayoutBinding::descriptor_type(DescriptorType::StorageBuffer);
    binding1.stages = ShaderStages::COMPUTE;
    binding1.descriptor_count = 1;

    let mut bindings = BTreeMap::new();
    bindings.insert(0u32, binding0);
    bindings.insert(1u32, binding1);

    let descriptor_set_layout = DescriptorSetLayout::new(
        device.clone(),
        DescriptorSetLayoutCreateInfo {
            bindings,
            ..Default::default()
        },
    )?;

    // Pipeline layout + push constants
    let pipeline_layout = PipelineLayout::new(
        device.clone(),
        PipelineLayoutCreateInfo {
            set_layouts: vec![descriptor_set_layout],
            // push_constant_ranges: vec![PushConstantRange {
            //     stages: ShaderStages::COMPUTE,
            //     offset: 0,
            //     size: std::mem::size_of::<BitonicPushConstants>() as u32,
            // }],
            ..Default::default()
        },
    )?;

    // Build stage and compute pipeline
    let stage = PipelineShaderStageCreateInfo::new(entry_point.clone());
    let pipeline_info = ComputePipelineCreateInfo::stage_layout(stage, pipeline_layout.clone());
    let pipeline = ComputePipeline::new(device.clone(), None, pipeline_info)?;
    Ok(pipeline)
}
