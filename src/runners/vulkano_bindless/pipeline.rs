use crate::error::CrateResult;
use std::sync::Arc;

use vulkano::{
    descriptor_set::layout::DescriptorSetLayout,
    device::Device,
    pipeline::{
        compute::{ComputePipeline, ComputePipelineCreateInfo},
        layout::{PipelineLayout, PipelineLayoutCreateInfo, PushConstantRange},
        PipelineShaderStageCreateInfo,
    },
    shader::{EntryPoint, ShaderStages},
};

/// Build a compute pipeline for the given entry point and descriptor set layout
/// A "pipeline" is for only one shader (not a sequence of shaders)
///
/// For bindless pipelines, we add a push constant range to pass buffer offsets to the shader.
/// Push constants are small amounts of data (up to 128 bytes) that can be updated very efficiently.
pub fn build_pipeline(
    device: Arc<Device>,
    descriptor_set_layout: Arc<DescriptorSetLayout>,
    entry_point: EntryPoint,
) -> CrateResult<Arc<ComputePipeline>> {
    // Pipeline layout + push constants
    // For bindless, we need to define a push constant range to pass buffer offsets
    // Our push constant struct is 2x u32 = 8 bytes
    let pipeline_layout = PipelineLayout::new(
        device.clone(),
        PipelineLayoutCreateInfo {
            set_layouts: vec![descriptor_set_layout],
            push_constant_ranges: vec![PushConstantRange {
                stages: ShaderStages::COMPUTE,
                offset: 0,
                size: 16, // 4 x u32 = 16 bytes (offset_0, offset_1, buffer_size, _padding)
            }],
            ..Default::default()
        },
    )
    .inspect_err(|e| println!("error in PipelineLayout::new: {e}"))?;

    // Build stage and compute pipeline
    let stage = PipelineShaderStageCreateInfo::new(entry_point.clone());
    let pipeline_info = ComputePipelineCreateInfo::stage_layout(stage, pipeline_layout.clone());

    let pipeline = ComputePipeline::new(device.clone(), None, pipeline_info)
        .inspect_err(|e| println!("error in ComputePipeline::new: {e}"))?;
    Ok(pipeline)
}
