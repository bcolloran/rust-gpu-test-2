use crate::error::Result;
use std::sync::Arc;

use vulkano::{
    descriptor_set::layout::DescriptorSetLayout,
    device::Device,
    pipeline::{
        compute::{ComputePipeline, ComputePipelineCreateInfo},
        layout::{PipelineLayout, PipelineLayoutCreateInfo},
        PipelineShaderStageCreateInfo,
    },
    shader::EntryPoint,
};

pub fn build_pipeline(
    device: Arc<Device>,
    descriptor_set_layout: Arc<DescriptorSetLayout>,
    entry_point: EntryPoint,
) -> Result<Arc<ComputePipeline>> {
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
