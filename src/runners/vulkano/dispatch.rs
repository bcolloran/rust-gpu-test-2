use std::sync::Arc;

use vulkano::{
    command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer},
    descriptor_set::DescriptorSet,
    pipeline::{ComputePipeline, Pipeline, PipelineBindPoint},
};

use crate::error::CrateResult;

/// Bind the descriptor set and dispatch the compute shader
/// This is basically like:
/// * providing arguments (the descriptor set)
/// * to a function/function pointer (the pipeline)
/// * then calling it (dispatch)
pub fn bind_and_dispatch(
    builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    pipeline: Arc<ComputePipeline>,
    descriptor_set: Arc<DescriptorSet>,
    num_wg: [u32; 3],
) -> CrateResult<()> {
    builder.bind_descriptor_sets(
        PipelineBindPoint::Compute,
        pipeline.layout().clone(),
        0,
        descriptor_set,
    )?;
    unsafe {
        builder.dispatch(num_wg)?;
    }
    Ok(())
}
