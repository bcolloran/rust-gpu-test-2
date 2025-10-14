/// Shader buffer mapping for bindless compute pipelines
///
/// This module handles the creation and management of compute pipelines that use
/// the bindless paradigm. Unlike the traditional approach where each buffer gets
/// its own descriptor binding, here we use:
/// - 2 descriptor bindings total (one for all u32 data, one for all Vec2 data)
/// - Push constants to pass buffer offsets to shaders
/// - The same pipeline can be reused for different buffer combinations by just
///   changing push constants

use crate::{
    error::CrateResult,
    runners::vulkano::shader_buffer_mapping::ComputePassInvocationInfo,
    runners::vulkano_bindless::{
        descriptor_sets, pipeline::build_pipeline, shader::shader_entry_point,
        unified_buffer::UnifiedBufferTracker,
    },
};
use std::sync::Arc;
use vulkano::{
    command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer},
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator,
        layout::{
            DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo,
            DescriptorType,
        },
        DescriptorSet, WriteDescriptorSet,
    },
    device::Device,
    pipeline::{ComputePipeline, Pipeline},
    shader::ShaderStages,
};

/// Information about a single shader dispatch in the bindless paradigm
#[allow(dead_code)]
struct BindlessShaderDispatch {
    /// Name of this invocation (e.g., "adder_ab", "step_particles_0")
    invocation_name: String,
    /// Name of the shader entry point (e.g., "adder", "step_particles")
    entry_point_name: String,
    /// Names of the logical buffers this shader accesses (e.g., ["a", "b"] or ["x", "v"])
    buffer_names: Vec<String>,
    /// The compute pipeline for this shader
    pipeline: Arc<ComputePipeline>,
}

/// A complete bindless compute pass with all necessary pipelines and descriptor sets
pub struct BindlessComputePass {
    /// The compute pipelines for each shader invocation
    dispatches: Vec<BindlessShaderDispatch>,
    /// Descriptor set containing both unified buffers (binding 0 = u32, binding 1 = Vec2)
    descriptor_set: Arc<DescriptorSet>,
}

impl BindlessComputePass {
    /// Create a new bindless compute pass
    ///
    /// This function:
    /// 1. Creates a shader module with bindless entry points
    /// 2. Creates descriptor set layouts for the unified buffers
    /// 3. Creates compute pipelines for each shader
    /// 4. Creates descriptor sets that bind the unified buffers
    pub fn new(
        device: Arc<Device>,
        compute_pass_info: &ComputePassInvocationInfo,
        descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
        unified_buffers: &UnifiedBufferTracker,
    ) -> CrateResult<Self> {
        // Create shader module
        let shader_module =
            crate::runners::vulkano_bindless::shader::shader_module(device.clone())?;
        println!("Bindless shader module created");

        // Create descriptor set layouts for the two bindings
        // Binding 0: unified u32 buffer
        // Binding 1: unified Vec2 buffer
        let descriptor_set_layout = create_bindless_descriptor_set_layout(device.clone())?;
        println!("Bindless descriptor set layouts created");

        // Create a single descriptor set that binds both unified buffers
        // The descriptor set layout has 2 bindings:
        // - Binding 0: unified u32 buffer
        // - Binding 1: unified Vec2 buffer
        let descriptor_set = descriptor_sets::build_concrete_descriptor_set(
            descriptor_set_allocator.clone(),
            descriptor_set_layout.clone(),
            vec![
                WriteDescriptorSet::buffer(0, unified_buffers.unified_u32_buffer.clone()),
                WriteDescriptorSet::buffer(1, unified_buffers.unified_vec2_buffer.clone()),
            ],
        )?;

        // Create pipelines for each shader invocation
        let mut dispatches = Vec::new();

        for pipeline_info in &compute_pass_info.pipelines {
            // Get the shader entry point (e.g., "bindless::adder")
            let entry_point =
                shader_entry_point(shader_module.clone(), &pipeline_info.entry_point_name)?;

            // Build the compute pipeline
            let pipeline =
                build_pipeline(device.clone(), descriptor_set_layout.clone(), entry_point)?;

            dispatches.push(BindlessShaderDispatch {
                invocation_name: pipeline_info.invocation_name.clone(),
                entry_point_name: pipeline_info.entry_point_name.clone(),
                buffer_names: pipeline_info.buf_names.clone(),
                pipeline,
            });
        }

        println!(
            "Bindless pipelines created: {} dispatches",
            dispatches.len()
        );

        Ok(Self {
            dispatches,
            descriptor_set,
        })
    }

    /// Dispatch all compute shaders in the pass
    ///
    /// For each shader invocation:
    /// 1. Bind the appropriate pipeline
    /// 2. Bind descriptor sets (always the same ones - the unified buffers)
    /// 3. Set push constants with buffer offsets
    /// 4. Dispatch the compute shader
    pub fn dispatch_all(
        &self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        num_workgroups: u32,
        unified_buffers: &UnifiedBufferTracker,
    ) -> CrateResult<()> {
        use vulkano::pipeline::PipelineBindPoint;

        for dispatch in &self.dispatches {
            // Bind the pipeline for this shader
            builder.bind_pipeline_compute(dispatch.pipeline.clone())?;

            // Bind the descriptor set (same for all shaders - contains both unified buffers)
            // The shader will access the appropriate binding (0 for u32, 1 for Vec2) based on
            // what it was compiled to use
            builder.bind_descriptor_sets(
                PipelineBindPoint::Compute,
                dispatch.pipeline.layout().clone(),
                0,
                self.descriptor_set.clone(),
            )?;

            // Calculate push constants (buffer offsets) for this dispatch
            let push_constants = calculate_push_constants(dispatch, unified_buffers)?;

            // Set push constants
            builder.push_constants(dispatch.pipeline.layout().clone(), 0, push_constants)?;

            // Dispatch the compute shader
            unsafe {
                builder.dispatch([num_workgroups, 1, 1])?;
            }
        }

        Ok(())
    }
}

/// Create the descriptor set layout for bindless rendering
///
/// This layout has only 2 bindings:
/// - Binding 0: Storage buffer for all u32 data
/// - Binding 1: Storage buffer for all Vec2 data
fn create_bindless_descriptor_set_layout(
    device: Arc<Device>,
) -> CrateResult<Arc<DescriptorSetLayout>> {
    let mut bindings = std::collections::BTreeMap::new();

    // Binding 0: unified u32 buffer
    let mut binding_0 = DescriptorSetLayoutBinding::descriptor_type(DescriptorType::StorageBuffer);
    binding_0.stages = ShaderStages::COMPUTE;
    binding_0.descriptor_count = 1;
    bindings.insert(0, binding_0);

    // Binding 1: unified Vec2 buffer
    let mut binding_1 = DescriptorSetLayoutBinding::descriptor_type(DescriptorType::StorageBuffer);
    binding_1.stages = ShaderStages::COMPUTE;
    binding_1.descriptor_count = 1;
    bindings.insert(1, binding_1);

    Ok(DescriptorSetLayout::new(
        device,
        DescriptorSetLayoutCreateInfo {
            bindings,
            ..Default::default()
        },
    )?)
}

/// Push constants structure matching what the shaders expect
/// Must match the layout in shaders/src/bindless.rs
#[derive(Clone, Copy, vulkano::buffer::BufferContents)]
#[repr(C)]
struct PushConstants {
    offset_0: u32,
    offset_1: u32,
    buffer_size: u32,
    _padding: u32,
}

/// Calculate push constants for a shader dispatch
///
/// Push constants contain the offsets where each logical buffer starts within
/// the unified buffer. The format depends on which shader we're calling:
/// - adder: [a_offset, b_offset]
/// - step_particles: [x_offset, v_offset]
/// - wrap_particles: [x_offset, 0 (unused)]
fn calculate_push_constants(
    dispatch: &BindlessShaderDispatch,
    unified_buffers: &UnifiedBufferTracker,
) -> CrateResult<PushConstants> {
    // Determine the offsets based on which buffers this shader uses
    let offsets: Vec<u32> = dispatch
        .buffer_names
        .iter()
        .map(|name| {
            // Try u32 buffers first
            if let Some(&offset) = unified_buffers.u32_offsets.get(name) {
                return offset;
            }
            // Then try Vec2 buffers
            if let Some(&offset) = unified_buffers.vec2_offsets.get(name) {
                return offset;
            }
            // If buffer not found, panic (this is a programmer error)
            panic!(
                "Buffer '{}' not found in unified buffers for dispatch '{}'",
                name, dispatch.invocation_name
            );
        })
        .collect();

    // Create push constants struct
    // If there's only one offset (like wrap_particles), pad with 0
    let offset_0 = offsets.get(0).copied().unwrap_or(0);
    let offset_1 = offsets.get(1).copied().unwrap_or(0);



    Ok(PushConstants {
        offset_0,
        offset_1,
        buffer_size: unified_buffers.logical_buffer_size as u32,
        _padding: 0,
    })
}
