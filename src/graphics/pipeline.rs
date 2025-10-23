//! Graphics pipeline creation and management
//!
//! This module builds the graphics pipeline that renders Vec2 points as pixels.
//! The pipeline uses shaders compiled from rust-gpu (shaders/src/lib.rs).

use crate::error::CrateResult;
use glam::Vec2;
use shared::grid::GridCell;
use std::sync::Arc;
use vulkano::{
    buffer::Subbuffer,
    descriptor_set::{DescriptorSet, WriteDescriptorSet},
    pipeline::{
        graphics::{
            color_blend::{ColorBlendAttachmentState, ColorBlendState},
            input_assembly::{InputAssemblyState, PrimitiveTopology},
            multisample::MultisampleState,
            rasterization::{CullMode, RasterizationState},
            vertex_input::VertexInputState,
            viewport::{Viewport, ViewportState},
            GraphicsPipelineCreateInfo,
        },
        layout::PipelineDescriptorSetLayoutCreateInfo,
        GraphicsPipeline, Pipeline, PipelineLayout, PipelineShaderStageCreateInfo,
    },
    render_pass::{RenderPass, Subpass},
    shader::ShaderModule,
};

/// Create a graphics pipeline for rendering points from a buffer
///
/// This pipeline:
/// - Uses a vertex shader that reads Vec2 positions from a storage buffer
/// - Renders points as individual pixels (using PointList topology)
/// - Converts positions from [0, 1] range to Vulkan clip space [-1, 1]
/// - Colors points based on their index for visual variety
pub fn create_graphics_pipeline(
    device: Arc<vulkano::device::Device>,
    vs: Arc<ShaderModule>,
    fs: Arc<ShaderModule>,
    render_pass: Arc<RenderPass>,
    viewport: Viewport,
) -> CrateResult<Arc<GraphicsPipeline>> {
    // Get the entry points from the shader modules
    let vs = vs.entry_point("main_vs").ok_or(
        crate::graphics::error::GraphicsError::VertexShaderEntryPointNotFound(
            "main_vs".to_string(),
        ),
    )?;
    let fs = fs.entry_point("main_fs").ok_or(
        crate::graphics::error::GraphicsError::FragmentShaderEntryPointNotFound(
            "main_fs".to_string(),
        ),
    )?;

    // We're not using traditional vertex buffers - positions come from a storage buffer
    let vertex_input_state = VertexInputState::new();

    let stages = [
        PipelineShaderStageCreateInfo::new(vs),
        PipelineShaderStageCreateInfo::new(fs),
    ];

    // Create the pipeline layout (describes descriptor sets, push constants, etc.)
    let layout = PipelineLayout::new(
        device.clone(),
        PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
            .into_pipeline_layout_create_info(device.clone())?,
    )?;

    let subpass = Subpass::from(render_pass.clone(), 0)
        .ok_or(crate::graphics::error::GraphicsError::SubpassCreationFailed)?;

    // Build the graphics pipeline
    let pipeline = GraphicsPipeline::new(
        device.clone(),
        None,
        GraphicsPipelineCreateInfo {
            stages: stages.into_iter().collect(),
            vertex_input_state: Some(vertex_input_state),
            // Input assembly: render each vertex as an individual point
            input_assembly_state: Some(InputAssemblyState {
                topology: PrimitiveTopology::PointList,
                ..Default::default()
            }),
            // Viewport and scissor rect
            viewport_state: Some(ViewportState {
                viewports: [viewport].into_iter().collect(),
                ..Default::default()
            }),
            // Rasterization: convert primitives to fragments
            rasterization_state: Some(RasterizationState {
                cull_mode: CullMode::None,
                ..Default::default()
            }),
            // Multisampling: anti-aliasing (disabled for simplicity)
            multisample_state: Some(MultisampleState::default()),
            // Color blending: how fragment colors combine with existing framebuffer colors
            color_blend_state: Some(ColorBlendState::with_attachment_states(
                subpass.num_color_attachments(),
                ColorBlendAttachmentState::default(),
            )),
            subpass: Some(subpass.into()),
            ..GraphicsPipelineCreateInfo::layout(layout)
        },
    )?;

    Ok(pipeline)
}

/// Create a descriptor set that binds a Vec2 buffer to binding 0
///
/// This allows the vertex shader to read positions from the buffer.
pub fn create_descriptor_set(
    _device: Arc<vulkano::device::Device>,
    pipeline: &Arc<GraphicsPipeline>,
    position_buffer: Subbuffer<[Vec2]>,
    descriptor_set_allocator: &Arc<
        vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator,
    >,
) -> CrateResult<Arc<DescriptorSet>> {
    let layout = pipeline
        .layout()
        .set_layouts()
        .get(0)
        .ok_or(crate::graphics::error::GraphicsError::NoDescriptorSetLayout(0))?;

    let descriptor_set = DescriptorSet::new(
        descriptor_set_allocator.clone(),
        layout.clone(),
        [WriteDescriptorSet::buffer(0, position_buffer)],
        [],
    )?;

    Ok(descriptor_set)
}

/// Create a graphics pipeline for rendering the grid heatmap
///
/// This pipeline:
/// - Uses a vertex shader that reads GridCell data from a storage buffer
/// - Renders instanced quads (one instance per grid cell)
/// - Uses push constants to pass grid dimensions
/// - Converts mass values to grayscale colors
pub fn create_grid_pipeline(
    device: Arc<vulkano::device::Device>,
    vs: Arc<ShaderModule>,
    fs: Arc<ShaderModule>,
    render_pass: Arc<RenderPass>,
    viewport: Viewport,
) -> CrateResult<Arc<GraphicsPipeline>> {
    // Get the entry points from the shader modules
    let vs = vs.entry_point("grid_vs").ok_or(
        crate::graphics::error::GraphicsError::VertexShaderEntryPointNotFound(
            "grid_vs".to_string(),
        ),
    )?;
    let fs = fs.entry_point("grid_fs").ok_or(
        crate::graphics::error::GraphicsError::FragmentShaderEntryPointNotFound(
            "grid_fs".to_string(),
        ),
    )?;

    // We're not using traditional vertex buffers - positions are generated in the shader
    let vertex_input_state = VertexInputState::new();

    let stages = [
        PipelineShaderStageCreateInfo::new(vs),
        PipelineShaderStageCreateInfo::new(fs),
    ];

    // Create the pipeline layout (describes descriptor sets, push constants, etc.)
    let layout = PipelineLayout::new(
        device.clone(),
        PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
            .into_pipeline_layout_create_info(device.clone())?,
    )?;

    let subpass = Subpass::from(render_pass.clone(), 0)
        .ok_or(crate::graphics::error::GraphicsError::SubpassCreationFailed)?;

    // Build the graphics pipeline
    let pipeline = GraphicsPipeline::new(
        device.clone(),
        None,
        GraphicsPipelineCreateInfo {
            stages: stages.into_iter().collect(),
            vertex_input_state: Some(vertex_input_state),
            // Input assembly: render triangles for quads (6 vertices per instance)
            input_assembly_state: Some(InputAssemblyState {
                topology: PrimitiveTopology::TriangleList,
                ..Default::default()
            }),
            // Viewport and scissor rect
            viewport_state: Some(ViewportState {
                viewports: [viewport].into_iter().collect(),
                ..Default::default()
            }),
            // Rasterization: convert primitives to fragments
            rasterization_state: Some(RasterizationState {
                cull_mode: CullMode::None,
                ..Default::default()
            }),
            // Multisampling: anti-aliasing (disabled for simplicity)
            multisample_state: Some(MultisampleState::default()),
            // Color blending: how fragment colors combine with existing framebuffer colors
            color_blend_state: Some(ColorBlendState::with_attachment_states(
                subpass.num_color_attachments(),
                ColorBlendAttachmentState::default(),
            )),
            subpass: Some(subpass.into()),
            ..GraphicsPipelineCreateInfo::layout(layout)
        },
    )?;

    Ok(pipeline)
}

/// Create a descriptor set for the grid pipeline that binds a GridCell buffer to binding 1
///
/// This allows the vertex shader to read grid cell data from the buffer.
pub fn create_grid_descriptor_set(
    _device: Arc<vulkano::device::Device>,
    pipeline: &Arc<GraphicsPipeline>,
    grid_buffer: Subbuffer<[GridCell]>,
    descriptor_set_allocator: &Arc<
        vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator,
    >,
) -> CrateResult<Arc<DescriptorSet>> {
    let layout = pipeline
        .layout()
        .set_layouts()
        .get(0)
        .ok_or(crate::graphics::error::GraphicsError::NoDescriptorSetLayout(0))?;

    let descriptor_set = DescriptorSet::new(
        descriptor_set_allocator.clone(),
        layout.clone(),
        [WriteDescriptorSet::buffer(1, grid_buffer)],
        [],
    )?;

    Ok(descriptor_set)
}
