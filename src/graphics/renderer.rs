//! Main graphics renderer that manages the render loop
//!
//! This module ties everything together:
//! - Window and event loop management
//! - Swapchain for presenting rendered images
//! - Command buffers for recording draw commands
//! - Synchronization for frame pacing

use crate::{
    error::{ChimeraError, CrateResult},
    graphics::{
        device::select_physical_device,
        pipeline::{create_descriptor_set, create_graphics_pipeline},
    },
};
use glam::Vec2;
use std::sync::Arc;
use vulkano::{
    buffer::Subbuffer,
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        PrimaryAutoCommandBuffer, RenderPassBeginInfo, SubpassBeginInfo, SubpassContents,
    },
    descriptor_set::allocator::StandardDescriptorSetAllocator,
    device::{Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo},
    image::{view::ImageView, Image, ImageUsage},
    instance::Instance,
    pipeline::{graphics::viewport::Viewport, Pipeline},
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass},
    shader::ShaderModule,
    swapchain::{self, Surface, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo},
    sync::{self, GpuFuture},
    Validated, VulkanError,
};

/// Main graphics renderer structure
///
/// This manages all the Vulkan resources needed for rendering:
/// - Device and queue for GPU operations
/// - Swapchain for displaying images on screen
/// - Render pass defining how we draw
/// - Graphics pipeline with our shaders
/// - Command buffers with recorded draw commands
/// - Synchronization primitives (fences) for frame pacing
pub struct GraphicsRenderer {
    device: Arc<Device>,
    queue: Arc<Queue>,
    swapchain: Arc<Swapchain>,
    render_pass: Arc<RenderPass>,
    framebuffers: Vec<Arc<Framebuffer>>,
    pipeline: Arc<vulkano::pipeline::GraphicsPipeline>,
    command_buffers: Vec<Arc<PrimaryAutoCommandBuffer>>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,

    // Flags for handling window resize and swapchain recreation
    window_resized: bool,
    recreate_swapchain: bool,

    // Shader modules (needed for pipeline recreation on resize)
    vs: Arc<ShaderModule>,
    fs: Arc<ShaderModule>,
    viewport: Viewport,

    // Buffer containing particle positions and number of particles to render
    position_buffer: Option<Subbuffer<[Vec2]>>,
    num_particles: usize,
}

impl GraphicsRenderer {
    /// Create a new graphics renderer from an existing device and queue
    ///
    /// This version shares the device with the compute pipeline to enable
    /// zero-copy buffer sharing between compute and graphics.
    /// The device must have been created with KHR_swapchain extension enabled.
    /// The queue must support both graphics operations and presentation to the surface.
    pub fn from_device(
        device: Arc<Device>,
        queue: Arc<Queue>,
        surface: Arc<Surface>,
        vs: Arc<ShaderModule>,
        fs: Arc<ShaderModule>,
    ) -> CrateResult<Self> {
        let physical_device = device.physical_device().clone();

        println!(
            "Using shared device: {} (type: {:?})",
            physical_device.properties().device_name,
            physical_device.properties().device_type
        );

        // Verify the queue supports presentation to this surface
        let queue_family_index = queue.queue_family_index();
        if !physical_device.surface_support(queue_family_index, &surface)? {
            return Err(ChimeraError::Other(format!(
                "Queue family {} does not support presentation to the surface",
                queue_family_index
            )));
        }

        // Get surface capabilities to create swapchain
        let caps = physical_device.surface_capabilities(&surface, Default::default())?;

        // Continue with swapchain creation
        Self::init_with_device_and_queue(device, queue, physical_device, surface, caps, vs, fs)
    }

    /// Create a new graphics renderer
    ///
    /// This sets up all the Vulkan resources needed for rendering.
    /// It requires pre-compiled SPIR-V shader modules for vertex and fragment shaders.
    pub fn new(
        instance: Arc<Instance>,
        surface: Arc<Surface>,
        vs: Arc<ShaderModule>,
        fs: Arc<ShaderModule>,
    ) -> CrateResult<Self> {
        // Find a suitable GPU and queue family
        let device_extensions = DeviceExtensions {
            khr_swapchain: true, // Required for presenting to screen
            ..DeviceExtensions::empty()
        };

        let (physical_device, queue_family_index) =
            select_physical_device(&instance, &surface, &device_extensions)?;

        println!(
            "Selected device: {} (type: {:?})",
            physical_device.properties().device_name,
            physical_device.properties().device_type
        );

        // Create logical device and queue
        let (device, mut queues) = Device::new(
            physical_device.clone(),
            DeviceCreateInfo {
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                enabled_extensions: device_extensions,
                ..Default::default()
            },
        )?;

        let queue = queues
            .next()
            .ok_or_else(|| ChimeraError::Other("No queue available".to_string()))?;

        // Get surface capabilities to create swapchain
        let caps = physical_device.surface_capabilities(&surface, Default::default())?;

        Self::init_with_device_and_queue(device, queue, physical_device, surface, caps, vs, fs)
    }

    /// Common initialization logic for both constructors
    fn init_with_device_and_queue(
        device: Arc<Device>,
        queue: Arc<Queue>,
        physical_device: Arc<vulkano::device::physical::PhysicalDevice>,
        surface: Arc<Surface>,
        caps: vulkano::swapchain::SurfaceCapabilities,
        vs: Arc<ShaderModule>,
        fs: Arc<ShaderModule>,
    ) -> CrateResult<Self> {
        // Choose swapchain format (color format for the images)
        let image_format = physical_device
            .surface_formats(&*surface, Default::default())?
            .first()
            .ok_or_else(|| ChimeraError::Other("No surface formats available".to_string()))?
            .0;

        // Choose composite alpha (how the window is blended with desktop)
        let composite_alpha = caps
            .supported_composite_alpha
            .into_iter()
            .next()
            .ok_or_else(|| ChimeraError::Other("No composite alpha modes available".to_string()))?;

        // Create the swapchain (a set of images we can render to and present)
        let (swapchain, images) = Swapchain::new(
            device.clone(),
            surface,
            SwapchainCreateInfo {
                min_image_count: caps.min_image_count + 1, // Request one more than minimum
                image_format,
                image_extent: caps.max_image_extent, // Use max supported size initially
                image_usage: ImageUsage::COLOR_ATTACHMENT, // We'll render to these images
                composite_alpha,
                ..Default::default()
            },
        )?;

        // Create a render pass - defines the structure of our rendering
        // This specifies we have one color attachment that we clear and store
        let render_pass = vulkano::single_pass_renderpass!(
            device.clone(),
            attachments: {
                color: {
                    format: image_format,
                    samples: 1,
                    load_op: Clear,
                    store_op: Store,
                },
            },
            pass: {
                color: [color],
                depth_stencil: {},
            },
        )?;

        // Create framebuffers - one for each swapchain image
        // A framebuffer connects the render pass attachments to actual images
        let framebuffers = create_framebuffers(&images, render_pass.clone());

        // Create viewport matching the swapchain dimensions
        let viewport = Viewport {
            offset: [0.0, 0.0],
            extent: [
                caps.max_image_extent[0] as f32,
                caps.max_image_extent[1] as f32,
            ],
            depth_range: 0.0..=1.0,
        };

        // Create the graphics pipeline with our shaders
        let pipeline = create_graphics_pipeline(
            device.clone(),
            vs.clone(),
            fs.clone(),
            render_pass.clone(),
            viewport.clone(),
        )?;

        // Create command buffer allocator
        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            Default::default(),
        ));

        // Create descriptor set allocator
        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            device.clone(),
            Default::default(),
        ));

        // Initialize with empty command buffers (will be created when position buffer is set)
        let command_buffers = vec![];

        Ok(Self {
            device,
            queue,
            swapchain,
            render_pass,
            framebuffers,
            pipeline,
            command_buffers,
            command_buffer_allocator,
            descriptor_set_allocator,
            window_resized: false,
            recreate_swapchain: false,
            vs,
            fs,
            viewport,
            position_buffer: None,
            num_particles: 0,
        })
    }

    /// Set the position buffer to render
    ///
    /// This updates the command buffers to render the particles from the given buffer.
    pub fn set_position_buffer(
        &mut self,
        position_buffer: Subbuffer<[Vec2]>,
        num_particles: usize,
    ) -> CrateResult<()> {
        self.position_buffer = Some(position_buffer.clone());
        self.num_particles = num_particles;

        // Recreate command buffers with the new buffer
        if !self.framebuffers.is_empty() {
            self.command_buffers = create_command_buffers_with_buffer(
                &self.command_buffer_allocator,
                &self.descriptor_set_allocator,
                &self.queue,
                &self.pipeline,
                &self.framebuffers,
                &self.render_pass,
                position_buffer,
                num_particles,
            )?;
        }

        Ok(())
    }

    /// Handle a single frame of rendering
    ///
    /// This method:
    /// 1. Acquires the next swapchain image
    /// 2. Waits for any previous rendering to that image to complete
    /// 3. Executes the command buffer to render
    /// 4. Presents the image to the screen
    pub fn render_frame(&mut self) -> CrateResult<()> {
        // Acquire the next image from the swapchain
        let (image_i, suboptimal, acquire_future) =
            match swapchain::acquire_next_image(self.swapchain.clone(), None)
                .map_err(Validated::unwrap)
            {
                Ok(r) => r,
                Err(VulkanError::OutOfDate) => {
                    // Swapchain is out of date (e.g., window resized), need to recreate
                    self.recreate_swapchain = true;
                    return Ok(());
                }
                Err(e) => return Err(e.into()),
            };

        if suboptimal {
            // Swapchain is suboptimal but still usable, recreate on next frame
            self.recreate_swapchain = true;
        }

        // Simple approach: just execute and wait
        // For better performance, you'd want to track fences per image
        let future = sync::now(self.device.clone())
            .join(acquire_future)
            .then_execute(
                self.queue.clone(),
                self.command_buffers[image_i as usize].clone(),
            )?
            .then_swapchain_present(
                self.queue.clone(),
                SwapchainPresentInfo::swapchain_image_index(self.swapchain.clone(), image_i),
            )
            .then_signal_fence_and_flush();

        match future.map_err(Validated::unwrap) {
            Ok(future) => {
                // Wait for the operation to complete
                // In a real app, you'd want async handling here
                future.wait(None)?;
            }
            Err(VulkanError::OutOfDate) => {
                self.recreate_swapchain = true;
            }
            Err(e) => {
                println!("Failed to flush future: {e}");
            }
        }

        Ok(())
    }

    /// Recreate the swapchain (needed after window resize)
    ///
    /// This recreates:
    /// - The swapchain with new dimensions
    /// - Framebuffers for the new swapchain images
    /// - The pipeline with updated viewport
    /// - Command buffers with the new pipeline and framebuffers
    pub fn recreate_swapchain(&mut self, new_dimensions: [u32; 2]) -> CrateResult<()> {
        // Recreate swapchain with new dimensions
        let (new_swapchain, new_images) = self.swapchain.recreate(SwapchainCreateInfo {
            image_extent: new_dimensions,
            ..self.swapchain.create_info()
        })?;

        self.swapchain = new_swapchain;

        // Recreate framebuffers for new images
        self.framebuffers = create_framebuffers(&new_images, self.render_pass.clone());

        // Update viewport for new dimensions
        self.viewport.extent = [new_dimensions[0] as f32, new_dimensions[1] as f32];

        // Recreate pipeline with new viewport
        self.pipeline = create_graphics_pipeline(
            self.device.clone(),
            self.vs.clone(),
            self.fs.clone(),
            self.render_pass.clone(),
            self.viewport.clone(),
        )?;

        // Recreate command buffers with the position buffer if it's set
        if let Some(position_buffer) = &self.position_buffer {
            self.command_buffers = create_command_buffers_with_buffer(
                &self.command_buffer_allocator,
                &self.descriptor_set_allocator,
                &self.queue,
                &self.pipeline,
                &self.framebuffers,
                &self.render_pass,
                position_buffer.clone(),
                self.num_particles,
            )?;
        }

        Ok(())
    }

    /// Check if swapchain needs recreation and handle it
    pub fn handle_resize(&mut self, new_dimensions: [u32; 2]) -> CrateResult<()> {
        if self.window_resized || self.recreate_swapchain {
            self.recreate_swapchain = false;
            self.recreate_swapchain(new_dimensions)?;
            self.window_resized = false;
        }
        Ok(())
    }

    /// Mark that the window was resized
    pub fn set_window_resized(&mut self) {
        self.window_resized = true;
    }
}

/// Helper function to create framebuffers from swapchain images
fn create_framebuffers(
    images: &[Arc<Image>],
    render_pass: Arc<RenderPass>,
) -> Vec<Arc<Framebuffer>> {
    images
        .iter()
        .map(|image| {
            let view = ImageView::new_default(image.clone()).unwrap();
            Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![view],
                    ..Default::default()
                },
            )
            .unwrap()
        })
        .collect::<Vec<_>>()
}

/// Helper function to create command buffers for rendering points from a buffer
///
/// Each command buffer:
/// 1. Begins a render pass with a clear color (dark blue)
/// 2. Binds the graphics pipeline
/// 3. Binds the descriptor set containing the position buffer
/// 4. Draws N vertices (one for each particle), reading positions from the buffer
/// 5. Ends the render pass
fn create_command_buffers_with_buffer(
    allocator: &Arc<StandardCommandBufferAllocator>,
    descriptor_set_allocator: &Arc<StandardDescriptorSetAllocator>,
    queue: &Arc<Queue>,
    pipeline: &Arc<vulkano::pipeline::GraphicsPipeline>,
    framebuffers: &[Arc<Framebuffer>],
    _render_pass: &Arc<RenderPass>, // Kept for consistency but not used
    position_buffer: Subbuffer<[Vec2]>,
    num_particles: usize,
) -> CrateResult<Vec<Arc<PrimaryAutoCommandBuffer>>> {
    // Create descriptor set that binds the position buffer
    let descriptor_set = create_descriptor_set(
        pipeline.device().clone(),
        pipeline,
        position_buffer,
        descriptor_set_allocator,
    )?;

    framebuffers
        .iter()
        .map(|framebuffer| {
            let mut builder = AutoCommandBufferBuilder::primary(
                allocator.clone(),
                queue.queue_family_index(),
                CommandBufferUsage::MultipleSubmit,
            )?;

            unsafe {
                builder
                    // Begin render pass, clearing to dark blue/black
                    .begin_render_pass(
                        RenderPassBeginInfo {
                            clear_values: vec![Some([0.0, 0.0, 0.05, 1.0].into())],
                            ..RenderPassBeginInfo::framebuffer(framebuffer.clone())
                        },
                        SubpassBeginInfo {
                            contents: SubpassContents::Inline,
                            ..Default::default()
                        },
                    )?
                    // Bind our graphics pipeline
                    .bind_pipeline_graphics(pipeline.clone())?
                    // Bind the descriptor set with the position buffer
                    .bind_descriptor_sets(
                        vulkano::pipeline::PipelineBindPoint::Graphics,
                        pipeline.layout().clone(),
                        0,
                        descriptor_set.clone(),
                    )?
                    // Draw N points, vertex shader will read positions from buffer using vertex_index
                    .draw(num_particles as u32, 1, 0, 0)?
                    .end_render_pass(Default::default())?;
            }

            Ok(builder.build()?)
        })
        .collect()
}
