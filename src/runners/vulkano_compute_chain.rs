use crate::{
    error::CrateResult,
    runners::vulkano::{
        buffer_specs::{DescriptorSetByName, IntoDescriptorSetByName},
        device::compute_capable_device_and_queue,
        shader::shader_module,
        shader_pipeline_builder::ShaderPipelineSpec,
        typed_subbuffer_by_name::TypedSubbufferByName,
    },
};
use std::sync::Arc;

use vulkano::{
    buffer::Subbuffer,
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        PrimaryAutoCommandBuffer,
    },
    descriptor_set::allocator::StandardDescriptorSetAllocator,
    device::{Device, Queue},
    instance::Instance,
    memory::allocator::StandardMemoryAllocator,
    sync::{self, GpuFuture},
};

#[allow(unused)]
pub struct VulkanoComputeChain<BS>
where
    BS: IntoDescriptorSetByName<Out: 'static + DescriptorSetByName + TypedSubbufferByName>,
{
    instance: Arc<Instance>,
    device: Arc<Device>,
    queue: Arc<Queue>,
    shader_module: Arc<vulkano::shader::ShaderModule>,
    memory_allocator: Arc<StandardMemoryAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,

    gpu_buffer_specs: <BS as IntoDescriptorSetByName>::Out,
    pipeline_specs: Vec<ShaderPipelineSpec>,

    command_buffer: Arc<PrimaryAutoCommandBuffer>,
}

pub fn build_compute_pass_command_buffer<T: DescriptorSetByName>(
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    queue: Arc<Queue>,
    device: Arc<Device>,
    shader_module: Arc<vulkano::shader::ShaderModule>,
    gpu_buffer_specs: &T,
    pipeline_specs: &Vec<ShaderPipelineSpec>,
) -> CrateResult<Arc<PrimaryAutoCommandBuffer>> {
    let mut builder = AutoCommandBufferBuilder::primary(
        command_buffer_allocator.clone(),
        queue.queue_family_index(),
        CommandBufferUsage::MultipleSubmit,
    )?;

    for spec in pipeline_specs.iter() {
        let pipeline = spec
            .to_builder()
            .with_entry_point(shader_module.clone())?
            .with_descriptor_set_layout(device.clone())?
            .with_pipeline(device.clone())?
            .with_descriptor_set(gpu_buffer_specs, descriptor_set_allocator.clone())?;
        pipeline.bind_and_dispatch(&mut builder)?;
    }

    Ok(builder.build()?)
}

impl<BS> VulkanoComputeChain<BS>
where
    BS: IntoDescriptorSetByName<Out: 'static + DescriptorSetByName + TypedSubbufferByName>,
{
    pub fn typed_subbuffer_by_name<T: 'static>(&self, name: &str) -> CrateResult<Subbuffer<[T]>> {
        self.gpu_buffer_specs.subbuffer::<T>(name)
    }

    /// Create a new Vulkano runner
    pub fn new(buffer_specs: &BS, pipeline_specs: Vec<ShaderPipelineSpec>) -> CrateResult<Self> {
        let (instance, device_name, device, queue) = compute_capable_device_and_queue()?;
        println!("Using device: {}", device_name);

        let shader_module = shader_module(device.clone())?;
        println!("Shader module created");

        //   allocators
        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            device.clone(),
            Default::default(),
        ));

        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            Default::default(),
        ));

        println!("Allocators created");

        let buffer_specs_for_gpu = buffer_specs.with_gpu_buffer(memory_allocator.clone())?;

        pipeline_specs
            .iter()
            .try_for_each(|spec| spec.validate_against_buffer_specs(&buffer_specs_for_gpu))?;

        let command_buffer = build_compute_pass_command_buffer(
            command_buffer_allocator.clone(),
            descriptor_set_allocator.clone(),
            queue.clone(),
            device.clone(),
            shader_module.clone(),
            &buffer_specs_for_gpu,
            &pipeline_specs,
        )?;

        println!("VulkanoRunner::new ok");
        Ok(Self {
            instance,
            device,
            queue,
            memory_allocator,
            descriptor_set_allocator,
            command_buffer_allocator,
            shader_module,

            gpu_buffer_specs: buffer_specs_for_gpu,

            // buffer_specs: buffer_specs.clone(),
            pipeline_specs,
            command_buffer,
        })
    }

    pub fn execute(&self) -> CrateResult<()> {
        let future = sync::now(self.device.clone())
            .then_execute(self.queue.clone(), self.command_buffer.clone())?
            .then_signal_fence_and_flush()?;
        future.wait(None)?;

        Ok(())
    }

    /// Get the Vulkan instance (needed for creating windows/surfaces)
    pub fn instance(&self) -> &Arc<Instance> {
        &self.instance
    }

    /// Get the device (useful for creating graphics resources on the same device)
    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }

    /// Get the queue (useful for graphics rendering on the same queue)
    pub fn queue(&self) -> &Arc<Queue> {
        &self.queue
    }

    /// Get the memory allocator
    pub fn memory_allocator(&self) -> &Arc<StandardMemoryAllocator> {
        &self.memory_allocator
    }
}
