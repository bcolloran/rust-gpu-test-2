use std::{collections::BTreeMap, sync::Arc};

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
    pipeline::ComputePipeline,
    shader::{EntryPoint, ShaderModule, ShaderStages},
};

use crate::{
    error::CrateResult,
    runners::vulkano::{
        buffer_specs::DescriptorSetByName, dispatch::bind_and_dispatch, pipeline::build_pipeline,
        shader::shader_entry_point,
    },
};

/// Builder for a shader pipeline.
/// Note that a Vulkan shader "pipeline" does not know the types of the buffers it will use. It only knows about the bindings. We only need actual buffer types when we create the descriptor sets Since a given pipeline can use multiple buffers that might be used by several pipelines, we track buffers by name. E
#[derive(Clone)]
pub struct ShaderPipelineBuilder<S> {
    spec: ShaderPipelineSpec,
    builder_state: S,
}

#[derive(Clone)]
struct ShaderPipelineSpec {
    invocation_name: String,
    entry_point_name: String,
    buf_names: Vec<String>,
    binding_nums_in_shader: Vec<u32>,
    num_workgroups: [u32; 3],
}

#[derive(Clone)]
pub struct InitialSpec {}

#[derive(Clone)]
pub struct HasEntryPoint {
    entry_point: EntryPoint,
}

#[derive(Clone)]
pub struct HasDescriptorSetLayout {
    entry_point: EntryPoint,
    descriptor_set_layout: Arc<DescriptorSetLayout>,
}

#[derive(Clone)]
pub struct HasPipeline {
    entry_point: EntryPoint,
    descriptor_set_layout: Arc<DescriptorSetLayout>,
    pipeline: Arc<ComputePipeline>,
}

#[derive(Clone)]
pub struct Ready {
    entry_point: EntryPoint,
    descriptor_set_layout: Arc<DescriptorSetLayout>,
    pipeline: Arc<ComputePipeline>,
    descriptor_set: Arc<DescriptorSet>,
}

// transition methods

impl ShaderPipelineBuilder<InitialSpec> {
    pub fn new(
        invocation_name: &str,
        entry_point_name: &str,
        buf_names: Vec<String>,
        binding_nums_in_shader: Vec<u32>,
        num_workgroups: [u32; 3],
    ) -> Self {
        Self {
            spec: ShaderPipelineSpec {
                invocation_name: invocation_name.to_string(),
                entry_point_name: entry_point_name.to_string(),
                buf_names,
                binding_nums_in_shader,
                num_workgroups,
            },
            builder_state: InitialSpec {},
        }
    }

    pub fn with_entry_point(
        self,
        shader_module: Arc<ShaderModule>,
    ) -> CrateResult<ShaderPipelineBuilder<HasEntryPoint>> {
        let entry_point = shader_entry_point(shader_module.clone(), &self.spec.entry_point_name)?;

        Ok(ShaderPipelineBuilder {
            spec: self.spec,
            builder_state: HasEntryPoint { entry_point },
        })
    }
}

impl ShaderPipelineBuilder<HasEntryPoint> {
    pub fn with_descriptor_set_layout(
        self,
        device: Arc<Device>,
    ) -> CrateResult<ShaderPipelineBuilder<HasDescriptorSetLayout>> {
        let mut bindings = BTreeMap::new();

        for shader_binding_num in self.spec.binding_nums_in_shader.clone().iter() {
            let mut binding_desc =
                DescriptorSetLayoutBinding::descriptor_type(DescriptorType::StorageBuffer);
            binding_desc.stages = ShaderStages::COMPUTE;
            binding_desc.descriptor_count = 1;

            bindings.insert(*shader_binding_num, binding_desc);
        }

        let descriptor_set_layout = DescriptorSetLayout::new(
            device.clone(),
            DescriptorSetLayoutCreateInfo {
                bindings,
                ..Default::default()
            },
        )?;

        Ok(ShaderPipelineBuilder {
            spec: self.spec,

            builder_state: HasDescriptorSetLayout {
                entry_point: self.builder_state.entry_point,
                descriptor_set_layout,
            },
        })
    }
}
impl ShaderPipelineBuilder<HasDescriptorSetLayout> {
    pub fn with_pipeline(
        self,
        device: Arc<Device>,
    ) -> CrateResult<ShaderPipelineBuilder<HasPipeline>> {
        let pipeline = build_pipeline(
            device,
            self.builder_state.descriptor_set_layout.clone(),
            self.builder_state.entry_point.clone(),
        )?;
        Ok(ShaderPipelineBuilder {
            spec: self.spec,

            builder_state: HasPipeline {
                entry_point: self.builder_state.entry_point,
                descriptor_set_layout: self.builder_state.descriptor_set_layout,
                pipeline,
            },
        })
    }
}

impl ShaderPipelineBuilder<HasPipeline> {
    pub fn with_descriptor_set<S: DescriptorSetByName>(
        self,
        buffer_specs: &S,
        descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    ) -> CrateResult<ShaderPipelineBuilder<Ready>> {
        let write_descriptor_sets: Vec<WriteDescriptorSet> = self
            .spec
            .buf_names
            .iter()
            .map(|name| buffer_specs.descriptor_set_by_name(name))
            .collect::<CrateResult<Vec<WriteDescriptorSet>>>()?;

        let descriptor_set =
            crate::runners::vulkano::descriptor_sets::build_concrete_descriptor_set(
                descriptor_set_allocator.clone(),
                self.builder_state.descriptor_set_layout.clone(),
                write_descriptor_sets.clone(),
            )?;

        Ok(ShaderPipelineBuilder {
            spec: self.spec,
            builder_state: Ready {
                entry_point: self.builder_state.entry_point,
                descriptor_set_layout: self.builder_state.descriptor_set_layout,
                pipeline: self.builder_state.pipeline,
                descriptor_set,
            },
        })
    }
}

impl ShaderPipelineBuilder<Ready> {
    pub fn bind_and_dispatch(
        &self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    ) -> CrateResult<()> {
        builder.bind_pipeline_compute(self.builder_state.pipeline.clone())?;

        bind_and_dispatch(
            builder,
            self.builder_state.pipeline.clone(),
            self.builder_state.descriptor_set.clone(),
            self.spec.num_workgroups,
        )?;

        Ok(())
    }
}
