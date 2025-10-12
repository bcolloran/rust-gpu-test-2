use std::{
    collections::{BTreeMap, HashSet},
    sync::Arc,
};

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
        buffer::BufNameToBufferAny, dispatch::bind_and_dispatch, pipeline::build_pipeline,
        shader::shader_entry_point,
    },
};

#[derive(Clone)]
pub struct ShaderInvocationTemplate {
    pub invocation_name: String,
    pub entry_point_name: String,
    pub buf_names: Vec<String>,
    pub binding_nums_in_shader: Vec<u32>,
}

impl ShaderInvocationTemplate {
    pub fn with_entry_point(&self, entry_point: EntryPoint) -> ShaderPipelineHasEntry {
        ShaderPipelineHasEntry {
            invocation_name: self.invocation_name.clone(),
            entry_point_name: self.entry_point_name.clone(),
            buf_names: self.buf_names.clone(),
            binding_nums_in_shader: self.binding_nums_in_shader.clone(),
            entry_point,
        }
    }
}

#[derive(Clone)]
pub struct ShaderPipelineHasEntry {
    pub invocation_name: String,
    pub entry_point_name: String,
    pub buf_names: Vec<String>,
    pub binding_nums_in_shader: Vec<u32>,
    pub entry_point: EntryPoint,
}

impl ShaderPipelineHasEntry {
    pub fn with_descriptor_set_layout(
        &self,
        layout: Arc<DescriptorSetLayout>,
    ) -> ShaderPipelineHasDescriptorSetLayout {
        ShaderPipelineHasDescriptorSetLayout {
            invocation_name: self.invocation_name.clone(),
            entry_point_name: self.entry_point_name.clone(),
            buf_names: self.buf_names.clone(),
            binding_nums_in_shader: self.binding_nums_in_shader.clone(),
            entry_point: self.entry_point.clone(),
            descriptor_set_layout: layout.clone(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct ShaderPipelineHasDescriptorSetLayout {
    pub invocation_name: String,
    pub entry_point_name: String,
    pub buf_names: Vec<String>,
    pub binding_nums_in_shader: Vec<u32>,
    pub entry_point: EntryPoint,
    pub descriptor_set_layout: Arc<DescriptorSetLayout>,
}

impl ShaderPipelineHasDescriptorSetLayout {
    pub fn with_pipeline(&self, pipeline: Arc<ComputePipeline>) -> ShaderPipelineHasPipeline {
        ShaderPipelineHasPipeline {
            invocation_name: self.invocation_name.clone(),
            entry_point_name: self.entry_point_name.clone(),
            buf_names: self.buf_names.clone(),
            binding_nums_in_shader: self.binding_nums_in_shader.clone(),
            entry_point: self.entry_point.clone(),
            descriptor_set_layout: self.descriptor_set_layout.clone(),
            pipeline,
        }
    }
}

#[derive(Clone)]
pub struct ShaderPipelineHasPipeline {
    pub invocation_name: String,
    pub entry_point_name: String,
    pub buf_names: Vec<String>,
    pub binding_nums_in_shader: Vec<u32>,
    pub entry_point: EntryPoint,
    pub descriptor_set_layout: Arc<DescriptorSetLayout>,
    pub pipeline: Arc<ComputePipeline>,
}

impl ShaderPipelineHasPipeline {
    pub fn with_descriptor_set(
        &self,
        descriptor_set: Arc<DescriptorSet>,
    ) -> ShaderPipelineHasConcreteDescriptorSet {
        ShaderPipelineHasConcreteDescriptorSet {
            invocation_name: self.invocation_name.clone(),
            entry_point_name: self.entry_point_name.clone(),
            buf_names: self.buf_names.clone(),
            binding_nums_in_shader: self.binding_nums_in_shader.clone(),
            entry_point: self.entry_point.clone(),
            descriptor_set_layout: self.descriptor_set_layout.clone(),
            pipeline: self.pipeline.clone(),
            descriptor_set,
        }
    }
}

#[derive(Clone)]
pub struct ShaderPipelineHasConcreteDescriptorSet {
    pub invocation_name: String,
    pub entry_point_name: String,
    pub buf_names: Vec<String>,
    pub binding_nums_in_shader: Vec<u32>,
    pub entry_point: EntryPoint,
    pub descriptor_set_layout: Arc<DescriptorSetLayout>,
    pub pipeline: Arc<ComputePipeline>,
    pub descriptor_set: Arc<DescriptorSet>,
}

//
//
//
//
//
//
//
//
//
//

#[derive(Clone)]
pub struct ComputePassInvocationInfo {
    pub pipelines: Vec<ShaderInvocationTemplate>,
}

impl ComputePassInvocationInfo {
    pub fn from_lists(lists: Vec<(&str, Vec<&str>, (&str, Vec<u32>))>) -> Self {
        let mut invocations = HashSet::new();

        ComputePassInvocationInfo {
            pipelines: lists
                .iter()
                .map(
                    |(invoke_name, list, (entry_name, binding_nums_in_shader))| {
                        if !invocations.insert(invoke_name.to_string()) {
                            panic!(
                                "Duplicate invocation name in shader buffer mapping: {invoke_name}"
                            );
                        };

                        ShaderInvocationTemplate {
                            invocation_name: invoke_name.to_string(),
                            entry_point_name: entry_name.to_string(),
                            buf_names: list.iter().map(|s| s.to_string()).collect(),
                            binding_nums_in_shader: binding_nums_in_shader.clone(),
                        }
                    },
                )
                .collect(),
        }
    }

    pub fn with_entry_points(
        &self,
        shader_module: Arc<ShaderModule>,
    ) -> ShaderPipelineInfosWithEntry {
        ShaderPipelineInfosWithEntry {
            pipelines: self
                .pipelines
                .iter()
                .map(|template| {
                    let entry_point =
                        shader_entry_point(shader_module.clone(), &template.entry_point_name)
                            .unwrap_or_else(|e| {
                                panic!(
                                    "Failed to get entry point '{}': {e}",
                                    template.entry_point_name
                                );
                            });
                    template.with_entry_point(entry_point)
                })
                .collect(),
        }
    }
}

#[derive(Clone)]
pub struct ShaderPipelineInfosWithEntry {
    pub pipelines: Vec<ShaderPipelineHasEntry>,
}
impl ShaderPipelineInfosWithEntry {
    pub fn with_descriptor_sets(
        &self,
        device: Arc<Device>,
        // global_buf_to_binding: BufNameToBinding,
    ) -> CrateResult<ShaderPipelineInfosWithDescriptorSetLayouts> {
        let mut pipelines = Vec::new();

        for pipeline_info in self.pipelines.iter() {
            println!(
                "Processing pipeline invocation '{}', entry point '{}'",
                pipeline_info.invocation_name, pipeline_info.entry_point_name
            );
            let mut bindings = BTreeMap::new();

            for (buf_name, shader_binding_num) in pipeline_info
                .buf_names
                .clone()
                .iter()
                .zip(pipeline_info.binding_nums_in_shader.clone().iter())
            {
                let mut binding_desc =
                    DescriptorSetLayoutBinding::descriptor_type(DescriptorType::StorageBuffer);
                binding_desc.stages = ShaderStages::COMPUTE;
                binding_desc.descriptor_count = 1;

                // let binding = global_buf_to_binding[&buf_name];
                println!(
                    "   Buffer name '{}' mapped to binding {}",
                    buf_name, shader_binding_num
                );
                bindings.insert(*shader_binding_num, binding_desc);
            }

            let layout = DescriptorSetLayout::new(
                device.clone(),
                DescriptorSetLayoutCreateInfo {
                    bindings,
                    ..Default::default()
                },
            )?;
            pipelines.push(pipeline_info.with_descriptor_set_layout(layout.clone()));
        }

        Ok(ShaderPipelineInfosWithDescriptorSetLayouts { pipelines })
    }
}

#[derive(Clone)]
pub struct ShaderPipelineInfosWithDescriptorSetLayouts {
    pub pipelines: Vec<ShaderPipelineHasDescriptorSetLayout>,
}
impl ShaderPipelineInfosWithDescriptorSetLayouts {
    pub fn with_pipelines(
        &self,
        device: Arc<Device>,
    ) -> CrateResult<ShaderPipelineInfosWithComputePipelines> {
        let new_pipelines = self
            .pipelines
            .iter()
            .map(|pipeline_info| {
                let pipeline = build_pipeline(
                    device.clone(),
                    pipeline_info.descriptor_set_layout.clone(),
                    pipeline_info.entry_point.clone(),
                )
                .inspect_err(|e| {
                    println!(
                        "Error during build_pipeline for entry point {:?}: {e}",
                        pipeline_info
                    );
                })?;
                Ok(pipeline_info.with_pipeline(pipeline))
            })
            .collect::<CrateResult<Vec<_>>>()?;
        Ok(ShaderPipelineInfosWithComputePipelines {
            pipelines: new_pipelines,
        })
    }
}

#[derive(Clone)]
pub struct ShaderPipelineInfosWithComputePipelines {
    pub pipelines: Vec<ShaderPipelineHasPipeline>,
}

impl ShaderPipelineInfosWithComputePipelines {
    pub fn with_descriptor_sets(
        &self,
        descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
        // write_descriptor_sets_map: &HashMap<String, Vec<WriteDescriptorSet>>,
        buf_any_map: &BufNameToBufferAny,
    ) -> CrateResult<ShaderPipelineInfosWithDescriptorSets> {
        let new_pipelines = self
            .pipelines
            .iter()
            .map(|pipeline_info| {
                // let write_descriptor_sets =
                //     &write_descriptor_sets_map[&pipeline_info.invocation_name];

                let write_descriptor_sets: Vec<WriteDescriptorSet> = pipeline_info
                    .buf_names
                    .iter()
                    .zip(pipeline_info.binding_nums_in_shader.iter())
                    .map(|(buf_name, binding_num)| {
                        if !buf_any_map.0.contains_key(buf_name) {
                            panic!("Buffer name '{}' not found in provided buffers", buf_name);
                        }
                        let buf_any = &buf_any_map.0[buf_name];

                        buf_any.write_descriptor_set_for_binding(*binding_num)
                    })
                    .collect();

                let descriptor_set =
                    crate::runners::vulkano::descriptor_sets::build_concrete_descriptor_set(
                        descriptor_set_allocator.clone(),
                        pipeline_info.descriptor_set_layout.clone(),
                        write_descriptor_sets.clone(),
                    )?;

                Ok(pipeline_info.with_descriptor_set(descriptor_set))
            })
            .collect::<CrateResult<Vec<_>>>()?;
        Ok(ShaderPipelineInfosWithDescriptorSets {
            pipelines: new_pipelines,
        })
    }
}

#[derive(Clone)]
pub struct ShaderPipelineInfosWithDescriptorSets {
    pub pipelines: Vec<ShaderPipelineHasConcreteDescriptorSet>,
}

impl ShaderPipelineInfosWithDescriptorSets {
    pub fn bind_and_dispatch_all(
        &self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        num_wg: u32,
    ) -> CrateResult<()> {
        for pipeline_info in self.pipelines.iter() {
            builder.bind_pipeline_compute(pipeline_info.pipeline.clone())?;
            bind_and_dispatch(
                builder,
                pipeline_info.pipeline.clone(),
                pipeline_info.descriptor_set.clone(),
                num_wg,
            )?;
        }
        Ok(())
    }
}
