use std::sync::Arc;

use vulkano::descriptor_set::{
    allocator::StandardDescriptorSetAllocator, layout::DescriptorSetLayout, DescriptorSet,
    WriteDescriptorSet,
};

use crate::error::CrateResult;

pub fn build_concrete_descriptor_set(
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    layout: Arc<DescriptorSetLayout>,
    write_descriptor_sets: Vec<WriteDescriptorSet>,
) -> CrateResult<Arc<DescriptorSet>> {
    let set = DescriptorSet::new(
        descriptor_set_allocator.clone(),
        layout,
        write_descriptor_sets,
        [],
    )?;
    Ok(set)
}
