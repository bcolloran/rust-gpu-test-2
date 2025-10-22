use std::sync::Arc;

use variadics_please::all_tuples_enumerated;
use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    descriptor_set::WriteDescriptorSet,
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
};

use crate::error::CrateResult;

pub struct DataAndBindingSpec<'a, T> {
    pub name: &'static str,
    pub binding: u32,
    pub data: &'a mut [T],
}

pub fn buf_spec<'a, T>(
    name: &'static str,
    binding: u32,
    data: &'a mut [T],
) -> DataAndBindingSpec<'a, T> {
    DataAndBindingSpec {
        name,
        binding,
        data,
    }
}

pub struct SubbufferAndBindingSpec<T> {
    pub name: &'static str,
    pub binding: u32,
    pub sub_buf: Subbuffer<[T]>,
}

pub trait IntoDescriptorSetByName {
    type Out: DescriptorSetByName;
    fn with_gpu_buffer(
        &self,
        memory_allocator: Arc<StandardMemoryAllocator>,
    ) -> CrateResult<Self::Out>;
}

impl<'a, T> IntoDescriptorSetByName for DataAndBindingSpec<'a, T>
where
    T: Copy + BufferContents,
    SubbufferAndBindingSpec<T>: DescriptorSetByName,
{
    type Out = SubbufferAndBindingSpec<T>;
    fn with_gpu_buffer(
        &self,
        memory_allocator: Arc<StandardMemoryAllocator>,
    ) -> CrateResult<SubbufferAndBindingSpec<T>> {
        let usage =
            BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC | BufferUsage::TRANSFER_DST;

        let sub_buf: Subbuffer<[T]> = Buffer::from_iter(
            memory_allocator,
            BufferCreateInfo {
                usage,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            self.data.iter().copied(),
        )?;

        Ok(SubbufferAndBindingSpec {
            name: self.name,
            binding: self.binding,
            sub_buf,
        })
    }
}

macro_rules! impl_into_descriptor_set_by_name_for_tuple {
    ($(($n:tt, $T:ident)),*) => {
        impl<$($T: IntoDescriptorSetByName),*> IntoDescriptorSetByName for ($($T,)*) {

            type Out = ($($T::Out,)*);
            fn with_gpu_buffer(&self,
                memory_allocator: Arc<StandardMemoryAllocator>) -> CrateResult<Self::Out> {
                Ok((
                    $(
                        self.$n.with_gpu_buffer(memory_allocator.clone())?,
                    )*
                ))
            }
        }
    };
}

// impl_descriptor_set_by_name_for_tuple!((0, T0));
all_tuples_enumerated!(impl_into_descriptor_set_by_name_for_tuple, 1, 15, T);

//
//
//

pub trait DescriptorSetByName {
    fn descriptor_set_by_name(&self, name: &str) -> CrateResult<WriteDescriptorSet>;
}

impl<S> DescriptorSetByName for SubbufferAndBindingSpec<S> {
    fn descriptor_set_by_name(&self, name: &str) -> CrateResult<WriteDescriptorSet> {
        if name == self.name {
            Ok(WriteDescriptorSet::buffer(
                self.binding,
                self.sub_buf.clone(),
            ))
        } else {
            Err(crate::error::ChimeraError::DescriptorSetNameNotFound(
                name.to_string(),
            ))
        }
    }
}

macro_rules! impl_descriptor_set_by_name_for_tuple {
    ($(($n:tt, $T:ident)),*) => {
        impl<$($T: DescriptorSetByName),*> DescriptorSetByName for ($($T,)*) {

            fn descriptor_set_by_name(&self, name: &str) -> CrateResult<WriteDescriptorSet> {
                $(
                    if let Ok(write_descriptor_set) = self.$n.descriptor_set_by_name(name) {
                        return Ok(write_descriptor_set);
                    }
                )*
                Err(crate::error::ChimeraError::DescriptorSetNameNotFound (
                     name.to_string()),
                )
            }
        }
    };
}

// impl_descriptor_set_by_name_for_tuple!((0, T0));
all_tuples_enumerated!(impl_descriptor_set_by_name_for_tuple, 1, 15, T);
