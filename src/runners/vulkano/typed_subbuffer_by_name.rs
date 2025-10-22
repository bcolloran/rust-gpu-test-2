use variadics_please::all_tuples_enumerated;
use vulkano::buffer::Subbuffer;

use crate::{
    error::CrateResult,
    runners::vulkano::buffer_specs::{DescriptorSetByName, SubbufferAndBindingSpec},
};

pub trait TypedSubbufferByName {
    fn subbuffer<T: 'static>(&self, name: &str) -> CrateResult<Subbuffer<[T]>>;
}

use std::any::Any;

impl<S: 'static> TypedSubbufferByName for SubbufferAndBindingSpec<S> {
    fn subbuffer<T: 'static>(&self, name: &str) -> CrateResult<Subbuffer<[T]>> {
        if self.name == name {
            // Erase & downcast to the requested type.
            let any: &dyn Any = &self.sub_buf;
            if let Some(buf) = any.downcast_ref::<Subbuffer<[T]>>() {
                return Ok(buf.clone());
            }
        }
        Err(crate::error::ChimeraError::TypedDescriptorSetNameNotFound(
            name.to_string(),
            std::any::type_name::<T>().to_string(),
        ))
    }
}

macro_rules! impl_typed_subbuffer_by_name_for_tuple {
    ($(($n:tt, $T:ident)),*) => {
        impl<$($T: TypedSubbufferByName),*> TypedSubbufferByName for ($($T,)*) {

            fn subbuffer<U: 'static>(&self, name: &str) -> CrateResult<Subbuffer<[U]>> {
                $(
                    if let Ok(sub_buf) = self.$n.subbuffer::<U>(name) {
                        return Ok(sub_buf);
                    }
                )*
                Err(crate::error::ChimeraError::TypedDescriptorSetNameNotFound (
                     name.to_string(),
                     std::any::type_name::<U>().to_string(),
                ))
            }
        }
    };
}

all_tuples_enumerated!(impl_typed_subbuffer_by_name_for_tuple, 1, 15, T);

// macro_rules! impl_descriptor_set_by_name_for_tuple {
//     ($(($n:tt, $T:ident)),*) => {
//         impl<$($T: DescriptorSetByName),*> DescriptorSetByName for ($($T,)*) {

//             fn descriptor_set_by_name(&self, name: &str) -> CrateResult<WriteDescriptorSet> {
//                 $(
//                     if let Ok(write_descriptor_set) = self.$n.descriptor_set_by_name(name) {
//                         return Ok(write_descriptor_set);
//                     }
//                 )*
//                 Err(crate::error::ChimeraError::DescriptorSetNameNotFound (
//                      name.to_string()),
//                 )
//             }
//         }
//     };
// }

// // impl_descriptor_set_by_name_for_tuple!((0, T0));
// all_tuples_enumerated!(impl_descriptor_set_by_name_for_tuple, 1, 15, T);
