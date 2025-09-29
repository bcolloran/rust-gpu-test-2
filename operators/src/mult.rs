use glam::UVec3;
use spirv_std::{glam, spirv};

fn mult_update(a: &mut u32, b: u32) {
    *a *= b
}

#[spirv(compute(threads(64)))]
pub fn mult(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] b: &[u32],
) {
    let i = id.x as usize;
    mult_update(&mut a[i], b[i]);
}
