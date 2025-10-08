#![cfg_attr(target_arch = "spirv", no_std)]
// HACK(eddyb) can't easily see warnings otherwise from `spirv-builder` builds.
#![deny(warnings)]

use glam::UVec3;
use shared::RowA;
use spirv_std::{glam, spirv};

pub mod mult;

#[spirv(compute(threads(64)))]
pub fn buf_map(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] in_buf: &[RowA],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] out_buf: &mut [f32],
) {
    let i = id.x as usize;
    let in_elt = &in_buf[i];
    let op = |x: &RowA| -> f32 { x.y * 2.0 + x.x as f32 };
    out_buf[i] = op(in_elt);
}
