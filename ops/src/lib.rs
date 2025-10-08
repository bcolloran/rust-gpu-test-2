#![no_std]

use glam::UVec3;
use shared::RowA;
use sparkler::buf::MapOp;
use spirv_std::{glam, spirv};

pub mod mult;

// #[sparkler::op_fn]
pub fn row_map(row: &RowA) -> (u32, f32, f32) {
    (row.x * 2, row.y * 3.0, row.x as f32 + row.y)
}

// #[sparkler::op_fn]
pub fn accumulate(init: (f32, u32), v: &(f32, u32)) -> (f32, u32) {
    (init.0 + v.0, init.1 + v.1)
}

// #[sparkler::op_fn]
pub fn combine_rows(a: &(f32, u32)) -> f32 {
    a.0 + a.1 as f32
}

// #[sparkler::op_fn]
pub fn reduce_1(a: &f32, b: &f32) -> f32 {
    a + b
}

#[allow(non_upper_case_globals)]
pub const buf_map: MapOp<RowA, f32> = MapOp::new(|x: &RowA| -> f32 { x.y * 2.0 + x.x as f32 });
