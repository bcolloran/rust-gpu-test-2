//! Simple demo showing the same compute kernel running on CPU, CUDA, and Vulkan

use std::vec;

use anyhow::Result;
use glam::Vec2;
use rust_gpu_chimera_demo::{
    runners::vulkano::shader_buffer_mapping::ComputePassInvocationInfo, *,
};

fn run_add_test(
    runner: &VulkanoRunner,
    a: &mut [u32],
    b: &[u32],
    c: &[u32],
    d: &[u32],
    x: &mut [Vec2],
    v: &[Vec2],
) -> Result<()> {
    // Get and log backend info
    let n = a.len();
    // Display results
    println!("    `a` pre: {:?}", &a[..10.min(n)]);
    println!("    `x` pre: {:?}", &x[..10.min(n)]);

    runner.run_compute_shader_sequence(a, b, c, d, x, v)?;

    println!("    `a` post: {:?}", &a[..10.min(n)]);
    println!("    `x` post: {:?}", &x[..10.min(n)]);

    Ok(())
}

fn main() -> Result<()> {
    let adder_kernel = ("adder", vec![0, 1]);
    let step_particles_kernel = ("step_particles", vec![2, 3]);
    let wrap_particles_kernel = ("wrap_particles", vec![2]);

    let shader_buffers = ComputePassInvocationInfo::from_lists(vec![
        ("adder_ab", vec!["a", "b"], adder_kernel.clone()),
        ("adder_ac", vec!["a", "c"], adder_kernel.clone()),
        ("adder_ad", vec!["a", "d"], adder_kernel.clone()),
        (
            "step_particles_0",
            vec!["x", "v"],
            step_particles_kernel.clone(),
        ),
        (
            "step_particles_1",
            vec!["x", "v"],
            step_particles_kernel.clone(),
        ),
        (
            "step_particles_2",
            vec!["x", "v"],
            step_particles_kernel.clone(),
        ),
        (
            "step_particles_3",
            vec!["x", "v"],
            step_particles_kernel.clone(),
        ),
        ("wrap_particles", vec!["x"], wrap_particles_kernel.clone()),
    ]);

    let n = 256;

    let mut a = vec![1u32; n];
    let b = (0..n as u32).collect::<Vec<u32>>();
    let c = vec![30u32; n];
    let d = (0..n as u32).map(|x| x * x).collect::<Vec<u32>>();

    let mut x = (0..n as u32)
        .map(|x| Vec2::new((x as f32).exp().sin(), (x as f32).exp().cos()))
        .collect::<Vec<Vec2>>();

    let v = (0..n as u32)
        .map(|x| Vec2::new((x as f32).exp().sin(), (x as f32).exp().cos()))
        .collect::<Vec<Vec2>>();

    let runner = VulkanoRunner::new(shader_buffers);
    match &runner {
        Ok(r) => {
            run_add_test(&r, &mut a, &b, &c, &d, &mut x, &v)?;
        }
        Err(e) => {
            eprintln!("  Vulkano initialization failed: {e}")
        }
    }

    Ok(())
}
