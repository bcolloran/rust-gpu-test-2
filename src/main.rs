//! Simple demo showing the same compute kernel running on CPU, CUDA, and Vulkan

use std::vec;

use anyhow::Result;
use glam::Vec2;
use rust_gpu_chimera_demo::{
    runners::vulkano::shader_buffer_mapping::ComputePassInvocationInfo, *,
};

fn run_add_test(
    runner: &VulkanoRunner,
    // global_buf_to_binding: BufNameToBinding,
    a: &mut [u32],
    b: &[u32],
    c: &[u32],
    d: &[u32],
    x: &mut [Vec2],
    v: &[Vec2],
) -> Result<()> {
    // Get and log backend info

    let len = a.len();
    let original_first_10_a = a[..10.min(len)].to_vec();
    let original_first_10_b = b[..10.min(len)].to_vec();
    // Display results
    println!("\n  Original `a` (first 10 ): {original_first_10_a:?}");
    println!("  Original `b` (first 10 ): {original_first_10_b:?}");

    println!("\n  Original `x` (first 10 ): {:?}", &x[..10.min(len)]);

    runner.execute_adder_kernel_pass(
        // global_buf_to_binding,
        a, b, c, d, x, v,
    )?;
    println!("  ➕ Addition operation completed successfully.");

    println!("\n  Post `x` (first 10 ): {:?}", &x[..10.min(len)]);

    println!(
        "  Result in `a` after adds (first 10): {:?}",
        &a[..10.min(len)]
    );

    Ok(())
}

fn run_test_on_backend<T>(
    data: &mut [T],
    // global_buf_to_binding: BufNameToBinding,
    entry_point_names_to_buffers: ComputePassInvocationInfo,
) -> Result<()>
where
    T: bytemuck::Pod + Send + Sync + std::fmt::Debug + PartialOrd + Clone,
{
    {
        let mut gpu_executed = false;

        let mut x = (0..data.len() as u32)
            .map(|x| Vec2::new((x as f32).exp().sin(), (x as f32).exp().cos()))
            .collect::<Vec<Vec2>>();

        let v = (0..data.len() as u32)
            .map(|x| Vec2::new((x as f32).exp().sin(), (x as f32).exp().cos()))
            .collect::<Vec<Vec2>>();

        if !gpu_executed {
            let runner = VulkanoRunner::new(
                // global_buf_to_binding.clone(),
                entry_point_names_to_buffers.clone(),
            );
            match &runner {
                Ok(r) => {
                    // run_sort_test(&runner, data, test_type, order)?;
                    run_add_test(
                        &r,
                        // global_buf_to_binding.clone(),
                        &mut vec![1u32; data.len()],
                        &(0..data.len() as u32).collect::<Vec<u32>>(),
                        &vec![30u32; data.len()],
                        &(0..data.len() as u32).map(|x| x * x).collect::<Vec<u32>>(),
                        &mut x,
                        &v,
                    )?;
                    gpu_executed = true;
                }
                Err(e) => {
                    eprintln!("  Vulkano initialization failed: {e}")
                }
            }
        }

        if !gpu_executed {
            eprintln!("\n  ❌ No GPU backend available");
            return Err(anyhow::anyhow!("No GPU backend available"));
        }
    }

    Ok(())
}

fn main() -> Result<()> {
    // let global_buf_to_binding = BufNameToBinding::from_list(vec![
    //     ("a", 0),
    //     ("b", 1),
    //     ("x", 2),
    //     ("v", 3),
    //     ("c", 4),
    //     ("d", 5),
    // ]);

    let n = 256;

    let a = vec![1u32; n];
    let b = (0..n as u32).collect::<Vec<u32>>();
    let x = (0..n as u32)
        .map(|x| Vec2::new((x as f32).exp().sin(), (x as f32).exp().cos()))
        .collect::<Vec<Vec2>>();

    let v = (0..n as u32)
        .map(|x| Vec2::new((x as f32).exp().sin(), (x as f32).exp().cos()))
        .collect::<Vec<Vec2>>();

    let bufs = (("a", a), ("b", b), ("x", x), ("v", v));

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

    let mut f32_data = vec![0.0f32; 1000];
    for (i, v) in f32_data.iter_mut().enumerate() {
        *v = ((i as f32 * std::f32::consts::PI) - 500.0) * 0.123;
    }
    run_test_on_backend(&mut f32_data, shader_buffers)?;

    Ok(())
}
