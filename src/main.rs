//! Simple demo showing the same compute kernel running on CPU, CUDA, and Vulkan

use anyhow::Result;
use rust_gpu_chimera_demo::{
    runners::vulkano::shader_buffer_mapping::{BufNameToBinding, EntryPointNameToBuffers},
    *,
};

fn log_backend_info(
    host: &str,
    backend: Option<&str>,
    adapter: Option<&str>,
    driver: Option<&str>,
) {
    println!("  Host: {host}");

    if let Some(b) = backend {
        println!("  Backend: {b}");
    }

    if let Some(a) = adapter {
        println!("  Adapter: {a}");
    }

    if let Some(d) = driver {
        if !d.is_empty() {
            println!("  Driver: {d}");
        }
    }
}

fn run_add_test<R>(runner: &R, a: &mut [u32], b: &[u32], c: &[u32], d: &[u32]) -> Result<()>
where
    R: SortRunner,
{
    // Get and log backend info
    let (host, backend, adapter, driver) = runner.backend_info();
    log_backend_info(host, backend, adapter.as_deref(), driver.as_deref());

    let len = a.len();
    let original_first_10_a = a[..10.min(len)].to_vec();
    let original_first_10_b = b[..10.min(len)].to_vec();

    runner.add(a, b, c, d)?;
    println!("  ➕ Addition operation completed successfully.");

    // Display results
    println!("\n  Original `a` (first 10 ): {original_first_10_a:?}");
    println!("  Original `b` (first 10 ): {original_first_10_b:?}");

    println!(
        "  Result in `a` after adds (first 10): {:?}",
        &a[..10.min(len)]
    );

    Ok(())
}

fn run_test_on_backend<T>(
    data: &mut [T],
    global_buf_to_binding: BufNameToBinding,
    entry_point_names_to_buffers: EntryPointNameToBuffers,
) -> Result<()>
where
    T: bytemuck::Pod + Send + Sync + std::fmt::Debug + PartialOrd + Clone,
{
    {
        let mut gpu_executed = false;

        if !gpu_executed {
            if let Ok(runner) = VulkanoRunner::new(
                global_buf_to_binding.clone(),
                entry_point_names_to_buffers.clone(),
            ) {
                // run_sort_test(&runner, data, test_type, order)?;
                run_add_test(
                    &runner,
                    &mut vec![1u32; data.len()],
                    &(0..data.len() as u32).collect::<Vec<u32>>(),
                    &vec![3u32; data.len()],
                    &(0..data.len() as u32).map(|x| x * x).collect::<Vec<u32>>(),
                )?;
                gpu_executed = true;
            } else if let Err(e) =
                VulkanoRunner::new(global_buf_to_binding, entry_point_names_to_buffers)
            {
                eprintln!("  Vulkano initialization failed: {e}");
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
    let global_buf_to_binding =
        BufNameToBinding::from_list(vec![("a", 0), ("b", 1), ("x", 2), ("v", 3)]);

    let shader_buffers = EntryPointNameToBuffers::from_lists(vec![
        ("adder", vec![("a", 0), ("b", 1)]),
        ("step_particles", vec![("x", 2), ("v", 3)]),
        ("wrap_particles", vec![("x", 2)]),
    ]);

    shader_buffers.validate_against_global_buf_names(&global_buf_to_binding);

    let mut f32_data = vec![0.0f32; 1000];
    for (i, v) in f32_data.iter_mut().enumerate() {
        *v = ((i as f32 * std::f32::consts::PI) - 500.0) * 0.123;
    }
    run_test_on_backend(&mut f32_data, global_buf_to_binding, shader_buffers)?;

    Ok(())
}
