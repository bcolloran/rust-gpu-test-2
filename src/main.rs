//! Simple demo showing the same compute kernel running on CPU, CUDA, and Vulkan

use anyhow::Result;
use rust_gpu_chimera_demo::*;
use shared::{SortOrder, SortableKey};

fn print_header() {
    println!("╔═══════════════════════════════════════════════════╗");
    println!("║     🧬 Rust GPU Chimera Demo - Bitonic Sort 🦀    ║");
    println!("╚═══════════════════════════════════════════════════╝");
}

fn print_test_header(test_name: &str) {
    let width = test_name.len().max(47);
    let top_bottom = "─".repeat(width + 2);
    println!("\n┌{top_bottom}┐");
    println!("│ {test_name:<width$} │");
    println!("└{top_bottom}┘");
}

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

fn run_sort_test<T, R>(runner: &R, data: &mut [T], test_type: &str, order: SortOrder) -> Result<()>
where
    T: SortableKey + bytemuck::Pod + Send + Sync + std::fmt::Debug + PartialOrd + Clone,
    R: SortRunner,
{
    // Get and log backend info
    let (host, backend, adapter, driver) = runner.backend_info();
    log_backend_info(host, backend, adapter.as_deref(), driver.as_deref());

    let len = data.len();
    let original_first_10 = data[..10.min(len)].to_vec();
    let original_last_10 = if len > 10 {
        data[len - 10..].to_vec()
    } else {
        vec![]
    };

    runner.sort(data, order)?;

    // Verify sort
    let is_sorted = match order {
        SortOrder::Ascending => data.windows(2).all(|w| w[0] <= w[1]),
        SortOrder::Descending => data.windows(2).all(|w| w[0] >= w[1]),
    };

    // Display results
    println!("\n  Original (first 10): {original_first_10:?}");
    if !original_last_10.is_empty() {
        println!("  Original (last 10):  {original_last_10:?}");
    }
    println!("  Sorted (first 10):   {:?}", &data[..10.min(len)]);
    if len > 10 {
        println!("  Sorted (last 10):    {:?}", &data[len - 10..]);
    }

    if is_sorted {
        println!("\n  ✅ {test_type} sort ({order}): PASSED ({len} elements)");
        Ok(())
    } else {
        Err(anyhow::anyhow!(
            "{} sort ({}) failed: array not properly sorted",
            test_type,
            order
        ))
    }
}

fn run_add_test<R>(runner: &R, a: &mut [u32], b: &[u32]) -> Result<()>
where
    R: SortRunner,
{
    // Get and log backend info
    let (host, backend, adapter, driver) = runner.backend_info();
    log_backend_info(host, backend, adapter.as_deref(), driver.as_deref());

    let len = a.len();
    let original_first_10_a = a[..10.min(len)].to_vec();
    let original_first_10_b = b[..10.min(len)].to_vec();

    runner.add(a, b)?;
    println!("  ➕ Addition operation completed successfully.");

    // Display results
    println!("\n  Original `a` (first 10 ): {original_first_10_a:?}");
    println!("  Original `b` (first 10 ): {original_first_10_b:?}");

    println!("  Result `a + b` (first 10): {:?}", &a[..10.min(len)]);

    Ok(())
}

fn run_test_on_backend<T>(data: &mut [T], test_type: &str, order: SortOrder) -> Result<()>
where
    T: SortableKey + bytemuck::Pod + Send + Sync + std::fmt::Debug + PartialOrd + Clone,
{
    #[cfg(not(any(feature = "wgpu", feature = "ash", feature = "vulkano")))]
    {
        let runner = CpuRunner;
        run_sort_test(&runner, data, test_type, order)?;
    }

    #[cfg(any(feature = "wgpu", feature = "ash", feature = "vulkano"))]
    {
        let mut gpu_executed = false;

        #[cfg(feature = "wgpu")]
        if !gpu_executed {
            if let Ok(runner) = futures::executor::block_on(WgpuRunner::new()) {
                run_sort_test(&runner, data, test_type, order)?;
                gpu_executed = true;
            } else if let Err(e) = futures::executor::block_on(WgpuRunner::new()) {
                eprintln!("  wgpu initialization failed: {e}");
            }
        }

        #[cfg(feature = "ash")]
        if !gpu_executed {
            if let Ok(runner) = AshRunner::new() {
                run_sort_test(&runner, data, test_type, order)?;
                gpu_executed = true;
            } else if let Err(e) = AshRunner::new() {
                eprintln!("  Vulkan initialization failed: {e}");
            }
        }

        #[cfg(feature = "vulkano")]
        if !gpu_executed {
            if let Ok(runner) = VulkanoRunner::new() {
                run_sort_test(&runner, data, test_type, order)?;
                run_add_test(
                    &runner,
                    &mut vec![1u32; data.len()],
                    &(0..data.len() as u32).collect::<Vec<u32>>(),
                    // &vec![2u32; data.len()],
                )?;
                gpu_executed = true;
            } else if let Err(e) = VulkanoRunner::new() {
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
    println!("{}", sparkler_test::add(1, 3));

    print_header();

    // print_test_header("Demo 1: Sorting 1000 u32 elements");
    // let mut u32_data = vec![0u32; 1000];
    // for (i, v) in u32_data.iter_mut().enumerate() {
    //     *v = ((i * 31337 + 42) % 1000) as u32;
    // }
    // run_test_on_backend(&mut u32_data, "u32", SortOrder::Ascending)?;

    // print_test_header("Demo 2: Sorting u32 with special values");
    // let mut u32_special = vec![
    //     42u32,
    //     7,
    //     999,
    //     0,
    //     13,
    //     256,
    //     128,
    //     1,
    //     u32::MAX,
    //     u32::MIN,
    //     u32::MAX / 2,
    //     u32::MAX - 1,
    //     1000000,
    //     999999,
    //     100,
    //     50,
    // ];
    // run_test_on_backend(&mut u32_special, "u32 special", SortOrder::Ascending)?;

    // print_test_header("Demo 3: Sorting 1000 i32 elements");
    // let mut i32_data = vec![0i32; 1000];
    // for (i, v) in i32_data.iter_mut().enumerate() {
    //     *v = ((i as i32 * 31337 - 500000) % 2000) - 1000;
    // }
    // run_test_on_backend(&mut i32_data, "i32", SortOrder::Ascending)?;

    // print_test_header("Demo 4: Sorting i32 with special values");
    // let mut i32_special = vec![
    //     -42i32,
    //     7,
    //     -999,
    //     0,
    //     13,
    //     -256,
    //     128,
    //     -1,
    //     i32::MAX,
    //     i32::MIN,
    //     i32::MAX / 2,
    //     i32::MIN / 2,
    //     -1000000,
    //     999999,
    //     -100,
    //     50,
    // ];
    // run_test_on_backend(&mut i32_special, "i32 special", SortOrder::Ascending)?;

    print_test_header("Demo 5: Sorting 1000 f32 elements");
    let mut f32_data = vec![0.0f32; 1000];
    for (i, v) in f32_data.iter_mut().enumerate() {
        *v = ((i as f32 * std::f32::consts::PI) - 500.0) * 0.123;
    }
    run_test_on_backend(&mut f32_data, "f32", SortOrder::Ascending)?;

    // print_test_header("Demo 6: Sorting f32 with special values");
    // let mut f32_special = vec![
    //     std::f32::consts::PI,
    //     -2.71,
    //     0.0,
    //     -0.0,
    //     1.41,
    //     -99.9,
    //     42.0,
    //     f32::INFINITY,
    //     f32::NEG_INFINITY,
    //     f32::MAX,
    //     f32::MIN,
    //     f32::MIN_POSITIVE,
    //     -f32::MIN_POSITIVE,
    //     1e-10,
    //     -1e10,
    //     0.1,
    // ];
    // run_test_on_backend(&mut f32_special, "f32 special", SortOrder::Ascending)?;

    // print_test_header("Demo 7: Sorting u32 descending");
    // let u32_desc = vec![42u32, 7, 999, 0, 13, 256, 128, 511, 1, 64];
    // run_test_on_backend(&mut u32_desc.clone(), "u32", SortOrder::Descending)?;

    // print_test_header("Demo 8: Sorting i32 descending with negatives");
    // let i32_desc = vec![-42i32, 7, -999, 0, 13, -256, 128, -1, 100, -100];
    // run_test_on_backend(&mut i32_desc.clone(), "i32", SortOrder::Descending)?;

    // print_test_header("Demo 9: Sorting f32 descending with special values");
    // let f32_desc = vec![
    //     std::f32::consts::PI,
    //     -2.71,
    //     0.0,
    //     -0.0,
    //     1.41,
    //     -99.9,
    //     42.0,
    //     f32::INFINITY,
    //     f32::NEG_INFINITY,
    //     f32::MAX,
    //     f32::MIN,
    // ];
    // run_test_on_backend(&mut f32_desc.clone(), "f32", SortOrder::Descending)?;

    // println!("\n═══════════════════════════════════════════════════");
    // println!("All demos completed successfully! 🎉");
    // println!("═══════════════════════════════════════════════════");

    Ok(())
}
