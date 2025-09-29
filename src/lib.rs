//! Rust GPU Chimera Demo Library
//!
//! This library demonstrates running the same Rust code on:
//! - CPU (native Rust)
//! - Vulkan (via rust-gpu/SPIR-V)

#![feature(once_cell_try)]

// Feature validation

#[cfg(all(feature = "wgpu", feature = "ash"))]
compile_error!("Cannot enable both 'wgpu' and 'ash' features at the same time");

// Allow vulkano to be combined for comparison; if exclusivity desired add similar compile_error.

#[cfg(all(target_os = "macos"))]
compile_error!("The 'cuda' feature is not supported on macOS. CUDA requires NVIDIA GPUs and is only available on Linux and Windows");

pub mod error;
pub mod runners;

use error::Result;
use shared::{BitonicParams, Pass, SortOrder, SortableKey, Stage};

/// Common trait for all sorting backends
pub trait SortRunner {
    /// Get backend information for logging
    ///
    /// Returns a tuple of (host, backend, adapter, driver)
    fn backend_info(
        &self,
    ) -> (
        &'static str,
        Option<&'static str>,
        Option<String>,
        Option<String>,
    );

    /// Execute a single kernel pass - platform-specific implementation required
    ///
    /// # Arguments
    /// * `data` - The data slice to sort in-place
    /// * `params` - Bitonic sort parameters for this pass
    fn execute_kernel_pass(&self, data: &mut [u32], params: BitonicParams) -> Result<()>;

    fn execute_adder_kernel_pass(&self, a: &mut [u32], b: &[u32]) -> Result<()>;

    /// Prepare data by converting to `u32` representation
    fn prepare_data<T: SortableKey>(&self, data: &[T]) -> (Vec<u32>, usize) {
        let gpu_data: Vec<u32> = data.iter().map(|x| x.to_sortable_u32()).collect();
        (gpu_data, data.len())
    }

    /// Pad data to power of 2 size with appropriate sentinel values
    fn pad_data(&self, data: &mut Vec<u32>, original_size: usize, order: SortOrder) {
        let padded_size = original_size.next_power_of_two();
        if padded_size > original_size {
            let sentinel = match order {
                SortOrder::Ascending => u32::MAX,
                SortOrder::Descending => u32::MIN,
            };
            data.resize(padded_size, sentinel);
        }
    }

    /// Run all bitonic sort stages and passes
    fn run_bitonic_stages(&self, data: &mut [u32], order: SortOrder) -> Result<()> {
        let n = data.len() as u32;
        let num_stages = (n as f32).log2() as u32;

        for stage in 0..num_stages {
            for pass in 0..=stage {
                let params = BitonicParams {
                    num_elements: n,
                    stage: Stage::new(stage),
                    pass_of_stage: Pass::new(pass),
                    sort_order: order.into(),
                };
                self.execute_kernel_pass(data, params)?;
            }
        }
        Ok(())
    }

    fn run_adder_pass(&self, a: &mut [u32], b: &[u32]) -> Result<()> {
        assert_eq!(a.len(), b.len());
        self.execute_adder_kernel_pass(a, b)
    }

    /// Convert sorted `u32` data back to original type
    fn finalize_data<T: SortableKey>(&self, gpu_data: &[u32], output: &mut [T]) {
        for (i, &val) in gpu_data.iter().take(output.len()).enumerate() {
            output[i] = T::from_sortable_u32(val);
        }
    }

    /// Sort data with specified order (ascending or descending)
    ///
    /// Sorts the given slice in-place using the bitonic sort algorithm.
    /// The data is converted to `u32` for sorting, then converted back.
    fn sort<T: SortableKey + bytemuck::Pod + Send + Sync>(
        &self,
        data: &mut [T],
        order: SortOrder,
    ) -> Result<()> {
        if data.len() <= 1 {
            return Ok(());
        }

        let (mut gpu_data, original_size) = self.prepare_data(data);
        self.pad_data(&mut gpu_data, original_size, order);
        self.run_bitonic_stages(&mut gpu_data, order)?;
        gpu_data.truncate(original_size);
        self.finalize_data(&gpu_data, data);

        Ok(())
    }

    fn add(&self, a: &mut [u32], b: &[u32]) -> Result<()> {
        assert_eq!(a.len(), b.len());
        self.run_adder_pass(a, b)
    }
}

// Re-export runners for convenience
pub use runners::CpuRunner;

#[cfg(feature = "wgpu")]
pub use runners::WgpuRunner;

#[cfg(feature = "ash")]
pub use runners::AshRunner;

#[cfg(feature = "vulkano")]
pub use runners::VulkanoRunner;

/// Compiled SPIR-V bytecode for the bitonic sort kernel
#[cfg(any(feature = "wgpu", feature = "ash", feature = "vulkano"))]
pub const BITONIC_SPIRV: &[u8] = include_bytes!(env!("BITONIC_KERNEL_SPV_PATH"));

#[cfg(any(feature = "vulkano"))]
pub const OTHER_SHADERS_SPIRV: &[u8] = include_bytes!(env!("OTHER_SHADERS_SPV_PATH"));
#[cfg(any(feature = "vulkano"))]
pub const OTHER_SHADERS_ENTRY_ADDER: &str = env!("OTHER_SHADERS_ENTRY_ADDER");

/// Verify that a slice is sorted in the specified order
#[cfg(test)]
pub fn verify_sorted<T: SortableKey + PartialOrd>(data: &[T], order: SortOrder) -> bool {
    match order {
        SortOrder::Ascending => data.windows(2).all(|w| w[0] <= w[1]),
        SortOrder::Descending => data.windows(2).all(|w| w[0] >= w[1]),
    }
}
