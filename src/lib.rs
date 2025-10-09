//! Rust GPU Chimera Demo Library
//!
//! This library demonstrates running the same Rust code on:
//! - CPU (native Rust)
//! - Vulkan (via rust-gpu/SPIR-V)

#![feature(once_cell_try)]

pub mod error;
pub mod runners;

use error::Result;
use shared::{SortOrder, SortableKey};

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
    // fn execute_kernel_pass(&self, data: &mut [u32], params: BitonicParams) -> Result<()>;

    fn execute_adder_kernel_pass(
        &self,
        a: &mut [u32],
        b: &[u32],
        c: &[u32],
        d: &[u32],
    ) -> Result<()>;

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

    fn run_adder_pass(&self, a: &mut [u32], b: &[u32], c: &[u32], d: &[u32]) -> Result<()> {
        assert_eq!(a.len(), b.len());
        self.execute_adder_kernel_pass(a, b, c, d)
    }

    /// Convert sorted `u32` data back to original type
    fn finalize_data<T: SortableKey>(&self, gpu_data: &[u32], output: &mut [T]) {
        for (i, &val) in gpu_data.iter().take(output.len()).enumerate() {
            output[i] = T::from_sortable_u32(val);
        }
    }

    fn add(&self, a: &mut [u32], b: &[u32], c: &[u32], d: &[u32]) -> Result<()> {
        assert_eq!(a.len(), b.len());
        self.run_adder_pass(a, b, c, d)
    }
}

pub use runners::VulkanoRunner;

#[cfg(any(feature = "vulkano"))]
pub const OTHER_SHADERS_SPIRV: &[u8] = include_bytes!(env!("SHADERS_SPV_PATH"));
#[cfg(any(feature = "vulkano"))]
pub const SHADERS_ENTRY_ADDER: &str = env!("SHADERS_ENTRY_ADDER");

/// Verify that a slice is sorted in the specified order
#[cfg(test)]
pub fn verify_sorted<T: SortableKey + PartialOrd>(data: &[T], order: SortOrder) -> bool {
    match order {
        SortOrder::Ascending => data.windows(2).all(|w| w[0] <= w[1]),
        SortOrder::Descending => data.windows(2).all(|w| w[0] >= w[1]),
    }
}
