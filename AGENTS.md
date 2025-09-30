

# Getting started
Before starting any implementation, the agent should read the entire specification provided by the user. The agent should make sure to understand the requirements and constraints of the task. The agent should come up with a plan for how to implement the task, breaking it down into smaller sub-tasks if necessary.

The agent should make note of any apparent misspecifications, ambiguities in the spec, or edge cases or surprises that come up during the implementation, and report those to the user in the post-implementation summary and in the PR description.

In most cases, the agent should make a best-effort attempt to implement the task, and inform the user of any uncertainties or issues that arise during the implementation. However, if the agent is *very* confident that the task is untenable, or if there are any *deep* uncertainties about the task, the agent should inform the user and ask for guidance before starting any implementation.


# Tests
**Before making any changes to the code, the agent must run all tests to ensure that they pass. If any tests fail, the agent must immediately halt all work and inform the user.**

## Running tests
The agent should always run `cargo test` in the workspace root to run all tests in the workspace, not just the tests in the current crate. Do not pass additional flags to `cargo test` unless specifically requested by the user.

## Test file organization
Try to ensure that the file organization for tests matches that of the implementation code, e.g., if `TraitX` for `TypeY` is implemented in the file `type_y.rs`, put tests in `tests/type_y.rs`. If there are many tests for a single file, consider putting them in a subdirectory, e.g., `tests/type_y/mod.rs` and `tests/type_y/other_tests.rs`.

## Multiple cases in one test
When it makes sense to repeatedly test a single function on multiple *input* cases, use a `#[test_case(data; "data case description")]` attribute on a test to specify the data cases. This allows the test to be run multiple times with different inputs, and will report each case separately in the test results.

This is "DRY"er than writing a separate test function for each case, and cleaner than putting multiple assertion statements in a single test function that loops over the data cases.

For example, we have this in the file `easy_hash/tests/test_utils.rs`:
```rust
#[test_case(0 ; "0u64")]
#[test_case(1 ; "1u64")]
#[test_case(u32::MAX as u64 ; "u32::MAX as u64")]
#[test_case(u64::MAX ; "u64::MAX")]
#[test_case(u64::MAX - 1 ; "u64::MAX - 1")]
#[test_case(0x1234_5678_9abc_def0 ; "0x1234_5678_9abc_def0")]
fn test_split_u64_roundtrip(val: u64) {
    let parts = split_u64(val);
    assert_eq!(join_u32s(parts[0], parts[1]), val);
}
```




# Implementing macros
## Test-driven development for macros
Before implementing or updating a macro the agent should write write test cases that show the expected input and output of the macro. Before writing the tests, the agent should enumerate the different scenarios that the macro should handle, and write test example input/output pairs for each scenario.

The agent should also make note of any invariants that should cause macro expansion to fail, and write test cases that ensure that the macro fails to compile when those invariants are violated.

The agent should then implement the macro to make the tests pass.

## Use fully qualified paths in generated code
When generating code in a macro, always use fully qualified paths to refer to types and functions. This avoids issues with name resolution and ensures that the generated code will compile correctly regardless of the context in which it is used. For example, instead of generating code that refers to `Vec`, generate code that refers to `::std::vec::Vec`.


# Implementation practices
Be sure to break up the macro implementation into small, manageable functions that handle specific parts/cases of the macro expansion. This will make the code easier to read and maintain, and, most importantly, will make the tests easier to write and understand (it is essential for reviews that the macros give clear examples of input and output; breaking up the macro implementation into small functions helps with this).



# Reference materials
The agent may examine the contents of the folder `reference_material/` for reference materials

The agent should not modify any files in the `reference_material/` folder.

There is a lot of information in the reference materials, so to avoid filling up the context window, the agent should only read files from the reference materials when it needs to look something up. The agent should not read all the reference materials at once.

The following reference materials are available:
- `reference_material/vulkano-book-main` The Vulkano book, with info on how to use Vulkano and examples.
- `reference_material/vulkano-master` The source code of the Vulkano library


# Commands
The agent may run cargo commands to build and test the project, but should not run any other commands.

# Web access
The agent may request any pages from github.com (accessing raw files via `https://github.com/.../raw/refs/...` (or whatever the best mechanism is currently) is advised), crates.io, and docs.rs for reference material.

