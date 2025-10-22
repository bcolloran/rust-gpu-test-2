

# Getting started
Before starting any implementation, the agent should read the entire specification provided by the user. The agent should make sure to understand the requirements and constraints of the task. The agent should come up with a plan for how to implement the task, breaking it down into smaller sub-tasks if necessary.

The agent should make note of any apparent misspecifications, ambiguities in the spec, or edge cases or surprises that come up during the implementation, and report those to the user in the post-implementation summary and in the PR description.

In most cases, the agent should make a best-effort attempt to implement the task, and inform the user of any uncertainties or issues that arise during the implementation. However, if the agent is *very* confident that the task is untenable, or if there are any *deep* uncertainties about the task, the agent should inform the user and ask for guidance before starting any implementation.


# Tests
**Before making any changes to the code, the agent must run all tests to ensure that they pass. If any tests fail, the agent must immediately halt all work and inform the user.**


## Testing *user-written* implementation code
- When writing tests, on code that has been written before the agent's current task, the agent must first analyze the existing code within the context of the overall codebase, along with any comments or descriptions that explain its purpose and functionality.
- The users code may contain errors or inconsistencies! The agents task is to be BRUTALLY CRITICAL of existing code, and to attempt to write tests that will poke holes in the code's correctness and reveal any bugs or issues.
- The agent should identify and address these issues in the tests, ensuring that the tests accurately reflect the intended behavior of the code. It's ok if the tests reveal bugs or issues in the existing code; the goal is to ensure that the code behaves correctly and reliably. Failing tests are ok, and should be reported back to the user for fixing.
- If the agent identifies any issues or inconsistencies in the existing code while writing tests, it should document these findings and provide suggestions for how to address them. However, the agent **must not** modify the existing code unless explicitly instructed to do so by the user.

## Testing *agent-written* implementation code
- When writing tests on code that the agent has written itself during the current task, the agent should ensure that the tests are comprehensive and cover all relevant scenarios for the newly written code.
- The agent should write tests that cover the expected behavior of the newly written code, including edge cases and potential failure modes.
- When writing tests for its own code, the agen may iteratively refine both the code and the tests to ensure that they are correct and reliable.

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


# Error handling
When implementing new features or modules, proper error handling is essential for maintainability and debuggability. This project uses a two-tier error system with strict conventions.

## Error hierarchy and conventions

### Two-tier error system
1. **`ChimeraError`** (top-level, in `src/error.rs`): Contains ALL library error types from external crates (vulkano, std, etc.) with `#[from]` attribute for automatic conversion
2. **Module-specific errors** (e.g., `GraphicsError`): Contains ONLY semantic/domain-specific errors that originate in our code (e.g., "no suitable device found", "shader entry point not found")

### Critical rule: NEVER use `map_err` for error conversion
**Always use the `?` operator directly and rely on automatic `From` trait conversions.** If you find yourself writing `.map_err()`, you're doing it wrong.

```rust
// ❌ BAD: Using map_err for conversion
let result = some_operation()
    .map_err(Into::into)?;

// ❌ BAD: Using map_err with string formatting
let result = some_operation()
    .map_err(|e| ChimeraError::Other(format!("Failed: {e}")))?;

// ✅ GOOD: Direct ? operator with automatic conversion
let result = some_operation()?;
```

### Where to put error variants

**Library errors → `ChimeraError`**
- Any error type from an external crate (vulkano, std::io::Error, etc.)
- Add a variant to `ChimeraError` with `#[from]` attribute
- This enables automatic conversion via `?` operator

```rust
// In src/error.rs
#[derive(Error, Debug)]
pub enum ChimeraError {
    #[error("vulkano VulkanError: {0}")]
    VulkanoVulkanError(#[from] vulkano::VulkanError),
    
    #[error("vulkano ValidatedVulkanError: {0}")]
    VulkanoValidatedVulkanError(#[from] Validated<vulkano::VulkanError>),
    
    // ... more library errors
}
```

**Semantic errors → Module-specific error enums**
- Errors that represent business logic failures in YOUR code
- Usually come from `.ok_or()` on `Option` types
- Should be descriptive and include context

```rust
// In src/graphics/error.rs
#[derive(Error, Debug)]
pub enum GraphicsError {
    #[error("No suitable device found")]
    NoSuitableDevice,
    
    #[error("Vertex shader entry point '{0}' not found")]
    VertexShaderEntryPointNotFound(String),
    
    // ... more semantic errors
}
```

### Handling `Validated<T>` errors from Vulkano
Vulkano wraps some errors in `Validated<T>`. **Do NOT use `.map_err(Validated::unwrap)`**. Instead, add the `Validated<ErrorType>` to `ChimeraError` with `#[from]` and match on `Validated::Error` when you need to handle specific error cases.

```rust
// ❌ BAD: Using map_err to unwrap Validated
let result = operation().map_err(Validated::unwrap)?;

// ✅ GOOD: Add Validated<ErrorType> to ChimeraError, then use ? directly
// In src/error.rs:
#[error("vulkano ValidatedVulkanError: {0}")]
VulkanoValidatedVulkanError(#[from] Validated<vulkano::VulkanError>),

// In your code - automatic conversion:
let result = operation()?;

// In your code - matching specific errors when needed:
match operation() {
    Ok(value) => value,
    Err(Validated::Error(VulkanError::OutOfDate)) => {
        // Handle specific error
    }
    Err(e) => return Err(e.into()),
}
```

## Avoid the `Other(String)` escape hatch
The `ChimeraError::Other(String)` variant exists as a last resort but should be avoided in production code. Every error should have its own specific variant in the appropriate error enum.

## Implementation strategy
When you encounter an error that doesn't have a proper variant:

1. **Identify the error type**: Remove the problematic error handling and attempt to compile - the compiler will reveal the actual error type
2. **Determine where it belongs**:
   - **Library error?** Add a variant to `ChimeraError` with `#[from]`
   - **Semantic error?** Add a variant to the module-specific error enum
3. **Use `?` operator**: Never use `.map_err()` - always rely on automatic `From` conversions

Example workflow:
```rust
// Step 1: You have this bad code
let device = devices.next()
    .ok_or_else(|| ChimeraError::Other("No device found".to_string()))?;

// Step 2: Add semantic error variant to module error enum
#[derive(Error, Debug)]
pub enum GraphicsError {
    #[error("No suitable device found")]
    NoSuitableDevice,
}

// Step 3: Use it directly
let device = devices.next()
    .ok_or(GraphicsError::NoSuitableDevice)?;
```

## Summary of rules
1. **Library errors** → `ChimeraError` with `#[from]`
2. **Semantic errors** → Module-specific error enums (e.g., `GraphicsError`)
3. **NEVER** use `.map_err()` for error conversion
4. **ALWAYS** use `?` operator with automatic `From` conversions
5. For `Validated<T>` errors, add `Validated<T>` to `ChimeraError` with `#[from]`
6. Avoid `ChimeraError::Other(String)` in production code


# Reference materials
The agent may examine the contents of the folder `reference_material/` for reference materials

The agent should not modify any files in the `reference_material/` folder.

There is a lot of information in the reference materials, so to avoid filling up the context window, the agent should only read files from the reference materials when it needs to look something up. The agent should not read all the reference materials at once.

The following reference materials are available:
- `reference_material/vulkano-book-main` The Vulkano book, with info on how to use Vulkano and examples.
- `reference_material/vulkano-master` The source code of the Vulkano library


# "Removing" files, "corrupted" files, and file size limits
Sometimes an agent will get stuck in a loop of thinking that a file is "corrupted", and will attempt to remove and rewrite the file over and over. 

To avoid this: rather than removing existing files, the agent should start fresh in a new file with a similar name. For example, if you think `some_file.rs` has become corrupted, create a new file `some_file_1.rs` (and so on in sequence), and just point all the modules that had used `some_file.rs` at the newest version.

Leave the old versions on disk. this will allow me to evaluate what went wrong.

## File size recommendation
If you try to limit your file sizes to a couple hundred lines and make sure that each file is focused on a smaller unit of fumctionality, there is less risk of one big file being corrupted


# Commands
The agent may run cargo commands to build and test the project, but should not run any other commands.

## Pre-approved cargo commands
The agent is strongly encouraged to run the following cargo commands as needed:
`cargo test --workspace --verbose`
`cargo check --lib 2>&1 | head -60`
`cargo check --lib 2>&1 | tail -60`

## Forbidden commands
The agent may **NEVER** run any commands that modify files on disk such as `rm`, `sed`, `mv`, `cp`, etc.

# Web access
The agent may request any pages from github.com (accessing raw files via `https://github.com/.../raw/refs/...` (or whatever the best mechanism is currently) is advised), crates.io, and docs.rs for reference material.

