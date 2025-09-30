takes a rust function and generates a struct that implements OpFn for it.

The rust function should be annotated with `#[op_fn]`.

The function must take a single argument (which may be tuple of multiple types).



For a fn `foo`, this macro will do the following:
- Create a marker struct `fooOpFn`.
- impl `FnTokens` for `fooOpFn` that generates the tokens for the function `foo`.
- create a new function `foo__op_fn_inner` that has the same body as `foo`.
- *Replace* the function `foo` with a const `OpFn<I,O,M>` struct called `foo`, where `I` and `O` are `foo`s input and output types, and `M` is the newly created marker type `fooOp.



Example usage:

```rust
use sparkler_macros::op_fn; 

#[op_fn]
fn add((a, b): (f32, f32)) -> f32 {
    a + b
}
```

Corresponding generated code:

```rust
#[allow(non_camel_case_types)]
struct addOpFn;

impl ::sparkler::FnTokens for addOpFn {
    fn to_tokens(&self) -> ::proc_macro2::TokenStream {
        ::quote::quote! {
            fn add((a, b): (f32, f32)) -> f32 {
                a + b
            }
        }
    }
}

fn add__op_fn_inner((a, b): (f32, f32)) -> f32 {
    a + b
}

#[allow(non_upper_case_globals)]
const add: OpFn<(u64, u64), u64, addOpFn> = OpFn {
    fn_marker: std::marker::PhantomData,
    str: "
fn add2((a, b): (u64, u64)) -> u64 {
     a + b
}
",
    f: add_inner,
};
```
