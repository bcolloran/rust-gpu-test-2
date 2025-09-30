use pretty_assertions::assert_eq;
use quote::quote;
use syn::parse_quote;

use crate::expand_op_fn_macro;

#[test]
fn test_simple() {
    let input = parse_quote! {
        #[op_fn]
        fn add((a, b): (u64, u64)) -> u64 {
            a + b
        }
    };

    let expected = quote! {
        fn add3_inner((a, b): (u64, u64)) -> u64 {
            a + b
        }

        #[allow(non_camel_case_types)]
        pub struct Op_add3;

        impl ::sparkler::FnTokens for Op_add3 {
            fn to_tokens() -> proc_macro2::TokenStream {
                quote::quote! {
                    fn add2((a, b): (u64, u64)) -> u64 {
                        a + b
                    }
                }
            }
        }

        #[allow(non_upper_case_globals)]
        pub const add: OpFn<(u64, u64), u64, Op_add3> = OpFn {
            fn_marker: std::marker::PhantomData,
            str: "
fn add2((a, b): (u64, u64)) -> u64 {
     a + b
}
",
            f: add3_inner,
        };
    };

    let actual = expand_op_fn_macro(input);
    assert_eq!(actual.to_string(), expected.to_string());
}
