use pretty_assertions::assert_eq;
use quote::quote;
use syn::parse_quote;

use crate::expand_op_fn_macro;

#[test]
fn test_simple() {
    let input: syn::ItemFn = parse_quote! {
        fn add((a, b): (u64, u64)) -> u64 { a + b }
    };

    let expected = quote! {
        fn add__op_fn_inner((a, b): (u64, u64)) -> u64 { a + b }

        #[allow(non_camel_case_types)]
        struct addOpFn;

        impl ::sparkler::FnTokens for addOpFn { fn to_tokens() -> proc_macro2::TokenStream { ::quote::quote! { fn add((a, b): (u64, u64)) -> u64 { a + b } } } }

        #[allow(non_upper_case_globals)]
        const add: ::sparkler::OpFn<(u64, u64), u64, addOpFn> = ::sparkler::OpFn::new(
        "\nfn add ((a , b) : (u64 , u64)) -> u64 { a + b }\n",
        add__op_fn_inner);
    };

    let actual = expand_op_fn_macro(input);
    assert_eq!(actual.to_string(), expected.to_string());
}
