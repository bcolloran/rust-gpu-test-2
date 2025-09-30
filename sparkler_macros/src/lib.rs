use proc_macro::TokenStream;
use proc_macro2::Ident as PMIdent;
use quote::quote;
use syn::{Attribute, Data, DataEnum, DeriveInput, Fields, Type, spanned::Spanned};

#[cfg(test)]
mod tests;

#[proc_macro]
pub fn op_fn(input: TokenStream) -> TokenStream {
    let derive_input = syn::parse_macro_input!(input);
    TokenStream::from(expand_op_fn_macro(derive_input))
}

pub(crate) fn expand_op_fn_macro(input: DeriveInput) -> proc_macro2::TokenStream {
    todo!()
}
