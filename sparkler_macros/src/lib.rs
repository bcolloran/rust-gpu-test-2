use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{ItemFn, ReturnType, Type, spanned::Spanned};

#[cfg(test)]
mod tests;

/// Attribute macro applied to a free function turning it into a `const OpFn` plus
/// a marker type that implements `::sparkler::FnTokens` for code generation.
///
/// See README for detailed transformation.
#[proc_macro_attribute]
pub fn op_fn(_attr: TokenStream, item: TokenStream) -> TokenStream {
    // Parse the incoming item as a function.
    let input_fn: ItemFn = match syn::parse(item) {
        Ok(f) => f,
        Err(e) => return e.to_compile_error().into(),
    };
    expand_op_fn_macro(input_fn).into()
}

/// Core expansion logic (exposed for tests).
pub(crate) fn expand_op_fn_macro(input_fn: ItemFn) -> proc_macro2::TokenStream {
    // Ensure exactly one argument.
    if input_fn.sig.inputs.len() != 1 {
        return syn::Error::new(
            input_fn.sig.span(),
            "#[op_fn] functions must take exactly one argument (use a tuple for multiple values)",
        )
        .to_compile_error();
    }

    // Extract the (sole) argument pattern & type.
    let arg = input_fn.sig.inputs.first().unwrap();
    let (_arg_pat, in_ty): (&syn::Pat, &Type) = match arg {
        syn::FnArg::Typed(pat_ty) => (&*pat_ty.pat, &*pat_ty.ty),
        syn::FnArg::Receiver(recv) => {
            return syn::Error::new(recv.span(), "methods are not supported").to_compile_error();
        }
    };

    // Determine output type (default to () if omitted).
    let out_ty: proc_macro2::TokenStream = match &input_fn.sig.output {
        ReturnType::Default => quote! { () },
        ReturnType::Type(_, ty) => quote! { #ty },
    };

    let vis = &input_fn.vis; // visibility of generated items
    let original_fn_name = &input_fn.sig.ident;

    // Marker struct name: <fn_name>OpFn (keep original casing, add suffix making it CamelCase-ish)
    // If original already starts lowercase (likely), we still allow non_camel_case_types.
    let marker_ident = format_ident!("{}OpFn", original_fn_name);
    let inner_ident = format_ident!("{}__op_fn_inner", original_fn_name);

    // Build inner function (same signature except name) reusing argument & return type.
    let attrs = &input_fn.attrs; // propagate non-macro attributes to inner fn? For now, we drop macro attr.
    // Filter out the op_fn attribute itself when copying attributes.
    let filtered_attrs: Vec<_> = attrs
        .iter()
        .filter(|a| !a.path().is_ident("op_fn"))
        .collect();

    let sig = &input_fn.sig;
    let generics = &sig.generics;
    if !generics.params.is_empty() {
        return syn::Error::new(
            sig.generics.span(),
            "#[op_fn] does not yet support generic functions",
        )
        .to_compile_error();
    }
    let output = &sig.output;
    let block = &input_fn.block;

    // Reconstruct a function item representing the original (for tokens inside FnTokens impl + string copy)
    let orig_fn_for_tokens = quote! {
        #(#filtered_attrs)*
        #vis fn #original_fn_name(#arg) #output #block
    };

    // Build code string (multi-line string) with a leading newline matching README style.
    let code_string = format!("\n{}\n", orig_fn_for_tokens.to_string());

    // Compose final expanded tokens.
    quote! {
    // Inner function containing the logic (call target for OpFn)
    #[allow(non_snake_case)]
    #(#filtered_attrs)*
    fn #inner_ident(#arg) #output #block

    #[allow(non_camel_case_types)]
    #vis struct #marker_ident;

    impl ::sparkler::FnTokens for #marker_ident {
        fn to_tokens() -> proc_macro2::TokenStream {
            ::quote::quote! { #orig_fn_for_tokens }
        }
    }

    #[allow(non_upper_case_globals)]
    #vis const #original_fn_name: ::sparkler::OpFn<#in_ty, #out_ty, #marker_ident> = ::sparkler::OpFn::new(

        #code_string,
        #inner_ident
      );
    }
}
