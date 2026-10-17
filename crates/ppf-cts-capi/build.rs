// File: crates/ppf-cts-capi/build.rs
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Best-effort generation of include/ppf_cts.h from the #[no_mangle]
// surface via cbindgen. A cbindgen failure emits a cargo warning and does
// NOT fail the build: the header is a developer convenience for the C/C++
// host, not a compile input for this staticlib.

fn main() {
    let crate_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let config_path = format!("{crate_dir}/cbindgen.toml");
    let out_path = format!("{crate_dir}/include/ppf_cts.h");

    // Regenerate only when the surface or config changes.
    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=src/params.rs");
    println!("cargo:rerun-if-changed=src/error.rs");
    println!("cargo:rerun-if-changed=cbindgen.toml");

    let config = match cbindgen::Config::from_file(&config_path) {
        Ok(c) => c,
        Err(e) => {
            println!("cargo:warning=ppf-cts-capi: cbindgen config load failed: {e}");
            return;
        }
    };

    match cbindgen::Builder::new()
        .with_crate(&crate_dir)
        .with_config(config)
        .generate()
    {
        Ok(bindings) => {
            if let Some(parent) = std::path::Path::new(&out_path).parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            bindings.write_to_file(&out_path);
        }
        Err(e) => {
            println!("cargo:warning=ppf-cts-capi: cbindgen header generation failed: {e}");
        }
    }
}
