use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    // Get the root workspace directory from CARGO_MANIFEST_DIR
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let root_path = PathBuf::from(manifest_dir);

    let web_path = root_path.join("web");
    let docs_pkg_path = root_path.join("docs/pkg");

    println!("Building WebAssembly module...");

    // Run wasm-pack build targeting web and outputting to docs/pkg using absolute paths
    let mut cmd = Command::new("wasm-pack");
    cmd.args([
        "build",
        &web_path.to_string_lossy(),
        "--target",
        "web",
        "--out-dir",
        &docs_pkg_path.to_string_lossy(),
    ]);

    // Run the command and wait for completion
    match cmd.status() {
        Ok(status) if status.success() => {
            // Success
        }
        Ok(status) => {
            eprintln!("wasm-pack build failed with exit code: {:?}", status.code());
            std::process::exit(1);
        }
        Err(e) => {
            eprintln!(
                "Failed to execute wasm-pack: {}. Is wasm-pack installed? (cargo install wasm-pack)",
                e
            );
            std::process::exit(1);
        }
    }

    println!("Cleaning up accidental/local duplicate outputs...");

    // Define paths to clean up
    let duplicate_pkg = web_path.join("pkg");
    let duplicate_docs = web_path.join("docs");
    let redundant_pk = root_path.join("docs/pk");

    for path in &[&duplicate_pkg, &duplicate_docs, &redundant_pk] {
        if path.exists() {
            if let Err(e) = fs::remove_dir_all(path) {
                eprintln!("Warning: Failed to remove {:?}: {}", path, e);
            } else {
                if let Some(name) = path.file_name() {
                    println!("Removed duplicate/redundant directory: {:?}", name);
                }
            }
        }
    }

    println!("Success! WASM build output is ready in docs/pkg/.");
}
