use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    // Get the version from CLI arguments (after '--'), defaulting to v1.0.0
    let args: Vec<String> = env::args().collect();
    let version = if args.len() > 1 {
        &args[1]
    } else {
        "v1.0.0"
    };

    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let root_path = PathBuf::from(manifest_dir);

    // Verify GitHub CLI (gh) is installed
    match Command::new("gh").arg("--version").output() {
        Ok(_) => {}
        Err(_) => {
            eprintln!("Error: GitHub CLI (gh) is not installed or not in PATH.");
            eprintln!("Please install it and log in using 'gh auth login' before running this tool.");
            std::process::exit(1);
        }
    }

    // Check if local model weights exist in target directory
    let mnist_weights = root_path.join("target/mnist-model/model.bin");
    let qd_weights = root_path.join("target/quickdraw-model/model.bin");

    if !mnist_weights.exists() {
        eprintln!("Error: Local MNIST weights not found at {:?}.", mnist_weights);
        eprintln!("Please run training first using: cargo run --release -- --dataset mnist");
        std::process::exit(1);
    }
    if !qd_weights.exists() {
        eprintln!(
            "Error: Local Quick Draw weights not found at {:?}.",
            qd_weights
        );
        eprintln!("Please run training first using: cargo run --release -- --dataset quickdraw");
        std::process::exit(1);
    }

    println!("Ensuring GitHub Release '{}' exists...", version);
    // Try creating the release. If it already exists, gh CLI will report a warning but continue safely.
    let notes = format!(
        "Pre-trained model weights for offline WebAssembly inference ({})",
        version
    );
    let _ = Command::new("gh")
        .args([
            "release",
            "create",
            version,
            "--title",
            version,
            "--notes",
            &notes,
        ])
        .status();

    println!("Preparing model binaries for upload and local dev...");
    let temp_mnist = root_path.join("mnist-model.bin");
    let temp_qd = root_path.join("quickdraw-model.bin");
    let docs_mnist = root_path.join("docs/mnist-model.bin");
    let docs_qd = root_path.join("docs/quickdraw-model.bin");

    if let Err(e) = fs::copy(&mnist_weights, &temp_mnist) {
        eprintln!("Failed to copy to {:?}: {}", temp_mnist, e);
        std::process::exit(1);
    }
    if let Err(e) = fs::copy(&qd_weights, &temp_qd) {
        eprintln!("Failed to copy to {:?}: {}", temp_qd, e);
        let _ = fs::remove_file(&temp_mnist);
        std::process::exit(1);
    }
    if let Err(e) = fs::copy(&mnist_weights, &docs_mnist) {
        eprintln!("Failed to copy to {:?}: {}", docs_mnist, e);
        let _ = fs::remove_file(&temp_mnist);
        let _ = fs::remove_file(&temp_qd);
        std::process::exit(1);
    }
    if let Err(e) = fs::copy(&qd_weights, &docs_qd) {
        eprintln!("Failed to copy to {:?}: {}", docs_qd, e);
        let _ = fs::remove_file(&temp_mnist);
        let _ = fs::remove_file(&temp_qd);
        std::process::exit(1);
    }

    println!(
        "Uploading model weights to GitHub Release {} (overwriting previous assets)...",
        version
    );
    let upload_status = Command::new("gh")
        .args([
            "release",
            "upload",
            version,
            "mnist-model.bin",
            "quickdraw-model.bin",
            "--clobber",
        ])
        .current_dir(&root_path)
        .status();

    // Clean up temporary files
    println!("Cleaning up temporary files...");
    let _ = fs::remove_file(&temp_mnist);
    let _ = fs::remove_file(&temp_qd);

    match upload_status {
        Ok(status) if status.success() => {
            // Success
        }
        _ => {
            eprintln!("Error: Failed to upload release assets to GitHub.");
            std::process::exit(1);
        }
    }

    println!("Updating weights version files...");
    let version_files = [
        root_path.join("web/weights-version.txt"),
        root_path.join("docs/weights-version.txt"),
    ];

    for file_path in &version_files {
        if let Err(e) = fs::write(file_path, version) {
            eprintln!("Warning: Failed to write to {:?}: {}", file_path, e);
        } else {
            if let Some(name) = file_path.file_name() {
                println!("Updated {:?} to {}.", name, version);
            }
        }
    }

    println!(
        "Success! Model weights uploaded successfully to GitHub Release {}.",
        version
    );
    println!("Remember to commit and push the updated version files so CI and the web UI use the new release.");
}
