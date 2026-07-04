# 🔢 MNIST & Quick, Draw! Classifier — Burn (Rust)

> An interactive handwritten digit and doodle classifier built with the [Burn](https://burn.dev/) deep learning framework in Rust. Train a CNN model, run inference from the CLI, or draw in the browser!
>
> 🚀 **[Try the Live WebAssembly Demo!](https://magikerwin.github.io/rust-burn-classifier-web/)**

![image](assets/web_demo_mnist.webp)

---

## 📑 Table of Contents

- [Features](#-features)
- [Model Architecture](#️-model-architecture)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
  - [Train the Model](#train-the-model)
  - [Run Tests](#run-tests)
  - [CLI Inference](#cli-inference)
  - [Interactive Web Server (Axum)](#interactive-web-server-axum-backend)
  - [Client-Side WebAssembly App](#client-side-webassembly-app-wasm)
- [Quick, Draw! Classification Details](#-quick-draw-classification-details)
- [References](#-references)
- [License](#-license)

---

## ✨ Features

- **CNN Architecture** — Conv2d → MaxPool → Conv2d → MaxPool → FC → FC with dropout
- **Interactive Web Demo** — Draw on a canvas and get real-time predictions
- **WebAssembly Client-Side Inference** — Runs entirely in the browser via WASM, no backend required
- **CLI Inference** — Predict with ASCII art visualization
- **Fully in Rust** — Training, inference, and web frontend in a unified workspace

---

## 🏗️ Model Architecture

```
Input [1×28×28]
  → Conv2d(1→8, 3×3, same) → ReLU → MaxPool(2×2)    → [8×14×14]
  → Conv2d(8→16, 3×3, same) → ReLU → MaxPool(2×2)   → [16×7×7]
  → Flatten                                           → [784]
  → Linear(784→128) → ReLU → Dropout(0.5)
  → Linear(128→10) → Softmax
```

---

## 📁 Project Structure

```
rust-burn-classifier-web/
├── web/                    # Rust WASM crate (wasm-pack entry point)
├── src/                    # Training & CLI inference (Burn backend)
│   ├── main.rs
│   ├── model.rs            # CNN model definition
│   ├── training.rs
│   └── ...
├── docs/                   # Static web frontend (served by GitHub Pages)
│   ├── index.html          # Single-page drawing app
│   └── pkg/                # Compiled WASM output (gitignored, built by CI)
├── assets/                 # README images and training curves
├── build.rs                # Copies model weights at build time
├── publish-weights.ps1     # Helper script to upload weights to GitHub Releases
└── .github/workflows/
    └── deploy.yml          # CI: build WASM → assert → deploy → verify
```

> **Git Branches:**
> - `master` — The only branch you need. All development happens here. Compiled binaries are gitignored and built fresh by CI on every deploy.

---


## 🚀 Getting Started

### Train the Model

By default, training runs on the CPU (`NdArray` backend):

```sh
cargo run --release
```

To train on your GPU (`Wgpu` backend):

```sh
cargo run --release -- --gpu
```

To train on the Google **Quick, Draw!** dataset (25 classes, downloads dynamically):

```sh
# CPU
cargo run --release -- --dataset quickdraw

# GPU
cargo run --release -- --dataset quickdraw --gpu
```

> **Note:** Always use `--release` for optimized tensor math performance.

#### 📊 Results

After 5 epochs of training:

| Dataset | Validation Accuracy | Validation Loss |
|---|---|---|
| **MNIST** (10 classes) | `~97%+` | `~0.10` |
| **Quick, Draw!** (25 classes) | `~80% - 85%` | `~0.50 - 0.70` |

<details>
<summary>📈 View MNIST Training Progress Curve</summary>

![MNIST Training Curve](assets/training_mnist.png)

</details>

---

### Run Tests

```sh
cargo test
```

---

### CLI Inference

Once trained, predict from the MNIST test set:

```sh
cargo run --release -- --predict
```

Predict from the Quick, Draw! test set:

```sh
cargo run --release -- --predict --dataset quickdraw
```

<details>
<summary>📝 Example Output (MNIST)</summary>

```text
Loading model for inference...

Input Image:
      ######                
      ################      
      ################      
           ###########      
                  ####      
                 ####       
                 ####       
                ####        
                ####        
               ####         
               ###          
              ####          
             ####           
            #####           
            ####            
           #####            
           ####             
          #####             
          #####             
          ####              
                            
Target Label (Ground Truth): 7
Top Predictions:
  1. 7            : 99.42%
  2. 9            : 0.35%
  3. 2            : 0.11%
```

</details>

<details>
<summary>📝 Example Output (Quick, Draw!)</summary>

```text
Loading model for inference (dataset: quickdraw)...

Input Image:
         #          
        ###         
       #####        
      #######       
     #########      
    ###########     
   #############    
  ###############   
 #################  

Target Label (Ground Truth): triangle
Top Predictions:
  1. triangle     : 96.81%
  2. mountain     : 2.14%
  3. house        : 0.45%
```

</details>

---

### Interactive Web Server (Axum backend)

Start the browser-based drawing pad backed by the Rust Axum server:

- **MNIST Digits (default)**:
  ```sh
  cargo run --release -- --serve
  ```
  Open **[http://127.0.0.1:3000](http://127.0.0.1:3000)** to draw digits (0–9).

- **Quick, Draw! Doodles**:
  ```sh
  cargo run --release -- --serve --dataset quickdraw
  ```
  Open **[http://127.0.0.1:3000](http://127.0.0.1:3000)** to draw and predict doodles (25 classes).

---

### Client-Side WebAssembly App (WASM)

The trained models compile to WebAssembly for fully client-side inference. Model weights are **decoupled from Git history** to avoid binary bloat:
- **Locally**: `build.rs` copies fresh weights from `target/` at compile time.
- **In CI**: `build.rs` downloads stable weights from GitHub Releases via `curl`.

#### 1. Build the WASM bundle locally

Make sure you have trained the models first, then:

1. **Install wasm-pack**:
   ```sh
   cargo install wasm-pack
   ```

2. **Build the WebAssembly module**:
   ```sh
   wasm-pack build web --target web --out-dir ../docs/pkg
   ```

3. **Serve locally**:
   ```sh
   cargo install basic-http-server
   basic-http-server docs
   ```
   Navigate to **[http://localhost:4000](http://localhost:4000)**.

#### 2. Automatic Deployments & Release Management

The CI workflow (`.github/workflows/deploy.yml`) automatically builds and deploys to GitHub Pages on every push to `master`. It also verifies the live WASM URL after deploying.

To update the model weights used by the CI runner:

1. Ensure the GitHub CLI (`gh`) is installed and authenticated:
   - **Install** (Windows): `winget install --id GitHub.cli` (restart VS Code after)
   - **Authenticate**: `gh auth login` → GitHub.com → HTTPS → browser

2. Upload your local weights to a GitHub Release:
   ```powershell
   # Default v1.0.0
   powershell -ExecutionPolicy Bypass -File ./publish-weights.ps1

   # Custom version tag
   powershell -ExecutionPolicy Bypass -File ./publish-weights.ps1 -Version v2.0.0
   ```

3. Trigger the deployment:
   - **Code changes**: `git push origin master`
   - **Weights only**: Go to **Actions → Deploy WebAssembly to GitHub Pages → Run workflow**

4. Verify your repository settings under **Settings → Pages → Build and deployment**:
   - **Source**: `GitHub Actions`

   > ℹ️ The workflow uses the official GitHub Pages API (`upload-pages-artifact` + `deploy-pages`), so no `gh-pages` branch is needed.

---

## 🎨 Quick, Draw! Classification Details

This project supports doodle classification using the public Google **[Quick, Draw! Dataset](https://github.com/googlecreativelab/quickdraw-dataset)**.

### Selected Categories (25 classes)

Rather than training on all 345 categories (39 GB of raw data), we train on a curated subset of **25 diverse and sketchable classes**:

| Group | Classes |
|---|---|
| Nature / Weather | `sun`, `moon`, `star`, `tree`, `flower` |
| Animals | `cat`, `dog`, `fish`, `butterfly` |
| Common Objects | `cup`, `key`, `umbrella`, `hat`, `clock`, `envelope`, `toothbrush` |
| Structures / Vehicles | `house`, `car` |
| Shapes | `circle`, `triangle`, `square`, `smiley face` |
| Clothing | `pants`, `t-shirt` |
| Food | `apple` |

### Key Design Considerations

1. **Compute & Storage**: 25 classes → ~250 MB footprint, fits in memory, trains in minutes on a consumer GPU.
2. **Model Capacity**: A simple CNN easily achieves high accuracy on 25 distinct shapes vs. struggling with all 345.
3. **Canvas Drawability**: These shapes are simple and iconic enough to draw clearly on a `28×28` canvas.
4. **License & Privacy**: CC BY 4.0, no personally identifiable information (PII).

---

## 📚 References

- [Burn — Deep Learning Framework for Rust](https://burn.dev/)
- [tracel-ai/burn MNIST example](https://github.com/tracel-ai/burn/blob/main/examples/mnist/examples/mnist.rs)

---

## 📄 License

This project is licensed under the [MIT License](./LICENSE).
