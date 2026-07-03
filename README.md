# 🔢 MNIST Digit Classifier — Burn (Rust)

> An interactive MNIST handwritten digit classifier built with the [Burn](https://burn.dev/) deep learning framework in Rust. Train a CNN model, run inference from the CLI, or draw digits in the browser!
> 
> 🚀 **[Try the Live WebAssembly Demo here!](https://magikerwin.github.io/rust-burn-classifier-web/)**

![image](assets/web_demo_mnist.webp)

## ✨ Features

- **CNN Architecture** — Conv2d → MaxPool → Conv2d → MaxPool → FC → FC with dropout
- **Interactive Web Demo** — Draw digits on a canvas and get real-time predictions
- **WebAssembly Client-Side Deployment** — Run inference directly in the browser via WASM with no backend server required
- **CLI Inference** — Predict digits with ASCII art visualization
- **Fully in Rust** — Training, inference, and web frontend in a unified workspace codebase

## 🏗️ Model Architecture

```
Input [1×28×28]
  → Conv2d(1→8, 3×3, same) → ReLU → MaxPool(2×2)    → [8×14×14]
  → Conv2d(8→16, 3×3, same) → ReLU → MaxPool(2×2)   → [16×7×7]
  → Flatten                                           → [784]
  → Linear(784→128) → ReLU → Dropout(0.5)
  → Linear(128→10) → Softmax
```

## 🚀 Getting Started

### Train the Model

By default, training runs on the CPU (using the `NdArray` backend):

```sh
cargo run --release
```

To train using your GPU (using the cross-platform `Wgpu` backend):

```sh
cargo run --release -- --gpu
```

To train doodle classification on the Google **Quick, Draw!** dataset (25 classes, downloads dataset dynamically):

```sh
# Train on CPU
cargo run --release -- --dataset quickdraw

# Train on GPU (cross-platform Wgpu backend)
cargo run --release -- --dataset quickdraw --gpu
```

> **Note:** Always use `--release` for optimized tensor math performance.

#### 📊 Results

After 5 epochs of training, the CNN model achieves the following validation metrics:

| Dataset | Validation Accuracy | Validation Loss |
|---|---|---|
| **MNIST** (10 classes) | `~97%+` | `~0.10` |
| **Quick, Draw!** (25 classes) | `~80% - 85%` | `~0.50 - 0.70` |

<details>
<summary>📈 View MNIST Training Progress Curve</summary>

![MNIST Training Curve](assets/training_mnist.png)

</details>

### Run Tests

```sh
cargo test
```

### Run CLI Inference

Once trained, predict digits from the MNIST test set:

```sh
cargo run --release -- --predict
```

To predict doodles from the Quick, Draw! test dataset:

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

### Run Interactive Web Server (Axum backend)

Start the browser-based drawing pad backed by the Rust Axum server. The web server dynamically configures its UI categories and canvas configuration depending on which dataset you select:

- **Run with MNIST Digits (default)**:
  ```sh
  cargo run --release -- --serve
  ```
  Then open **[http://127.0.0.1:3000](http://127.0.0.1:3000)** to draw digits (0-9) and run predictions.

- **Run with Quick, Draw! Doodles**:
  ```sh
  cargo run --release -- --serve --dataset quickdraw
  ```
  Then open **[http://127.0.0.1:3000](http://127.0.0.1:3000)** to draw and predict doodles (25 classes).

### Run Client-Side WebAssembly App (WASM)

This project compiles the trained models into WebAssembly to run inference fully client-side. To avoid Git binary bloat, the model weights are **completely decoupled** from Git history and LFS:
- **Locally**: `build.rs` automatically copies fresh weights from `target/` on compilation.
- **In CI (GitHub Actions)**: `build.rs` automatically downloads stable weights from GitHub Releases using `curl` at build time.

#### 1. Compile the WASM bundle locally
Make sure you have trained the models locally first. Then run:

1. **Install wasm-pack**:
   ```sh
   cargo install wasm-pack
   ```

2. **Build the WebAssembly module**:
   ```sh
   wasm-pack build web --target web --out-dir ../docs/pkg
   ```

3. **Serve the frontend locally**:
   Install and run `basic-http-server`:
   ```sh
   cargo install basic-http-server
   basic-http-server docs
   ```
   Navigate to **[http://localhost:4000](http://localhost:4000)** to run serverless inference.

#### 2. Automatic Deployments & Release Management
The project includes a GitHub Action workflow that automatically compiles and deploys your static page to the `gh-pages` branch whenever you push changes to `master`.

To update the online weights used by the CI runner:
1. Ensure the GitHub CLI (`gh`) is installed and authenticated.
2. Run the helper script to create/update release `v1.0.0` and upload your local weights:
   ```powershell
   ./publish-weights.ps1
   ```
3. Push your code changes to GitHub:
   ```sh
   git push origin master
   ```
4. Verify your repository settings under **Settings -> Pages** is configured to serve from the **`gh-pages`** branch (root).

## 🎨 Quick, Draw! Classification Details

This project supports doodle classification using the public Google **[Quick, Draw! Dataset](https://github.com/googlecreativelab/quickdraw-dataset)**. 

### Selected Categories (25 classes)
Rather than training on all 345 categories (which totals 39 GB of raw bitmap data), we train on a curated subset of **25 diverse and easily sketchable classes**:
* **Nature/Weather**: `sun`, `moon`, `star`, `tree`, `flower`
* **Animals**: `cat`, `dog`, `fish`, `butterfly`
* **Common Objects**: `cup`, `key`, `umbrella`, `hat`, `clock`, `envelope`, `toothbrush`
* **Structures/Vehicles**: `house`, `car`
* **Shapes/Drawings**: `circle`, `triangle`, `square`, `smiley face`
* **Clothing**: `pants`, `t-shirt`
* **Food**: `apple`

### Key Design Considerations
1. **Compute & Storage Constraints**: Restricting the dataset to 25 categories reduces the data footprint to ~250 MB, allowing the dataset to fit entirely in memory and train in minutes on a consumer GPU.
2. **Model Capacity**: A simple CNN architecture struggles to differentiate 345 classes, but easily achieves high accuracy on 25 distinct shapes.
3. **Canvas Drawability**: These categories are simple and iconic enough for humans to draw clearly on a small `28x28` pixel canvas.
4. **License & Privacy**: The dataset is safe and free to use under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** license, and contains no personally identifiable information (PII).

## 📚 References

- [Burn — Deep Learning Framework for Rust](https://burn.dev/)
- [tracel-ai/burn MNIST example](https://github.com/tracel-ai/burn/blob/main/examples/mnist/examples/mnist.rs)

## 📄 License

This project is licensed under the [MIT License](./LICENSE).
