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

```sh
cargo run --release
```

> **Note:** Always use `--release` for optimized tensor math performance.

#### 📊 Results

After 5 epochs of training, the CNN model achieves the following on the MNIST validation set:

| Metric              | Value    |
|---------------------|----------|
| **Validation Accuracy** | `~97%+`  |
| **Validation Loss**     | `~0.10`  |

![image](assets/training_mnist.png)

### Run Tests

```sh
cargo test
```

### Run CLI Inference

Once trained, predict digits from the MNIST test set:

```sh
cargo run --release -- --predict
```

<details>
<summary>📝 Example Output</summary>

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
Model Prediction           : 7
```

</details>

### Run Interactive Web Server (Axum backend)

Start the browser-based drawing pad backed by the Rust Axum server:

```sh
cargo run --release -- --serve
```

Then open **[http://127.0.0.1:3000](http://127.0.0.1:3000)** to draw digits and see predictions served via HTTP.

### Run Client-Side WebAssembly App (WASM)

Compile the model to WebAssembly to run inference fully inside the browser client-side (perfect for static hosting like GitHub Pages):

1. **Install the WebAssembly packager**:
   ```sh
   cargo install wasm-pack
   ```

2. **Convert trained weights to binary format**:
   ```sh
   cargo run --bin convert
   ```

3. **Build the WASM module into the static frontend**:
   ```sh
   wasm-pack build web --target web --out-dir ../docs/pkg
   ```

4. **Serve the static frontend locally**:
   Install and run `basic-http-server`:
   ```sh
   cargo install basic-http-server
   basic-http-server docs
   ```
   Then navigate to **[http://localhost:4000](http://localhost:4000)** to draw digits and run serverless, client-side inference!

## 📚 References

- [Burn — Deep Learning Framework for Rust](https://burn.dev/)
- [tracel-ai/burn MNIST example](https://github.com/tracel-ai/burn/blob/main/examples/mnist/examples/mnist.rs)

## 📄 License

This project is licensed under the [MIT License](./LICENSE).
