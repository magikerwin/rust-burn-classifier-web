mod model;
mod data;
mod training;
mod inference;
mod quickdraw;


use burn::{
    backend::{Autodiff, NdArray},
    data::dataset::{vision::MnistDataset, Dataset},
    optim::AdamConfig,
};
use crate::training::{train, TrainingConfig};
use crate::inference::{load_model, predict_probabilities};

use axum::{
    extract::State,
    response::Html,
    routing::{get, post},
    Json, Router,
};
use std::sync::Arc;
use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize)]
struct AppConfig {
    dataset: String,
    classes: Vec<String>,
}

/// Shared state across the web server requests
#[derive(Clone)]
struct AppState {
    model: Arc<std::sync::Mutex<crate::model::Model<NdArray>>>,
    device: <NdArray as burn::prelude::Backend>::Device,
    config: AppConfig,
}

/// JSON payload structure for /predict requests
#[derive(Deserialize)]
struct PredictRequest {
    image: Vec<f32>,
}

/// JSON response structure for /predict responses
#[derive(Serialize)]
struct PredictResponse {
    prediction: usize,
    probabilities: Vec<f32>,
}

/// Handler that serves the HTML drawing canvas frontend page
async fn index_handler() -> Html<&'static str> {
    Html(include_str!("index.html"))
}

/// Handler that handles post requests to run model predictions on drawing inputs
async fn predict_handler(
    State(state): State<AppState>,
    Json(payload): Json<PredictRequest>,
) -> Json<PredictResponse> {
    // 1. Convert the incoming Vec<f32> to a fixed [f32; 784] array
    let mut raw_image = [0.0f32; 784];
    let len = payload.image.len().min(784);
    raw_image[..len].copy_from_slice(&payload.image[..len]);

    // Self-healing: if the frontend sends [0, 1] normalized pixels, scale them to [0, 255]
    let max_pixel = raw_image.iter().fold(0.0f32, |m, &x| m.max(x));
    if max_pixel <= 1.0 && max_pixel > 0.0 {
        for val in raw_image.iter_mut() {
            *val *= 255.0;
        }
    }

    // 2. Perform prediction and extract softmax probabilities
    let model = state.model.lock().unwrap();
    
    let (prediction, probabilities) = predict_probabilities(&model, raw_image, &state.device);

    Json(PredictResponse {
        prediction,
        probabilities,
    })
}

/// Handler that serves the dataset and labels configuration
async fn config_handler(
    State(state): State<AppState>,
) -> Json<AppConfig> {
    Json(state.config.clone())
}

#[tokio::main]
async fn main() {
    // CLI argument parsing
    let args: Vec<String> = std::env::args().collect();
    let run_inference = args.contains(&"--predict".to_string());
    let run_server = args.contains(&"--serve".to_string());
    let run_gpu = args.contains(&"--gpu".to_string());

    let dataset_arg = args.iter()
        .position(|arg| arg == "--dataset")
        .and_then(|pos| args.get(pos + 1))
        .map(|s| s.as_str())
        .unwrap_or("mnist");

    let num_classes = if dataset_arg == "quickdraw" { quickdraw::QUICKDRAW_CLASSES.len() } else { 10 };
    let artifact_dir = if dataset_arg == "quickdraw" { "./target/quickdraw-model" } else { "./target/mnist-model" };

    if run_server {
        // ==========================================
        // BRANCH A: RUN INTERACTIVE WEB SERVER
        // ==========================================
        println!("Loading model for web server...");
        let device = Default::default(); // NdArray CPU device
        let model = Arc::new(std::sync::Mutex::new(load_model(artifact_dir, &device, num_classes)));
        
        let classes = if dataset_arg == "quickdraw" {
            quickdraw::QUICKDRAW_CLASSES.iter().map(|&s| s.to_string()).collect()
        } else {
            (0..10).map(|i| i.to_string()).collect()
        };
        let config = AppConfig {
            dataset: dataset_arg.to_string(),
            classes,
        };
        let state = AppState { model, device, config };

        // Construct Axum application routing
        let app = Router::new()
            .route("/", get(index_handler))
            .route("/predict", post(predict_handler))
            .route("/api/config", get(config_handler))
            .with_state(state);

        // Bind TCP listener to port 3000
        let listener = tokio::net::TcpListener::bind("127.0.0.1:3000")
            .await
            .unwrap();
        println!("\n==================================================");
        println!("   Burn MNIST Drawing App Web Server Running!");
        println!("   Open your browser to: http://127.0.0.1:3000");
        println!("==================================================\n");

        axum::serve(listener, app).await.unwrap();

    } else if run_inference {
        // ==========================================
        // BRANCH B: RUN CLI PREDICTION (TEST SAMPLE)
        // ==========================================
        println!("Loading model for inference (dataset: {})...", dataset_arg);
        let device = Default::default();
        let model = load_model(artifact_dir, &device, num_classes);

        // Fetch a sample from the selected dataset
        let (flattened_image, class_name) = if dataset_arg == "quickdraw" {
            let test_dataset = quickdraw::QuickDrawDataset::new(false, 1);
            let sample = test_dataset.get(0).expect("Failed to get sample");
            let mut flattened_image = [0.0f32; 784];
            for i in 0..28 {
                for j in 0..28 {
                    flattened_image[i * 28 + j] = sample.image[i][j];
                }
            }
            let label = sample.label as usize;
            (flattened_image, quickdraw::QUICKDRAW_CLASSES[label].to_string())
        } else {
            let test_dataset = MnistDataset::test();
            let sample = test_dataset.get(0).expect("Failed to get sample");
            let mut flattened_image = [0.0f32; 784];
            for i in 0..28 {
                for j in 0..28 {
                    flattened_image[i * 28 + j] = sample.image[i][j];
                }
            }
            let label = sample.label as usize;
            (flattened_image, label.to_string())
        };

        // Draw a simple ASCII art representing the input
        println!("\nInput Image:");
        for i in 0..28 {
            for j in 0..28 {
                if flattened_image[i * 28 + j] > 0.5 {
                    print!("#");
                } else if flattened_image[i * 28 + j] > 0.1 {
                    print!(".");
                } else {
                    print!(" ");
                }
            }
            println!();
        }

        // Perform prediction and print top 3 probabilities
        let (_predicted_digit, probabilities) = predict_probabilities(&model, flattened_image, &device);
        
        let mut prob_indices: Vec<(usize, f32)> = probabilities
            .into_iter()
            .enumerate()
            .collect();
        prob_indices.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        println!("\nTarget Label (Ground Truth): {}", class_name);
        println!("Top Predictions:");
        for (i, (idx, prob)) in prob_indices.iter().take(3).enumerate() {
            let name = if dataset_arg == "quickdraw" {
                quickdraw::QUICKDRAW_CLASSES[*idx].to_string()
            } else {
                idx.to_string()
            };
            println!("  {}. {:<12} : {:.2}%", i + 1, name, prob * 100.0);
        }

    } else {
        // ==========================================
        // BRANCH C: RUN TRAINING LOOP
        // ==========================================
        let config = TrainingConfig::new(AdamConfig::new());

        if dataset_arg == "quickdraw" {
            let train_dataset = quickdraw::QuickDrawDataset::new(true, quickdraw::TRAIN_SAMPLES_PER_CLASS);
            let valid_dataset = quickdraw::QuickDrawDataset::new(false, quickdraw::VAL_SAMPLES_PER_CLASS);

            if run_gpu {
                println!("Starting Quick, Draw! training on GPU (WGPU backend)...");
                train::<Autodiff<burn::backend::Wgpu>, _, _>(
                    artifact_dir,
                    config,
                    burn::backend::wgpu::WgpuDevice::default(),
                    train_dataset,
                    valid_dataset,
                    num_classes,
                );
            } else {
                println!("Starting Quick, Draw! training on CPU (NdArray backend)...");
                train::<Autodiff<NdArray>, _, _>(
                    artifact_dir,
                    config,
                    Default::default(),
                    train_dataset,
                    valid_dataset,
                    num_classes,
                );
            }
        } else {
            let train_dataset = MnistDataset::train();
            let valid_dataset = MnistDataset::test();

            if run_gpu {
                println!("Starting MNIST training on GPU (WGPU backend)...");
                train::<Autodiff<burn::backend::Wgpu>, _, _>(
                    artifact_dir,
                    config,
                    burn::backend::wgpu::WgpuDevice::default(),
                    train_dataset,
                    valid_dataset,
                    num_classes,
                );
            } else {
                println!("Starting MNIST training on CPU (NdArray backend)...");
                train::<Autodiff<NdArray>, _, _>(
                    artifact_dir,
                    config,
                    Default::default(),
                    train_dataset,
                    valid_dataset,
                    num_classes,
                );
            }
        }
        println!("Training finished! Model saved successfully to {}", artifact_dir);
    }
}
