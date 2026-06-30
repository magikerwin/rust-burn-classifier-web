use crate::model::Model;
use burn::{
    backend::NdArray,
    prelude::*,
    record::{CompactRecorder, Recorder},
    tensor::activation::softmax,
};

/// Loads the trained model weights from the artifact directory and returns the Model.
pub fn load_model(artifact_dir: &str, device: &<NdArray as Backend>::Device) -> Model<NdArray> {
    let recorder = CompactRecorder::new();
    
    // Load the saved weights using the compact recorder
    let record = recorder
        .load(format!("{artifact_dir}/model").into(), device)
        .expect("Failed to load model parameters");

    // Reconstruct the model architecture and load the weights
    Model::<NdArray>::new(device).load_record(record)
}

/// Performs prediction on a raw 28x28 flattened image array.
pub fn predict(model: &Model<NdArray>, raw_image: [f32; 784], device: &<NdArray as Backend>::Device) -> usize {
    // 1. Convert the raw array into a 4D Burn Tensor: shape [1, 1, 28, 28]
    let input = Tensor::<NdArray, 1>::from_floats(raw_image, device)
        .reshape([1, 1, 28, 28]);

    // 2. Perform the forward pass (inference)
    let output = model.forward(input);

    // 3. Find the index with the highest probability value (ArgMax)
    let predicted = output.argmax(1);

    // 4. Extract the index value as a standard scalar integer
    let value = predicted.into_scalar() as usize;

    value
}

/// Performs prediction on a raw 28x28 flattened image array, returning both the predicted digit and all 10 softmax probabilities.
pub fn predict_probabilities(
    model: &Model<NdArray>,
    raw_image: [f32; 784],
    device: &<NdArray as Backend>::Device,
) -> (usize, Vec<f32>) {
    // 1. Convert raw array to 4D Tensor [1, 1, 28, 28]
    let input = Tensor::<NdArray, 1>::from_floats(raw_image, device)
        .reshape([1, 1, 28, 28]);

    // 2. Perform forward pass
    let output = model.forward(input);

    // 3. Extract the predicted class index using argmax
    let predicted = output.clone().argmax(1).into_scalar() as usize;

    // 4. Run softmax to get probability scores (values between 0.0 and 1.0)
    let probs = softmax(output, 1);
    let probs_vec = probs.into_data().into_vec::<f32>().expect("Failed to extract probability data");

    (predicted, probs_vec)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_predict() {
        let device = Default::default();
        let model = Model::<NdArray>::new(&device);

        // Run prediction on a dummy image (all zeros)
        let dummy_image = [0.0f32; 784];
        let predicted_digit = predict(&model, dummy_image, &device);

        // The predicted digit should be a valid class index between 0 and 9
        assert!(predicted_digit < 10);
    }

    #[test]
    fn test_predict_probabilities() {
        let device = Default::default();
        let model = Model::<NdArray>::new(&device);

        let dummy_image = [0.0f32; 784];
        let (predicted_digit, probabilities) = predict_probabilities(&model, dummy_image, &device);

        // 1. Predicted digit is valid
        assert!(predicted_digit < 10);

        // 2. We get exactly 10 probabilities (one for each digit 0-9)
        assert_eq!(probabilities.len(), 10);

        // 3. Softmax properties: sum of probabilities must be close to 1.0
        let sum: f32 = probabilities.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }
}
