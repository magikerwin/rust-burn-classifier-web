mod model;

use wasm_bindgen::prelude::*;
use burn::{
    backend::NdArray,
    module::Module,
    prelude::*,
    record::{BinBytesRecorder, FullPrecisionSettings, Recorder},
    tensor::activation::softmax,
};
use crate::model::Model;

#[wasm_bindgen]
pub struct MnistPredictor {
    model: Model<NdArray>,
    device: <NdArray as Backend>::Device,
}

#[wasm_bindgen]
impl MnistPredictor {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        console_error_panic_hook::set_once();
        let device = Default::default();
        
        let bytes = include_bytes!("../../target/mnist-model/model.bin");
        let recorder = BinBytesRecorder::<FullPrecisionSettings>::default();
        let record = recorder.load(bytes.to_vec(), &device)
            .expect("Failed to load embedded model weights");
            
        let model = Model::<NdArray>::new(&device).load_record(record);
        
        Self { model, device }
    }
    
    pub fn predict(&self, raw_image: &[f32]) -> Result<Vec<f32>, JsValue> {
        if raw_image.len() != 784 {
            return Err(JsValue::from_str("Input must be exactly 784 pixels"));
        }
        
        let mut image_array = [0.0f32; 784];
        image_array.copy_from_slice(raw_image);
        
        // Self-healing: if frontend sends normalized [0, 1] pixels, scale them to [0, 255]
        let max_pixel = image_array.iter().fold(0.0f32, |m, &x| m.max(x));
        if max_pixel <= 1.0 && max_pixel > 0.0 {
            for val in image_array.iter_mut() {
                *val *= 255.0;
            }
        }
        
        // 1. Convert raw array to 4D Tensor [1, 1, 28, 28]
        let input = Tensor::<NdArray, 1>::from_floats(image_array, &self.device)
            .reshape([1, 1, 28, 28]);
            
        // 2. Run inference
        let output = self.model.forward(input);
        
        // 3. Apply softmax
        let probs = softmax(output, 1);
        let probs_vec = probs.into_data().into_vec::<f32>()
            .map_err(|e| JsValue::from_str(&format!("Failed to extract tensor data: {:?}", e)))?;
            
        Ok(probs_vec)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    #[wasm_bindgen_test]
    fn test_predictor_creation_and_predict() {
        let predictor = MnistPredictor::new();
        let dummy_image = [0.0f32; 784];
        let probs = predictor.predict(&dummy_image).unwrap();
        assert_eq!(probs.len(), 10);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4);
    }
}
