use wasm_bindgen::prelude::*;
use burn::{
    backend::NdArray,
    module::Module,
    prelude::*,
    record::{BinBytesRecorder, FullPrecisionSettings, Recorder},
    tensor::activation::softmax,
};
use model_shared::Model;

const MODEL_VERSION: &str = include_str!("../weights-version.txt");

#[wasm_bindgen]
pub fn get_compiled_model_version() -> String {
    MODEL_VERSION.trim().to_string()
}



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
        
        let bytes = include_bytes!(concat!(env!("OUT_DIR"), "/mnist-model.bin"));
        let recorder = BinBytesRecorder::<FullPrecisionSettings>::default();
        let record = recorder.load(bytes.to_vec(), &device)
            .expect("Failed to load embedded model weights");
            
        let model = Model::<NdArray>::new(&device, 10).load_record(record);
        
        Self { model, device }
    }
    
    pub fn predict(&self, raw_image: &[f32]) -> Result<Vec<f32>, JsValue> {
        predict_internal(&self.model, &self.device, raw_image, false) // MNIST expects [0, 255]
    }
}

#[wasm_bindgen]
pub struct QuickdrawPredictor {
    model: Model<NdArray>,
    device: <NdArray as Backend>::Device,
}

#[wasm_bindgen]
impl QuickdrawPredictor {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        console_error_panic_hook::set_once();
        let device = Default::default();
        
        let bytes = include_bytes!(concat!(env!("OUT_DIR"), "/quickdraw-model.bin"));
        let recorder = BinBytesRecorder::<FullPrecisionSettings>::default();
        let record = recorder.load(bytes.to_vec(), &device)
            .expect("Failed to load embedded model weights");
            
        let model = Model::<NdArray>::new(&device, 25).load_record(record);
        
        Self { model, device }
    }
    
    pub fn predict(&self, raw_image: &[f32]) -> Result<Vec<f32>, JsValue> {
        predict_internal(&self.model, &self.device, raw_image, true) // Quickdraw expects [0, 1]
    }
}

fn predict_internal(
    model: &Model<NdArray>,
    device: &<NdArray as Backend>::Device,
    raw_image: &[f32],
    normalize_to_one: bool,
) -> Result<Vec<f32>, JsValue> {
    if raw_image.len() != 784 {
        return Err(JsValue::from_str("Input must be exactly 784 pixels"));
    }
    
    let mut image_array = [0.0f32; 784];
    image_array.copy_from_slice(raw_image);
    
    let max_pixel = image_array.iter().fold(0.0f32, |m, &x| m.max(x));
    
    if normalize_to_one {
        // Quickdraw expects [0.0..1.0]. If input is in [0..255] range, normalize it.
        if max_pixel > 1.0 {
            for val in image_array.iter_mut() {
                *val /= 255.0;
            }
        }
    } else {
        // MNIST expects [0.0..255.0]. If input is in [0..1] range, scale it.
        if max_pixel <= 1.0 && max_pixel > 0.0 {
            for val in image_array.iter_mut() {
                *val *= 255.0;
            }
        }
    }
    
    // 1. Convert raw array to 4D Tensor [1, 1, 28, 28]
    let input = Tensor::<NdArray, 1>::from_floats(image_array, device)
        .reshape([1, 1, 28, 28]);
        
    // 2. Run inference
    let output = model.forward(input);
    
    // 3. Apply softmax
    let probs = softmax(output, 1);
    let probs_vec = probs.into_data().into_vec::<f32>()
        .map_err(|e| JsValue::from_str(&format!("Failed to extract tensor data: {:?}", e)))?;
        
    Ok(probs_vec)
}


#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    #[wasm_bindgen_test]
    fn test_mnist_predictor_creation_and_predict() {
        let predictor = MnistPredictor::new();
        let dummy_image = [0.0f32; 784];
        let probs = predictor.predict(&dummy_image).unwrap();
        assert_eq!(probs.len(), 10);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4);
    }

    #[wasm_bindgen_test]
    fn test_quickdraw_predictor_creation_and_predict() {
        let predictor = QuickdrawPredictor::new();
        let dummy_image = [0.0f32; 784];
        let probs = predictor.predict(&dummy_image).unwrap();
        assert_eq!(probs.len(), 25);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4);
    }
}
