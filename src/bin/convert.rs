use burn::{
    backend::NdArray,
    module::Module,
    record::{BinFileRecorder, FullPrecisionSettings, CompactRecorder, Recorder},
};
use rust::model::Model; // Expose and use the rust library model module

fn main() {
    let device = Default::default();
    let artifact_dir = "./target/mnist-model";
    
    println!("Loading model from compact recorder (model.mpk)...");
    let recorder = CompactRecorder::new();
    let record = recorder
        .load(format!("{artifact_dir}/model").into(), &device)
        .expect("Failed to load model parameters");
    
    let model = Model::<NdArray>::new(&device, 10).load_record(record);
    
    println!("Saving model with BinFileRecorder (model.bin)...");
    model
        .save_file(format!("{artifact_dir}/model"), &BinFileRecorder::<FullPrecisionSettings>::new())
        .expect("Failed to save model parameters to binary");
        
    println!("Conversion completed successfully!");
}
