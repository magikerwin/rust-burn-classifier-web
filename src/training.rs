// TODO: Remove this warning suppression once training is integrated into main.rs.
#![allow(dead_code)]

use crate::{
    data::{MnistBatch, MnistBatcher},
    model::Model,
};
use burn::{
    data::{
        dataloader::DataLoaderBuilder,
        dataset::Dataset,
    },
    nn::loss::CrossEntropyLossConfig,
    optim::AdamConfig,
    prelude::*,
    record::CompactRecorder,
    tensor::backend::AutodiffBackend,
    train::{
        metric::{AccuracyMetric, LossMetric},
        ClassificationOutput, LearnerBuilder, TrainOutput, TrainStep, ValidStep,
    },
};

/// Configuration struct holding the training hyperparameters.
/// Deriving `Config` allows saving and loading this configuration as a JSON file easily.
#[derive(Config)]
pub struct TrainingConfig {
    #[config(default = 5)]
    pub num_epochs: usize,      // Number of epochs to train for
    #[config(default = 64)]
    pub batch_size: usize,      // Number of samples in each batch
    #[config(default = 4)]
    pub num_workers: usize,     // Number of parallel threads used for data loading
    #[config(default = 42)]
    pub seed: u64,              // Seed for random number generators (reproducibility)
    pub optimizer: AdamConfig,  // Optimizer configuration (e.g. learning rate, betas)
}

/// Orchestrates the training pipeline. Decoupled from specific datasets using generic types.
pub fn train<B: AutodiffBackend, D1, D2>(
    artifact_dir: &str,
    config: TrainingConfig,
    device: B::Device,
    train_dataset: D1,
    valid_dataset: D2,
) where
    D1: Dataset<burn::data::dataset::vision::MnistItem> + 'static,
    D2: Dataset<burn::data::dataset::vision::MnistItem> + 'static,
{
    // Ensure the output directory exists and save the execution config as JSON
    std::fs::create_dir_all(artifact_dir).ok();
    config.save(format!("{artifact_dir}/config.json")).expect("Save config failed");

    // Set the backend random seed for reproducible initialization and shuffling
    B::seed(config.seed);

    // Initialize the batcher for training data (needs autodiff backend B to track gradients)
    let batcher_train = MnistBatcher::<B>::new(device.clone());
    
    // Initialize the batcher for validation data (uses B::InnerBackend which skips tracking gradients)
    let batcher_valid = MnistBatcher::<B::InnerBackend>::new(device.clone());

    // Build the training DataLoader
    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(train_dataset);

    // Build the validation DataLoader
    let dataloader_valid = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(valid_dataset);

    // Configure and build the training driver (Learner)
    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(AccuracyMetric::new())  // Track training accuracy
        .metric_valid_numeric(AccuracyMetric::new())  // Track validation accuracy
        .metric_train_numeric(LossMetric::new())      // Track training loss
        .metric_valid_numeric(LossMetric::new())      // Track validation loss
        .with_file_checkpointer(CompactRecorder::new()) // Save training checkpointers on disk
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs)
        .build(
            Model::<B>::new(&device), // Instantiate the MLP model
            config.optimizer.init(),  // Initialize the optimizer state
            1e-4,                    // Learning rate
        );

    // Start the training and validation loops
    let model_trained = learner.fit(dataloader_train, dataloader_valid);

    // Save the final trained model parameters (weights and biases) to disk
    model_trained
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("Model saving failed");
}

/// Implement the TrainStep trait for the Model to specify how it performs a single training iteration.
impl<B: AutodiffBackend> TrainStep<MnistBatch<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, batch: MnistBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward(batch.images);
        let loss = CrossEntropyLossConfig::new()
            .init(&item.device())
            .forward(item.clone(), batch.targets.clone());

        TrainOutput::new(
            self,
            loss.backward(),
            ClassificationOutput::new(loss, item, batch.targets),
        )
    }
}

/// Implement the ValidStep trait for the Model to specify how it evaluates validation data.
impl<B: Backend> ValidStep<MnistBatch<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, batch: MnistBatch<B>) -> ClassificationOutput<B> {
        let item = self.forward(batch.images);
        let loss = CrossEntropyLossConfig::new()
            .init(&item.device())
            .forward(item.clone(), batch.targets.clone());

        ClassificationOutput::new(loss, item, batch.targets)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::{Autodiff, NdArray};
    use burn::data::dataset::{vision::MnistItem, InMemDataset};

    #[test]
    fn test_train() {
        let artifact_dir = "./target/test-mnist-model";
        type TestBackend = Autodiff<NdArray>;
        let device = Default::default();

        // 1. Create a tiny mock dataset in memory (2 items)
        let item1 = MnistItem {
            image: [[0.0; 28]; 28],
            label: 3,
        };
        let item2 = MnistItem {
            image: [[1.0; 28]; 28],
            label: 7,
        };
        let train_dataset = InMemDataset::new(vec![item1.clone(), item2.clone()]);
        let valid_dataset = InMemDataset::new(vec![item1, item2]);

        // 2. Configure training for 1 epoch, batch size 2, 1 worker thread
        let mut config = TrainingConfig::new(AdamConfig::new());
        config.num_epochs = 1;
        config.batch_size = 2;
        config.num_workers = 1;

        // 3. Dry-run training on the mock dataset (will finish instantly)
        train::<TestBackend, _, _>(artifact_dir, config, device, train_dataset, valid_dataset);

        // 4. Verify that parameters were successfully optimized and saved
        assert!(std::path::Path::new(&format!("{artifact_dir}/model.mpk")).exists() 
            || std::path::Path::new(&format!("{artifact_dir}/model.bin")).exists()
            || std::path::Path::new(&format!("{artifact_dir}/model")).exists());

        // 5. Clean up test outputs
        std::fs::remove_dir_all(artifact_dir).ok();
    }
}
