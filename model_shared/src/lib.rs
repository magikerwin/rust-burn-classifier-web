use burn::{
    nn::{
        conv::{Conv2d, Conv2dConfig},
        pool::{MaxPool2d, MaxPool2dConfig},
        LayerNorm, LayerNormConfig,
        PaddingConfig2d,
        Dropout, DropoutConfig,
        Linear, LinearConfig,
    },
    prelude::*,
    tensor::activation::relu,
};


#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    conv1: Conv2d<B>,
    ln1: LayerNorm<B>,
    conv2: Conv2d<B>,
    ln2: LayerNorm<B>,
    conv3: Conv2d<B>,
    ln3: LayerNorm<B>,
    pool: MaxPool2d,
    fc1: Linear<B>,
    fc2: Linear<B>,
    dropout: Dropout,
}

impl<B: Backend> Model<B> {
    pub fn new(device: &B::Device, num_classes: usize) -> Self {
        // conv1: 1 → 16 channels
        let conv1 = Conv2dConfig::new([1, 16], [3, 3])
            .with_padding(PaddingConfig2d::Same)
            .init(device);
        let ln1 = LayerNormConfig::new(16).init(device);

        // conv2: 16 → 32 channels
        let conv2 = Conv2dConfig::new([16, 32], [3, 3])
            .with_padding(PaddingConfig2d::Same)
            .init(device);
        let ln2 = LayerNormConfig::new(32).init(device);

        // conv3: 32 → 64 channels (no pooling follows this conv)
        let conv3 = Conv2dConfig::new([32, 64], [3, 3])
            .with_padding(PaddingConfig2d::Same)
            .init(device);
        let ln3 = LayerNormConfig::new(64).init(device);

        let pool = MaxPool2dConfig::new([2, 2])
            .with_strides([2, 2])
            .init();

        // After two 2×2 pools: 28 → 14 → 7.
        // conv3 keeps shape at 7×7.
        // GAP (Global Average Pooling) averages the 7×7 map to a single channel scalar.
        // So fc1 receives exactly 64 channels.
        let fc1 = LinearConfig::new(64, 128).init(device);
        let fc2 = LinearConfig::new(128, num_classes).init(device);

        let dropout = DropoutConfig::new(0.35).init();

        Self {
            conv1,
            ln1,
            conv2,
            ln2,
            conv3,
            ln3,
            pool,
            fc1,
            fc2,
            dropout,
        }
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 2> {
        // Block 1: Conv → LayerNorm (transposed) → ReLU → Pool (28 -> 14)
        let x = self.conv1.forward(input);
        let x = x.swap_dims(1, 3);
        let x = self.ln1.forward(x);
        let x = x.swap_dims(1, 3);
        let x = relu(x);
        let x = self.pool.forward(x);

        // Block 2: Conv → LayerNorm (transposed) → ReLU → Pool (14 -> 7)
        let x = self.conv2.forward(x);
        let x = x.swap_dims(1, 3);
        let x = self.ln2.forward(x);
        let x = x.swap_dims(1, 3);
        let x = relu(x);
        let x = self.pool.forward(x);

        // Block 3: Conv → LayerNorm (transposed) → ReLU (No pool, stays at 7x7)
        let x = self.conv3.forward(x);
        let x = x.swap_dims(1, 3);
        let x = self.ln3.forward(x);
        let x = x.swap_dims(1, 3);
        let x = relu(x);

        // Global Average Pooling: [Batch, 64, 7, 7] → [Batch, 64, 1, 1]
        let x = x.mean_dim(2).mean_dim(3);

        // Flatten: [Batch, 64, 1, 1] → [Batch, 64]
        let shape = x.shape();
        let batch_size = shape.dims[0];
        let x = x.reshape([batch_size, 64]);

        // Classifier
        let x = self.fc1.forward(x);
        let x = relu(x);
        let x = self.dropout.forward(x);
        self.fc2.forward(x)
    }
}
