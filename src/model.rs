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

const IMAGE_WIDTH: usize = 28;
const IMAGE_HEIGHT: usize = 28;

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    conv1: Conv2d<B>,
    ln1: LayerNorm<B>,
    conv2: Conv2d<B>,
    ln2: LayerNorm<B>,
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

        let pool = MaxPool2dConfig::new([2, 2])
            .with_strides([2, 2])
            .init();

        // After two 2×2 max-pools: 28 → 14 → 7
        // Flattened: 32 channels × 7 × 7 = 1568
        let fc1 = LinearConfig::new(32 * (IMAGE_WIDTH / 4) * (IMAGE_HEIGHT / 4), 256).init(device);
        let fc2 = LinearConfig::new(256, num_classes).init(device);

        // Lower dropout slightly since LayerNorm helps regularize
        let dropout = DropoutConfig::new(0.35).init();

        Self {
            conv1,
            ln1,
            conv2,
            ln2,
            pool,
            fc1,
            fc2,
            dropout,
        }
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 2> {
        // Block 1: Conv → LayerNorm (transposed) → ReLU → Pool
        let x = self.conv1.forward(input);
        // Transpose [B, C, H, W] -> [B, H, W, C] to normalize over channels (last dim)
        let x = x.swap_dims(1, 3);
        let x = self.ln1.forward(x);
        // Transpose back [B, H, W, C] -> [B, C, H, W]
        let x = x.swap_dims(1, 3);
        let x = relu(x);
        let x = self.pool.forward(x);

        // Block 2: Conv → LayerNorm (transposed) → ReLU → Pool
        let x = self.conv2.forward(x);
        // Transpose [B, C, H, W] -> [B, H, W, C]
        let x = x.swap_dims(1, 3);
        let x = self.ln2.forward(x);
        // Transpose back [B, H, W, C] -> [B, C, H, W]
        let x = x.swap_dims(1, 3);
        let x = relu(x);
        let x = self.pool.forward(x);

        // Flatten: [Batch, 32, 7, 7] → [Batch, 1568]
        let shape = x.shape();
        let batch_size = shape.dims[0];
        let x = x.reshape([batch_size, 32 * (IMAGE_WIDTH / 4) * (IMAGE_HEIGHT / 4)]);

        // FC layers
        let x = self.fc1.forward(x);
        let x = relu(x);
        let x = self.dropout.forward(x);
        self.fc2.forward(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    const NUM_CLASSES: usize = 10;

    #[test]
    fn test_model_forward() {
        type TestBackend = NdArray;
        let device = Default::default();

        let model = Model::<TestBackend>::new(&device, NUM_CLASSES);
        let input = Tensor::<TestBackend, 4>::zeros([4, 1, IMAGE_WIDTH, IMAGE_HEIGHT], &device);
        let output = model.forward(input);

        let shape = output.shape();
        assert_eq!(shape.dims, [4, NUM_CLASSES]);
    }

    #[test]
    fn test_model_forward_dynamic_classes() {
        type TestBackend = NdArray;
        let device = Default::default();
        let num_classes = 25;

        let model = Model::<TestBackend>::new(&device, num_classes);
        let input = Tensor::<TestBackend, 4>::zeros([4, 1, IMAGE_WIDTH, IMAGE_HEIGHT], &device);
        let output = model.forward(input);

        let shape = output.shape();
        assert_eq!(shape.dims, [4, num_classes]);
    }
}
