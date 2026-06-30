use burn::{
    nn::{
        conv::{Conv2d, Conv2dConfig},
        pool::{MaxPool2d, MaxPool2dConfig},
        PaddingConfig2d,
        Dropout, DropoutConfig,
        Linear, LinearConfig,
    },
    prelude::*,
    tensor::activation::relu,
};

const IMAGE_WIDTH: usize = 28;
const IMAGE_HEIGHT: usize = 28;
const NUM_CLASSES: usize = 10;

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    pool: MaxPool2d,
    fc1: Linear<B>,
    fc2: Linear<B>,
    dropout: Dropout,
}

impl<B: Backend> Model<B> {
    pub fn new(device: &B::Device) -> Self {
        let conv1 = Conv2dConfig::new([1, 8], [3, 3])
            .with_padding(PaddingConfig2d::Same)
            .init(device);
        let conv2 = Conv2dConfig::new([8, 16], [3, 3])
            .with_padding(PaddingConfig2d::Same)
            .init(device);
        let pool = MaxPool2dConfig::new([2, 2])
            .with_strides([2, 2])
            .init();
        // After two 2x2 max-pools: 28 -> 14 -> 7
        let fc1 = LinearConfig::new(16 * (IMAGE_WIDTH / 4) * (IMAGE_HEIGHT / 4), 128).init(device);
        let fc2 = LinearConfig::new(128, NUM_CLASSES).init(device);
        let dropout = DropoutConfig::new(0.5).init();

        Self {
            conv1,
            conv2,
            pool,
            fc1,
            fc2,
            dropout,
        }
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 2> {
        let x = self.conv1.forward(input);
        let x = relu(x);
        let x = self.pool.forward(x);

        let x = self.conv2.forward(x);
        let x = relu(x);
        let x = self.pool.forward(x);

        // Reshape/flatten for FC layer: [Batch, Channels, Height, Width] -> [Batch, Channels * Height * Width]
        let shape = x.shape();
        let batch_size = shape.dims[0];
        let x = x.reshape([batch_size, 16 * (IMAGE_WIDTH / 4) * (IMAGE_HEIGHT / 4)]);

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

    #[test]
    fn test_model_forward() {
        type TestBackend = NdArray;
        let device = Default::default();

        let model = Model::<TestBackend>::new(&device);
        let input = Tensor::<TestBackend, 4>::zeros([4, 1, IMAGE_WIDTH, IMAGE_HEIGHT], &device);
        let output = model.forward(input);

        let shape = output.shape();
        assert_eq!(shape.dims, [4, NUM_CLASSES]);
    }
}
