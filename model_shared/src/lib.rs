use burn::{
    module::Module,
    nn::{
        conv::{Conv2d, Conv2dConfig},
        BatchNorm, BatchNormConfig,
        Dropout, DropoutConfig,
        PaddingConfig2d,
        Linear, LinearConfig,
    },
    prelude::*,
    tensor::activation::relu,
};

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    // Stem Layer: Basic feature extraction (28x28, 1 -> 32 channels)
    stem_conv: Conv2d<B>,
    stem_bn: BatchNorm<B, 2>, // Note: 2 denotes 2D spatial dimensions (H, W)

    // Block 1: 28x28 -> 14x14, 32 -> 64 channels
    conv1: Conv2d<B>,
    bn1: BatchNorm<B, 2>,
    proj1: Conv2d<B>,
    proj1_bn: BatchNorm<B, 2>, // Projection also requires BatchNorm

    // Block 2: 14x14 -> 7x7, 64 -> 128 channels
    conv2: Conv2d<B>,
    bn2: BatchNorm<B, 2>,
    proj2: Conv2d<B>,
    proj2_bn: BatchNorm<B, 2>,

    // Block 3: Stays at 7x7, 128 -> 128 channels
    conv3: Conv2d<B>,
    bn3: BatchNorm<B, 2>,
    proj3: Conv2d<B>,
    proj3_bn: BatchNorm<B, 2>,

    // Classifier (Includes Dropout to prevent overfitting)
    dropout: Dropout,
    fc: Linear<B>,
}

impl<B: Backend> Model<B> {
    pub fn new(device: &B::Device, num_classes: usize) -> Self {
        // --- Stem Layer ---
        let stem_conv = Conv2dConfig::new([1, 32], [3, 3])
            .with_stride([1, 1]) // No downsampling, preserve details
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);
        let stem_bn = BatchNormConfig::new(32).init(device);

        // --- Block 1 ---
        let conv1 = Conv2dConfig::new([32, 64], [3, 3])
            .with_stride([2, 2]) // Downsampling
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);
        let bn1 = BatchNormConfig::new(64).init(device);
        let proj1 = Conv2dConfig::new([32, 64], [1, 1])
            .with_stride([2, 2])
            .init(device);
        let proj1_bn = BatchNormConfig::new(64).init(device);

        // --- Block 2 ---
        let conv2 = Conv2dConfig::new([64, 128], [3, 3])
            .with_stride([2, 2]) // Downsampling
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);
        let bn2 = BatchNormConfig::new(128).init(device);
        let proj2 = Conv2dConfig::new([64, 128], [1, 1])
            .with_stride([2, 2])
            .init(device);
        let proj2_bn = BatchNormConfig::new(128).init(device);

        // --- Block 3 ---
        let conv3 = Conv2dConfig::new([128, 128], [3, 3])
            .with_stride([1, 1]) // No downsampling
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);
        let bn3 = BatchNormConfig::new(128).init(device);
        let proj3 = Conv2dConfig::new([128, 128], [1, 1])
            .with_stride([1, 1])
            .init(device);
        let proj3_bn = BatchNormConfig::new(128).init(device);

        // --- Classifier ---
        let dropout = DropoutConfig::new(0.5).init(); // 50% random dropout
        let fc = LinearConfig::new(128, num_classes).init(device);

        Self {
            stem_conv,
            stem_bn,
            conv1,
            bn1,
            proj1,
            proj1_bn,
            conv2,
            bn2,
            proj2,
            proj2_bn,
            conv3,
            bn3,
            proj3,
            proj3_bn,
            dropout,
            fc,
        }
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 2> {
        // --- Stem ---
        let x = self.stem_conv.forward(input);
        let x = self.stem_bn.forward(x);
        let x = relu(x);

        // --- Block 1 ---
        let y = self.conv1.forward(x.clone());
        let y = self.bn1.forward(y);
        let y = relu(y); // Step 1: non-linearity in main branch
        
        let shortcut = self.proj1.forward(x);
        let shortcut = self.proj1_bn.forward(shortcut); // Step 2: Shortcut also uses BatchNorm
        let x = relu(y + shortcut);

        // --- Block 2 ---
        let y = self.conv2.forward(x.clone());
        let y = self.bn2.forward(y);
        let y = relu(y);
        
        let shortcut = self.proj2.forward(x);
        let shortcut = self.proj2_bn.forward(shortcut);
        let x = relu(y + shortcut);

        // --- Block 3 ---
        let y = self.conv3.forward(x.clone());
        let y = self.bn3.forward(y);
        let y = relu(y);
        
        let shortcut = self.proj3.forward(x);
        let shortcut = self.proj3_bn.forward(shortcut);
        let x = relu(y + shortcut);

        // --- Global Average Pooling ---
        // [Batch, 128, 7, 7] -> [Batch, 128, 1, 1]
        let x = x.mean_dim(2).mean_dim(3);

        // Flatten (Reshape)
        // [Batch, 128, 1, 1] -> [Batch, 128]
        let shape = x.shape();
        let batch_size = shape.dims[0];
        let x = x.reshape([batch_size, 128]);

        // --- Dropout & Classifier ---
        let x = self.dropout.forward(x); // Apply dropout for regularization
        self.fc.forward(x)
    }
}
