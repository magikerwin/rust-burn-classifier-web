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
    // Stem Layer (1 -> 24 channels)
    stem_conv: Conv2d<B>,
    stem_bn: BatchNorm<B, 2>,

    // Block 1: 28x28 -> 14x14 (24 -> 48 channels)
    conv1: Conv2d<B>,
    bn1: BatchNorm<B, 2>,
    proj1: Conv2d<B>,
    proj1_bn: BatchNorm<B, 2>,

    // Block 2: 14x14 -> 7x7 (48 -> 96 channels)
    conv2: Conv2d<B>,
    bn2: BatchNorm<B, 2>,
    proj2: Conv2d<B>,
    proj2_bn: BatchNorm<B, 2>,

    // Block 3: 7x7 -> 7x7 (96 -> 96 channels)
    // NOTE: proj3 is completely removed for latency optimization!
    conv3: Conv2d<B>,
    bn3: BatchNorm<B, 2>,

    dropout: Dropout,
    fc: Linear<B>,
}

impl<B: Backend> Model<B> {
    pub fn new(device: &B::Device, num_classes: usize) -> Self {
        // --- Stem Layer ---
        let stem_conv = Conv2dConfig::new([1, 24], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);
        let stem_bn = BatchNormConfig::new(24).init(device);

        // --- Block 1 ---
        let conv1 = Conv2dConfig::new([24, 48], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);
        let bn1 = BatchNormConfig::new(48).init(device);
        let proj1 = Conv2dConfig::new([24, 48], [1, 1])
            .with_stride([2, 2])
            .init(device);
        let proj1_bn = BatchNormConfig::new(48).init(device);

        // --- Block 2 ---
        let conv2 = Conv2dConfig::new([48, 96], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);
        let bn2 = BatchNormConfig::new(96).init(device);
        let proj2 = Conv2dConfig::new([48, 96], [1, 1])
            .with_stride([2, 2])
            .init(device);
        let proj2_bn = BatchNormConfig::new(96).init(device);

        // --- Block 3 (Identity Block) ---
        let conv3 = Conv2dConfig::new([96, 96], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);
        let bn3 = BatchNormConfig::new(96).init(device);

        // --- Classifier ---
        let dropout = DropoutConfig::new(0.5).init();
        let fc = LinearConfig::new(96, num_classes).init(device);

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
        let y = relu(y);
        
        let shortcut = self.proj1.forward(x);
        let shortcut = self.proj1_bn.forward(shortcut);
        let x = relu(y + shortcut);

        // --- Block 2 ---
        let y = self.conv2.forward(x.clone());
        let y = self.bn2.forward(y);
        let y = relu(y);
        
        let shortcut = self.proj2.forward(x);
        let shortcut = self.proj2_bn.forward(shortcut);
        let x = relu(y + shortcut);

        // --- Block 3 (Identity Shortcut) ---
        let y = self.conv3.forward(x.clone());
        let y = self.bn3.forward(y);
        let y = relu(y);
        
        // NO projection convolution here. Pure identity mapping.
        // This saves a 96x96 channel multiplication over the 7x7 spatial grid!
        let x = relu(y + x); 

        // --- GAP & Classifier ---
        let x = x.mean_dim(2).mean_dim(3);
        
        let shape = x.shape();
        let batch_size = shape.dims[0];
        let x = x.reshape([batch_size, 96]);

        let x = self.dropout.forward(x);
        self.fc.forward(x)
    }
}
