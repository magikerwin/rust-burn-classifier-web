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
pub struct SeparableConv2d<B: Backend> {
    depthwise: Conv2d<B>,
    dw_bn: BatchNorm<B, 2>,
    pointwise: Conv2d<B>,
    pw_bn: BatchNorm<B, 2>,
}

impl<B: Backend> SeparableConv2d<B> {
    pub fn new(device: &B::Device, in_channels: usize, out_channels: usize, stride: [usize; 2]) -> Self {
        // 1. Depthwise: 3x3 spatial filter per channel
        let depthwise = Conv2dConfig::new([in_channels, in_channels], [3, 3])
            .with_stride(stride) // Stride is applied here
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_groups(in_channels) // Depthwise grouping
            .init(device);
        let dw_bn = BatchNormConfig::new(in_channels).init(device);

        // 2. Pointwise: 1x1 channel mixer
        let pointwise = Conv2dConfig::new([in_channels, out_channels], [1, 1])
            .with_stride([1, 1])
            .init(device);
        let pw_bn = BatchNormConfig::new(out_channels).init(device);

        Self {
            depthwise,
            dw_bn,
            pointwise,
            pw_bn,
        }
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.depthwise.forward(input);
        let x = self.dw_bn.forward(x);
        let x = relu(x);

        let x = self.pointwise.forward(x);
        let x = self.pw_bn.forward(x);
        // Notice we don't apply ReLU after the pointwise here,
        // because we will apply it after adding the residual shortcut in the main model.
        x
    }
}

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    // Stem Layer (Standard Conv)
    stem_conv: Conv2d<B>,
    stem_bn: BatchNorm<B, 2>,

    // Block 1: Separable + Projection Shortcut
    conv1: SeparableConv2d<B>,
    proj1: Conv2d<B>,
    proj1_bn: BatchNorm<B, 2>,

    // Block 2: Separable + Projection Shortcut
    conv2: SeparableConv2d<B>,
    proj2: Conv2d<B>,
    proj2_bn: BatchNorm<B, 2>,

    // Block 3: Separable + Identity Shortcut
    conv3: SeparableConv2d<B>,

    dropout: Dropout,
    fc: Linear<B>,
}

impl<B: Backend> Model<B> {
    pub fn new(device: &B::Device, num_classes: usize) -> Self {
        // --- Stem (Standard 3x3) ---
        let stem_conv = Conv2dConfig::new([1, 24], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);
        let stem_bn = BatchNormConfig::new(24).init(device);

        // --- Block 1 (Separable) ---
        let conv1 = SeparableConv2d::new(device, 24, 48, [2, 2]);
        let proj1 = Conv2dConfig::new([24, 48], [1, 1])
            .with_stride([2, 2])
            .init(device);
        let proj1_bn = BatchNormConfig::new(48).init(device);

        // --- Block 2 (Separable) ---
        let conv2 = SeparableConv2d::new(device, 48, 96, [2, 2]);
        let proj2 = Conv2dConfig::new([48, 96], [1, 1])
            .with_stride([2, 2])
            .init(device);
        let proj2_bn = BatchNormConfig::new(96).init(device);

        // --- Block 3 (Separable, Identity mapping) ---
        let conv3 = SeparableConv2d::new(device, 96, 96, [1, 1]);

        let dropout = DropoutConfig::new(0.5).init();
        let fc = LinearConfig::new(96, num_classes).init(device);

        Self {
            stem_conv, stem_bn,
            conv1, proj1, proj1_bn,
            conv2, proj2, proj2_bn,
            conv3,
            dropout, fc,
        }
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 2> {
        // Stem
        let x = self.stem_conv.forward(input);
        let x = self.stem_bn.forward(x);
        let x = relu(x);

        // Block 1
        let y = self.conv1.forward(x.clone());
        let shortcut = self.proj1.forward(x);
        let shortcut = self.proj1_bn.forward(shortcut);
        let x = relu(y + shortcut);

        // Block 2
        let y = self.conv2.forward(x.clone());
        let shortcut = self.proj2.forward(x);
        let shortcut = self.proj2_bn.forward(shortcut);
        let x = relu(y + shortcut);

        // Block 3 (Identity)
        let y = self.conv3.forward(x.clone());
        let x = relu(y + x);

        // Classifier
        let x = x.mean_dim(2).mean_dim(3);
        let shape = x.shape();
        let x = x.reshape([shape.dims[0], 96]);
        let x = self.dropout.forward(x);
        
        self.fc.forward(x)
    }
}
