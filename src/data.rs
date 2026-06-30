// TODO: Remove this warning suppression once the batcher is integrated into the training loop.
#![allow(dead_code)]

use burn::{
    data::{dataloader::batcher::Batcher, dataset::vision::MnistItem},
    prelude::*,
};

// Clone is required by the DataLoader to send copies of the batcher to multiple worker threads.
#[derive(Clone, Debug)]
pub struct MnistBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> MnistBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

#[derive(Clone, Debug)]
pub struct MnistBatch<B: Backend> {
    pub images: Tensor<B, 2>,
    pub targets: Tensor<B, 1, Int>,
}

impl<B: Backend> Batcher<MnistItem, MnistBatch<B>> for MnistBatcher<B> {
    fn batch(&self, items: Vec<MnistItem>) -> MnistBatch<B> {
        let images = items
            .iter()
            .map(|item| TensorData::from(item.image))
            .map(|data| Tensor::<B, 2>::from_data(data, &self.device))
            // Reshape from [28, 28] to [1, 28 * 28] to flatten the image
            .map(|tensor| tensor.reshape([1, 28 * 28]))
            .collect::<Vec<_>>();

        let targets = items
            .iter()
            .map(|item| {
                Tensor::<B, 1, Int>::from_data(
                    TensorData::from([item.label as i32]),
                    &self.device,
                )
            })
            .collect::<Vec<_>>();

        // Concatenate all tensors along the batch dimension (dim 0)
        let images = Tensor::cat(images, 0);
        let targets = Tensor::cat(targets, 0);

        MnistBatch { images, targets }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    #[test]
    fn test_batcher() {
        type TestBackend = NdArray;
        let device = Default::default();
        let batcher = MnistBatcher::<TestBackend>::new(device);

        let item1 = MnistItem {
            image: [[0.0; 28]; 28],
            label: 3,
        };
        let item2 = MnistItem {
            image: [[1.0; 28]; 28],
            label: 7,
        };

        let batch = batcher.batch(vec![item1, item2]);

        let image_shape = batch.images.shape();
        assert_eq!(image_shape.dims, [2, 28 * 28]);

        let target_shape = batch.targets.shape();
        assert_eq!(target_shape.dims, [2]);
    }
}
