use crate::backbone::{Backbone, BackboneConfig};
use burn::{
    config::Config,
    module::Module,
    nn::loss::CrossEntropyLossConfig,
    nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig},
    tensor::{Int, Tensor, backend::Backend},
    train::ClassificationOutput,
};

#[derive(Config, Debug)]
pub struct ClassificationConfig {
    pub backbone: BackboneConfig,
    pub num_classes: usize,
}

impl ClassificationConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Classification<B> {
        Classification::new(device, self)
    }
}

#[derive(Module, Debug)]
struct Head<B: Backend> {
    linear: Linear<B>,
    ln: LayerNorm<B>, // layer norm
}

impl<B: Backend> Head<B> {
    pub fn new(hidden_dim: usize, num_classes: usize, device: &B::Device) -> Self {
        Self {
            linear: LinearConfig::new(hidden_dim, num_classes).init(device),
            ln: LayerNormConfig::new(hidden_dim).init(device),
        }
    }
    pub fn forward(&self, cls: Tensor<B, 2>) -> Tensor<B, 2> {
        self.linear.forward(self.ln.forward(cls))
    }
}

#[derive(Module, Debug)]
pub struct Classification<B: Backend> {
    backbone: Backbone<B>,
    head: Head<B>,
}

impl<B: Backend> Classification<B> {
    pub fn new(device: &B::Device, config: &ClassificationConfig) -> Self {
        Self {
            backbone: Backbone::new(device, &config.backbone),
            head: Head::new(config.backbone.hidden_dim, config.num_classes, device),
        }
    }

    pub fn forward(&self, images: Tensor<B, 4>) -> Tensor<B, 2> {
        self.head.forward(self.backbone.forward(images))
    }
    pub fn forward_classification(
        &self,
        images: Tensor<B, 4>,
        labels: Tensor<B, 1, Int>,
    ) -> ClassificationOutput<B> {
        let logits = self.forward(images);
        let loss = CrossEntropyLossConfig::new()
            .init(&logits.device())
            .forward(logits.clone(), labels.clone());
        ClassificationOutput {
            loss,
            output: logits,
            targets: labels,
        }
    }
}
