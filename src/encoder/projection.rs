use crate::backbone::{Backbone, BackboneConfig};
use burn::{
    config::Config,
    module::Module,
    nn::{BatchNorm, BatchNormConfig, Linear, LinearConfig},
    tensor::{Tensor, backend::Backend},
};

#[derive(Config, Debug)]
pub struct ProjectionConfig {
    pub backbone: BackboneConfig,
}

impl ProjectionConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Projection<B> {
        Projection::new(device, self)
    }
}

#[derive(Module, Debug)]
struct Head<B: Backend> {
    linear: Linear<B>,
    bn: BatchNorm<B>, // batch norm
}

impl<B: Backend> Head<B> {
    pub fn new(hidden_dim: usize, device: &B::Device) -> Self {
        Self {
            linear: LinearConfig::new(hidden_dim, hidden_dim).init(device),
            bn: BatchNormConfig::new(hidden_dim).init(device),
        }
    }
    pub fn forward(&self, cls: Tensor<B, 2>) -> Tensor<B, 2> {
        self.bn.forward(self.linear.forward(cls))
    }
}

#[derive(Module, Debug)]
pub struct Projection<B: Backend> {
    backbone: Backbone<B>,
    head: Head<B>,
}

impl<B: Backend> Projection<B> {
    pub fn new(device: &B::Device, config: &ProjectionConfig) -> Self {
        Self {
            backbone: Backbone::new(device, &config.backbone),
            head: Head::new(config.backbone.hidden_dim, device),
        }
    }

    pub fn forward(&self, images: Tensor<B, 4>) -> Tensor<B, 2> {
        self.head.forward(self.backbone.forward(images))
    }
}
