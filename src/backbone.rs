use burn::{
    config::Config,
    module::Module,
    nn::{
        conv::{Conv2d, Conv2dConfig},
        transformer::{TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput},
    },
    tensor::{Distribution, Tensor, backend::Backend},
};

#[derive(Config, Debug)]
pub struct BackboneConfig {
    /// Input image size (height and width, e.g., 224)
    pub image_size: usize,
    /// The dimension of the hidden layers.
    pub hidden_dim: usize,
    /// The number of transformer layers.
    pub num_layers: usize,
    /// The number of attention heads.
    pub num_heads: usize,
    /// The dimension of the MLP layers.
    pub mlp_dim: usize,
    /// The size of each patch.
    pub patch_size: usize,
    /// The number of channels in the input images.
    #[config(default = 3)]
    pub num_channels: usize,
    /// Dropout rate
    #[config(default = 0.1)]
    pub dropout: f64,
}

/// Vision Transformer — image [B, C, H, W] → CLS embedding [B, D]
#[derive(Module, Debug)]
pub struct Backbone<B: Backend> {
    pub(crate) patch_embed: Conv2d<B>,
    pub(crate) cls_token: Tensor<B, 2>,
    pub(crate) pos_embed: Tensor<B, 3>,
    pub(crate) transformer: TransformerEncoder<B>,
    pub(crate) hidden_dim: usize,
    pub(crate) num_patches: usize,
}

impl<B: Backend> Backbone<B> {
    pub fn new(device: &B::Device, config: &BackboneConfig) -> Self {
        let n_per_side = config.image_size / config.patch_size;
        let num_patches = n_per_side * n_per_side;
        let seq_len = num_patches + 1;

        let patch_embed = Conv2dConfig::new(
            [config.num_channels, config.hidden_dim],
            [config.patch_size, config.patch_size],
        )
        .with_stride([config.patch_size, config.patch_size])
        .init(device);

        let cls_token = Tensor::random(
            [1, config.hidden_dim],
            Distribution::Normal(0.0, 0.02),
            device,
        );

        let pos_embed = Tensor::random(
            [1, seq_len, config.hidden_dim],
            Distribution::Normal(0.0, 0.02),
            device,
        );

        let transformer = TransformerEncoderConfig::new(
            config.hidden_dim,
            config.mlp_dim,
            config.num_heads,
            config.num_layers,
        )
        .with_norm_first(true)
        .with_dropout(config.dropout)
        .init(device);

        Self {
            patch_embed,
            cls_token,
            pos_embed,
            transformer,
            hidden_dim: config.hidden_dim,
            num_patches,
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 2> {
        let [batch_size, _c, _h, _w] = x.dims();

        let x = self.patch_embed.forward(x);
        let [b, d, hp, wp] = x.dims();
        let x = x.reshape([b, d, hp * wp]).swap_dims(1, 2);

        let cls = self
            .cls_token
            .clone()
            .unsqueeze_dim(0)
            .repeat_dim(0, batch_size);
        let x = Tensor::cat(vec![cls, x], 1) + self.pos_embed.clone();

        let out = self.transformer.forward(TransformerEncoderInput::new(x));
        out.slice([0..batch_size, 0..1, 0..self.hidden_dim])
            .squeeze_dim(1)
    }
}
