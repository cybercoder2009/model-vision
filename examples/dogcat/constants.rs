#![allow(dead_code)]

// env
pub const GPU_ID: usize = 0;
pub const ARTIFACT_DIR: &str = "artifacts";

// data
pub const NUM_WORKERS: usize = 2;
pub const NUM_EPOCHS: usize = 1;
pub const NUM_SHUFFLE: u64 = 42;
pub const NUM_CLASSES: usize = 2;
pub const TRAIN_RATIO: f32 = 0.8;
pub const BATCH_SIZE: usize = 16;

// backbone
pub const NUM_LAYERS: usize = 12;
pub const HIDDEN_DIM: usize = 192;
pub const IMAGE_SIZE: usize = 224; // pixels
pub const NUM_HEADS: usize = 3;
pub const MLP_DIM: usize = 768;
pub const PATCH_SIZE: usize = 14; // pixels
pub const LEARNING_RATE: f64 = 1e-3;
