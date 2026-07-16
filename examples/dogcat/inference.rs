mod constants;

use crate::constants::{ARTIFACT_DIR, GPU_ID};
use burn::{
    backend::wgpu::{Wgpu, WgpuDevice},
    config::Config,
    module::Module,
    record::{DefaultRecorder, Recorder},
    tensor::Tensor,
};
use image::imageops::FilterType;
use model_vit::ClassificationConfig;
use std::path::PathBuf;

fn main() {
    type B = Wgpu<f32, i32>;
    let device = WgpuDevice::DiscreteGpu(GPU_ID);

    // Load config from JSON
    let config_path = PathBuf::from(ARTIFACT_DIR).join("config.json");
    if !config_path.exists() {
        eprintln!("No artifacts/config.json — run `cargo run --example dogcat_train` first");
        return;
    }
    let config = ClassificationConfig::load(&config_path).expect("Failed to load config");
    let image_size = config.backbone.image_size;
    let model = config.init(&device);

    // Load trained weights
    let model_path = PathBuf::from(ARTIFACT_DIR).join("model");
    let model_file_path = PathBuf::from(ARTIFACT_DIR).join("model.mpk");
    if !model_file_path.exists() {
        eprintln!("No artifacts/model.mpk — run `cargo run --example dogcat_train` first");
        return;
    }
    let record = DefaultRecorder::new()
        .load(model_path, &device)
        .unwrap();
    let model = model.load_record(record);

    // Load and preprocess image
    let img = image::open("archive/cats_set/cat.4001.jpg").expect("Failed to open image");
    let img = img
        .resize_exact(image_size as u32, image_size as u32, FilterType::Triangle)
        .to_rgb8();

    let mut data = Vec::with_capacity(3 * image_size * image_size);
    for c in 0..3usize {
        for y in 0..image_size {
            for x in 0..image_size {
                let p = img.get_pixel(x as u32, y as u32);
                data.push((p[c] as f32 / 127.5) - 1.0);
            }
        }
    }

    let image: Tensor<B, 4> = Tensor::from_data(
        burn::tensor::TensorData::new(data, [1, 3, image_size, image_size]),
        &device,
    );

    let logits = model.forward(image);
    let values: Vec<f32> = logits.to_data().to_vec().unwrap();
    let max = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = values.iter().map(|x| (x - max).exp()).collect();
    let sum: f32 = exp.iter().sum();
    let prob: Vec<f32> = exp.iter().map(|x| x / sum).collect();
    println!("Cat: {:.4}  Dog: {:.4}", prob[0], prob[1]);
}
