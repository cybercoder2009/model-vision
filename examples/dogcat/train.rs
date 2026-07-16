mod constants;
mod data;

use crate::constants::*;
use burn::{
    backend::{
        Autodiff,
        wgpu::{Wgpu, WgpuDevice},
    },
    config::Config,
    data::{dataloader::DataLoaderBuilder, dataset::Dataset},
    module::Module,
    optim::AdamWConfig,
    record::DefaultRecorder,
    tensor::backend::Backend,
    train::{
        ClassificationOutput, InferenceStep, Learner, SupervisedTraining, TrainOutput, TrainStep,
        checkpoint::MetricCheckpointingStrategy,
        metric::{
            LossMetric,
            store::{Aggregate, Direction, Split},
        },
    },
};
use data::{CatDogBatch, CatDogBatcher, CatDogDataset};
use model_vit::{BackboneConfig, Classification, ClassificationConfig};
use std::path::PathBuf;

#[derive(Module, Debug)]
struct Model<B: Backend> {
    classification: Classification<B>,
}

impl<B: burn::tensor::backend::AutodiffBackend> TrainStep for Model<B> {
    type Input = CatDogBatch<B>;
    type Output = ClassificationOutput<B>;
    fn step(&self, batch: CatDogBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let out = self
            .classification
            .forward_classification(batch.images, batch.labels);
        TrainOutput::new(self, out.loss.backward(), out)
    }
}

impl<B: Backend> InferenceStep for Model<B> {
    type Input = CatDogBatch<B>;
    type Output = ClassificationOutput<B>;
    fn step(&self, batch: CatDogBatch<B>) -> ClassificationOutput<B> {
        self.classification
            .forward_classification(batch.images, batch.labels)
    }
}

fn main() {
    // prepare
    type B = Autodiff<Wgpu<f32, i32>>;
    let device = WgpuDevice::DiscreteGpu(GPU_ID);
    let config = ClassificationConfig::new(
        BackboneConfig::new(
            IMAGE_SIZE, HIDDEN_DIM, NUM_LAYERS, NUM_HEADS, MLP_DIM, PATCH_SIZE,
        ),
        NUM_CLASSES,
    );
    let model = Model {
        classification: config.init(&device),
    };

    // dataset
    let dataset = CatDogDataset::new("archive");
    let n = dataset.len();
    let n_train = (n as f32 * TRAIN_RATIO) as usize;
    let batcher_train = CatDogBatcher::<B> {
        device: device.clone(),
        image_size: IMAGE_SIZE,
    };
    let batcher_valid = CatDogBatcher::<Wgpu<f32, i32>> {
        device: device.clone(),
        image_size: IMAGE_SIZE,
    };
    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(BATCH_SIZE)
        .shuffle(NUM_SHUFFLE)
        .num_workers(NUM_WORKERS)
        .build(burn::data::dataset::InMemDataset::new(
            dataset.items[..n_train].to_vec(),
        ));
    let dataloader_valid = DataLoaderBuilder::new(batcher_valid)
        .batch_size(BATCH_SIZE)
        .num_workers(NUM_WORKERS)
        .build(burn::data::dataset::InMemDataset::new(
            dataset.items[n_train..].to_vec(),
        ));

    // training
    let loss_metric = LossMetric::<Wgpu<f32, i32>>::new();
    let checkpoint_strategy = MetricCheckpointingStrategy::new(
        &loss_metric,
        Aggregate::Mean,
        Direction::Lowest,
        Split::Valid,
    );
    let trained_model = SupervisedTraining::new(ARTIFACT_DIR, dataloader_train, dataloader_valid)
        .with_file_checkpointer(DefaultRecorder::new())
        .with_checkpointing_strategy(checkpoint_strategy)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .num_epochs(NUM_EPOCHS)
        .launch(Learner::new(
            model,
            AdamWConfig::new().init(),
            LEARNING_RATE,
        ));

    // output
    let dst = PathBuf::from(ARTIFACT_DIR);
    config
        .save(dst.join("config.json"))
        .expect("Failed to save config");
    trained_model.model.classification
        .save_file(dst.join("model"), &DefaultRecorder::new())
        .expect("Failed to save trained model");
    println!(
        "Done → Model saved to {}/model.mpk and config saved to {}/config.json",
        ARTIFACT_DIR, ARTIFACT_DIR
    );
}
