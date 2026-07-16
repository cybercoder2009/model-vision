use burn::{
    data::dataloader::batcher::Batcher,
    tensor::{backend::Backend, Int, Tensor, TensorData},
    data::dataset::Dataset,
};
use image::imageops::FilterType;
use std::path::PathBuf;
use walkdir::WalkDir;

#[derive(Clone, Debug)]
pub struct CatDogItem {
    pub path: PathBuf,
    pub label: usize,
}

pub struct CatDogDataset {
    pub items: Vec<CatDogItem>,
}

impl CatDogDataset {
    pub fn new(root: &str) -> Self {
        let mut items = Vec::new();
        for entry in WalkDir::new(PathBuf::from(root).join("cats_set"))
            .into_iter().filter_map(|e| e.ok())
        {
            if entry.path().is_file() {
                items.push(CatDogItem { path: entry.path().to_owned(), label: 0 });
            }
        }
        for entry in WalkDir::new(PathBuf::from(root).join("dogs_set"))
            .into_iter().filter_map(|e| e.ok())
        {
            if entry.path().is_file() {
                items.push(CatDogItem { path: entry.path().to_owned(), label: 1 });
            }
        }
        Self { items }
    }
}

impl Dataset<CatDogItem> for CatDogDataset {
    fn get(&self, index: usize) -> Option<CatDogItem> { self.items.get(index).cloned() }
    fn len(&self) -> usize { self.items.len() }
}

#[derive(Clone)]
pub struct CatDogBatcher<B: Backend> {
    pub device: B::Device,
    pub image_size: usize,
}

#[derive(Clone, Debug)]
pub struct CatDogBatch<B: Backend> {
    pub images: Tensor<B, 4>,
    pub labels: Tensor<B, 1, Int>,
}

impl<B: Backend> Batcher<B, CatDogItem, CatDogBatch<B>> for CatDogBatcher<B> {
    fn batch(&self, items: Vec<CatDogItem>, _device: &B::Device) -> CatDogBatch<B> {
        let batch = items.len();
        let s = self.image_size;
        let mut img = Vec::with_capacity(batch * 3 * s * s);
        let mut lbl = Vec::with_capacity(batch);

        for item in items {
            lbl.push(item.label as i32);
            let im = image::open(&item.path).unwrap()
                .resize_exact(s as u32, s as u32, FilterType::Triangle)
                .to_rgb8();
            for c in 0..3usize {
                for y in 0..s {
                    for x in 0..s {
                        let p = im.get_pixel(x as u32, y as u32);
                        img.push((p[c] as f32 / 127.5) - 1.0);
                    }
                }
            }
        }

        CatDogBatch {
            images: Tensor::<B, 4>::from_data(TensorData::new(img, [batch, 3, s, s]), &self.device),
            labels: Tensor::<B, 1, Int>::from_data(TensorData::new(lbl, [batch]), &self.device),
        }
    }
}