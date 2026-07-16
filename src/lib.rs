pub mod backbone;
pub mod encoder;

pub use backbone::{Backbone, BackboneConfig};
pub use encoder::classification::ClassificationConfig;
pub use encoder::projection::ProjectionConfig;
pub use encoder::{Classification, Projection};