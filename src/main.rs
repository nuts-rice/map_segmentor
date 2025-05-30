use image::GenericImageView;
use image::imageops::FilterType;
use ort::Environment;
use ort::SessionBuilder;
use ort::Value;
use ort::session::Session;
use std::sync::Arc;

use ndarray::{Array, ArrayBase, Axis, IxDyn};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
mod map_render;
enum SegmentLabel {
    Building,
    Road,
    Water,
    Vegetation,
    Other,
}

type BbCoords = (f32, f32, f32, f32);

struct Segment {
    coords: BbCoords,
    segment_type: Option<SegmentLabel>,
    confidence: f32,
}

struct Segmentor {
    encoder: Session,
    threshold: f32,
}

impl Segmentor {
    pub fn new(model_path: &str, threshold: f32) -> Self {
        let env = Arc::new(
            Environment::builder()
                .with_name("sam2.1_large")
                .build()
                .unwrap(),
        );
        let encoder = SessionBuilder::new(&env)
            .unwrap()
            .with_model_from_file(model_path)
            .unwrap();

        Self { encoder, threshold }
    }

    pub fn process_tile(&mut self) {
        unimplemented!()
    }

    pub fn download_tile_map(&self) {
        unimplemented!()
    }

    async fn classify_segments(&mut self, segments: Vec<Segment>) -> Result<Vec<Segment>> {
        let mut classified_segments: Vec<Segment> = Vec::new();
        //for segment in segments {

        //}
        Ok(classified_segments)
    }

    async fn generate_segments(&mut self, image: &str) -> Result<Vec<Segment>> {
        let mut segments: Vec<Segment> = Vec::new();
        let img = image::open(image).unwrap();
        let (img_w, img_h) = (img.width() as f32, img.height() as f32);
        let img_resized = img.resize(1024, 1024, FilterType::CatmullRom);
        let (resized_width, resized_height) =
            (img_resized.width() as f32, img_resized.height() as f32);

        // Copy the image pixels to the tensor, normalizing them using mean and standard deviations
        // for each color channel
        let mut input = Array::zeros((1, 3, 1024, 1024)).into_dyn();
        let mean = vec![123.675, 116.28, 103.53];
        let std = vec![58.395, 57.12, 57.375];
        for pixel in img_resized.pixels() {
            let x = pixel.0 as usize;
            let y = pixel.1 as usize;
            let [r, g, b, _] = pixel.2.0;
            input[[0, 0, y, x]] = (r as f32 - mean[0]) / std[0];
            input[[0, 1, y, x]] = (g as f32 - mean[1]) / std[1];
            input[[0, 2, y, x]] = (b as f32 - mean[2]) / std[2];
        }

        // Prepare tensor for the SAM encoder model
        /*
                let outputs: SessionOutputs = self.encoder.run(inputs!["images" => TensorRef::from_array_view(&input)?])?;
                    let mut segment_bbs: Vec<BbCoords> = Vec::new();
                    for row in outputs.axis_iter(Axis(0)) {
                        let row: Vec<_> = row.iter().copied().collect();
                }
        */

        /*
        let encoder_outputs = self.encoder.run(input).unwrap();
        let embeddings = encoder_outputs
            .get(0).unwrap()
            .try_extract::<f32>()
            .ok().unwrap()
            .view()
            .t()
            .reversed_axes()
            .into_owned();
        */

        return Ok(segments);
    }

    pub async fn encode_image(
        &mut self,
        image: &str,
    ) -> Option<(Array<f32, IxDyn>, f32, f32, f32, f32)> {
        let img = image::open(image).ok()?;
        let (img_w, img_h) = (img.width() as f32, img.height() as f32);
        let img_resized = img.resize(1024, 1024, FilterType::CatmullRom);
        let (resized_width, resized_height) =
            (img_resized.width() as f32, img_resized.height() as f32);

        // Copy the image pixels to the tensor, normalizing them using mean and standard deviations
        // for each color channel
        let mut input = Array::zeros((1, 3, 1024, 1024)).into_dyn();
        let mean = vec![123.675, 116.28, 103.53];
        let std = vec![58.395, 57.12, 57.375];
        for pixel in img_resized.pixels() {
            let x = pixel.0 as usize;
            let y = pixel.1 as usize;
            let [r, g, b, _] = pixel.2.0;
            input[[0, 0, y, x]] = (r as f32 - mean[0]) / std[0];
            input[[0, 1, y, x]] = (g as f32 - mean[1]) / std[1];
            input[[0, 2, y, x]] = (b as f32 - mean[2]) / std[2];
        }

        // Prepare tensor for the SAM encoder model
        let input_as_values = &input.as_standard_layout();
        let encoder_inputs =
            vec![Value::from_array(self.encoder.allocator(), input_as_values).ok()?];
        let encoder_outputs = self.encoder.run(encoder_inputs).ok()?;
        let embeddings = encoder_outputs
            .get(0)?
            .try_extract::<f32>()
            .ok()?
            .view()
            .t()
            .reversed_axes()
            .into_owned();
        return Some((embeddings, img_w, img_h, resized_width, resized_height));
    }

    async fn detect_buildings(&mut self, image: &str) -> Result<()> {
        let mut buildings: Vec<Segment> = Vec::new();

        unimplemented!();
    }
}
//use ort::Result;
pub fn main() -> Result<()> {
    let img_path = "../Screenshot_2025-05-27_16-39-42.png";
    let mut segmentor = Segmentor::new("./models/sam2.1_large.onnx", 0.5);
    map_render::run();
    Ok(())
}
