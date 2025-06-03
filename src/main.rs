use image::GenericImageView;
use image::Pixel;
use image::imageops::FilterType;
use ort::Environment;
use ort::SessionBuilder;
use ort::Value;

use ort::session::Session;
use std::sync::{Arc, Mutex};

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

#[derive(Debug, Clone, Copy)]
struct BbCoords {
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
}

struct Segment {
    coords: BbCoords,
    segment_type: Option<SegmentLabel>,
    confidence: Option<f32>,
    mask: Option<Array<u8, IxDyn>>,
}

impl BbCoords {
    fn new(x1: f32, y1: f32, x2: f32, y2: f32) -> Self {
        Self { x1, y1, x2, y2 }
    }
    fn area(&self) -> f32 {
        (self.x2 - self.x1) * (self.y2 - self.y1)
    }

    fn intersection(&self, other: &Self) -> f32 {
        let x_overlap = (self.x2.min(other.x2) - self.x1.max(other.x1)).max(0.0);
        let y_overlap = (self.y2.min(other.y2) - self.y1.max(other.y1)).max(0.0);
        x_overlap * y_overlap
    }

    fn union(&self, other: &Self) -> f32 {
        self.area() + other.area() - self.intersection(other)
    }
}

struct Segmentor {
    encoder: Session,
    decoder: Session,
    embeddings: Mutex<Array<f32, IxDyn>>,
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
        let decoder = SessionBuilder::new(&env)
            .unwrap()
            .with_model_from_file(model_path)
            .unwrap();
        let embeddings = Mutex::new(ArrayBase::zeros((1, 256, 64, 64)).into_dyn());

        Self {
            encoder,
            decoder,
            embeddings,
            threshold,
        }
    }

    //TODO
    pub fn process_tile(&mut self) {
        unimplemented!()
    }

    async fn classify_segments(&mut self, segments: Vec<Segment>) -> Result<Vec<Segment>> {
        let mut classified_segments: Vec<Segment> = Vec::new();
        //for segment in segments {

        //}
        Ok(classified_segments)
    }

    async fn generate_mask_for_point(
        &self,
        x: f32,
        y: f32,
        img_w: f32,
        img_h: f32,
        embeddings: &Mutex<Array<f32, IxDyn>>,
    ) -> Result<Option<Segment>> {
        // Prepare point input (normalized coordinates)
        let point_input = Array::from_shape_vec((1, 1, 2), vec![x / img_w, y / img_h])?.into_dyn();
        let points_as_values = point_input.as_standard_layout();
        let point_labels = Array::from_shape_vec((1, 2), vec![2.0_f32, 3.0_f32])
            .unwrap()
            .into_dyn()
            .into_owned();
        let point_labels_as_values = &point_labels.as_standard_layout();
        /*
                let embeddings


                // Run decoder
                let embeddings_tensor = Value::from_array(self.decoder.allocator(), embeddings)?;

                let outputs = self.decoder.run(vec![embeddings_tensor, point_tensor])?;
                let mask_output = outputs.get(0).ok_or("No mask output")?;
                let mask = mask_output.try_extract::<f32>()?.into_dyn();

                // Threshold mask to binary
                let binary_mask = mask.mapv(|v| if v > 0.5 { 1u8 } else { 0u8 });

                // Calculate bounding box from mask
                if let Some(bbox) = self.mask_to_bbox(&binary_mask, img_w, img_h) {
                    let segment = Segment {
                        coords: bbox,
                        segment_type: None,
                        confidence: self.calculate_mask_confidence(&binary_mask),
                        mask: Some(binary_mask),
                    };
                    Ok(Some(segment))
                } else {
                    Ok(None)
                }
        */
        unimplemented!()
    }

    fn mask_to_bbox(&self, mask: &Array<u8, IxDyn>, img_w: f32, img_h: f32) -> Option<BbCoords> {
        let mut min_x = f32::MAX;
        let mut min_y = f32::MAX;
        let mut max_x = f32::MIN;
        let mut max_y = f32::MIN;
        let mut has_positive = false;

        let shape = mask.shape();
        let height = shape[2] as f32;
        let width = shape[3] as f32;
        for y in 0..shape[2] {
            for x in 0..shape[3] {
                if mask[[0, 0, y, x]] > 0 {
                    has_positive = true;
                    let x_val = x as f32 * img_w / width;
                    let y_val = y as f32 * img_h / height;
                    min_x = min_x.min(x_val);
                    min_y = min_y.min(y_val);
                    max_x = max_x.max(x_val);
                    max_y = max_y.max(y_val);
                }
            }
        }
        if has_positive {
            Some(BbCoords::new(min_x, min_y, max_x, max_y))
        } else {
            None
        }
    }

    //TODO
    async fn generate_segments(
        &mut self,
        img_w: f32,
        img_h: f32,
        embeddings: &Array<f32, IxDyn>,
    ) -> Result<Vec<Segment>> {
        let mut segments: Vec<Segment> = Vec::new();

        Ok(segments)
    }

    //TODO:
    //SESSION TYPED encoder tx img
    pub async fn encode_image(&mut self, image: &str) -> Option<(Array<f32, IxDyn>, f32, f32)> {
        let img = image::open(image).ok()?;
        let (img_w, img_h) = (img.width() as f32, img.height() as f32);
        let img_resized = img.resize_exact(1024, 1024, FilterType::CatmullRom);

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

        return Some((embeddings, img_w, img_h));
    }

    //TODO
    async fn detect_buildings(&mut self, image: &str) -> Result<()> {
        let mut buildings: Vec<Segment> = Vec::new();

        unimplemented!();
    }
}
//use ort::Result;
#[tokio::main]
pub async fn main() -> Result<()> {
    let img_path = "../Screenshot_2025-05-27_16-39-42.png";
    let mut segmentor = Segmentor::new("./models/sam2.1_large.onnx", 0.5);
    let _ = segmentor.encode_image(img_path).await;
    map_render::run();
    Ok(())
}
