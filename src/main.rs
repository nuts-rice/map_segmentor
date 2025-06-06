use image::GenericImageView;
use image::imageops::FilterType;
use image::{Pixel, Rgba, RgbaImage};
use ort::Environment;
use ort::SessionBuilder;
use ort::Value;

use ort::session::Session;
use std::sync::{Arc, Mutex};

use ndarray::{Array, ArrayBase, ArrayD, Axis, IxDyn};

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
    mask: Array<u8, IxDyn>,
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

    pub async fn segment_image(&mut self, image_path: &str) -> Result<(RgbaImage, Vec<Segment>)> {
        let (embeddings, img_w, img_h) = self
            .encode_image(image_path)
            .await
            .ok_or("Failed to encode image")
            .unwrap();
        let img = image::open(image_path)?.to_rgba8();
        let mut segments = self.generate_segments(img_w, img_h, &embeddings).await?;
        self.classify_segments(&mut segments).await?;
        let mask_overlay = self.create_mask_overlay(&img, &segments);
        Ok((mask_overlay, segments))
    }

    async fn classify_segments(&mut self, segments: &mut [Segment]) -> Result<Vec<Segment>> {
        let mut classified_segments: Vec<Segment> = Vec::new();
        //for segment in segments {

        //}
        Ok(classified_segments)
    }

    fn create_mask_overlay(&self, img: &RgbaImage, segments: &[Segment]) -> RgbaImage {
        let mut overlay = img.clone();
        let colors = [
            Rgba([255, 0, 0, 128]),     // Red - Buildings
            Rgba([0, 0, 255, 128]),     // Blue - Roads
            Rgba([0, 255, 255, 128]),   // Cyan - Water
            Rgba([0, 255, 0, 128]),     // Green - Vegetation
            Rgba([128, 128, 128, 128]), // Gray - Other
        ];
        for segment in segments {
            let color_idx = match segment.segment_type {
                Some(SegmentLabel::Building) => 0,
                Some(SegmentLabel::Road) => 1,
                Some(SegmentLabel::Water) => 2,
                Some(SegmentLabel::Vegetation) => 3,
                Some(SegmentLabel::Other) | None => 4,
            };
        }

        overlay
    }

    async fn generate_mask_for_point(
        &self,
        x: f32,
        y: f32,
        img_w: f32,
        img_h: f32,
    ) -> Result<Array<u8, IxDyn>> {
        let embeddings = self.embeddings.lock().unwrap();
        let embeddings_as_values = embeddings.as_standard_layout();
        let decoder = &self.decoder;
        // Prepare point input (normalized coordinates)
        let point_input = Array::from_shape_vec((1, 1, 2), vec![x / img_w, y / img_h])
            .unwrap()
            .into_dyn();
        let points_as_values = point_input.as_standard_layout();
        let point_labels = Array::from_shape_vec((1, 2), vec![2.0_f32, 3.0_f32])
            .unwrap()
            .into_dyn()
            .into_owned();
        let point_labels_as_values = &point_labels.as_standard_layout();
        let point_tensor = Value::from_array(self.decoder.allocator(), &points_as_values).unwrap();
        let embeddings_tensor =
            Value::from_array(self.decoder.allocator(), &embeddings_as_values).unwrap();
        let outputs = decoder
            .run(vec![
                embeddings_tensor,
                point_tensor,
                Value::from_array(self.decoder.allocator(), &point_labels_as_values).unwrap(),
            ])
            .unwrap();
        let output = outputs
            .get(0)
            .unwrap()
            .try_extract::<f32>()
            .unwrap()
            .view()
            .t()
            .reversed_axes()
            .into_dyn()
            .into_owned();
        let binary_mask = output.mapv(|v| if v > self.threshold { 1u8 } else { 0u8 });
        Ok(binary_mask)
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
        let grid_size = 10;
        let step_x = img_w / grid_size as f32;
        let step_y = img_h / grid_size as f32;
        let mut points = Vec::new();
        for i in 0..grid_size {
            for j in 0..grid_size {
                points.push((i as f32 * step_x, j as f32 * step_y));
            }
        }
        for (x, y) in points {
            let mask = self
                .generate_mask_for_point(x, y, img_w, img_h)
                .await
                .unwrap();

            if let Some(bbox) = self.mask_to_bbox(&mask, img_w, img_h) {
                let segment = Segment {
                    coords: bbox,
                    segment_type: None,
                    confidence: None,
                    mask: mask,
                };
                segments.push(segment);
            }
        }

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
    fn draw_bounding_box(&self, image: &mut RgbaImage, bbox: BbCoords, color: Rgba<u8>) {
        let (x1, y1, x2, y2) = (
            bbox.x1 as u32,
            bbox.y1 as u32,
            bbox.x2 as u32,
            bbox.y2 as u32,
        );

        // Draw horizontal lines
        for x in x1..=x2 {
            if x < image.width() {
                if y1 < image.height() {
                    image.put_pixel(x, y1, color);
                }
                if y2 < image.height() {
                    image.put_pixel(x, y2, color);
                }
            }
        }

        // Draw vertical lines
        for y in y1..=y2 {
            if y < image.height() {
                if x1 < image.width() {
                    image.put_pixel(x1, y, color);
                }
                if x2 < image.width() {
                    image.put_pixel(x2, y, color);
                }
            }
        }
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
