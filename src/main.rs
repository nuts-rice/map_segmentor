use image::GenericImageView;
use image::imageops::FilterType;
use image::{Pixel, Rgba, RgbaImage};
use ort::environment::Environment;
use ort::session::Session;
use ort::session::SessionOutputs;
use ort::session::builder::SessionBuilder;
use ort::value::Tensor;
use ort::value::Value;
use std::sync::{Arc, Mutex};

use ndarray::{Array, ArrayBase, ArrayD, Axis, IxDyn};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
mod map_render;
//TODO: consult https://github.com/jamjamjon/usls/blob/main/examples/yolo-sam2/main.rs see if usls
//and yolo
//would work
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

#[derive(Debug, Clone)]
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

const YOLOV8M_URL: &str = "https://cdn.pyke.io/0/pyke:ort-rs/example-models@0.0.0/yolov8m.onnx";

struct Segmentor {
    encoder: Session,
    decoder: Session,
    threshold: f32,
}

impl Segmentor {
    pub fn new(model_path: &str, threshold: f32) -> Self {
        let env = Arc::new(ort::init().with_name("sam2.1_large").commit().unwrap());
        let encoder = Session::builder()
            .unwrap()
            .commit_from_url(YOLOV8M_URL)
            .unwrap();
        let decoder = Session::builder()
            .unwrap()
            .commit_from_url(YOLOV8M_URL)
            .unwrap();

        Self {
            encoder,
            decoder,
            threshold,
        }
    }

    //TODO
    pub fn process_tile(&mut self) {
        unimplemented!()
    }

    pub async fn segment_image(&mut self, image_path: &str) -> Result<(RgbaImage, Vec<Segment>)> {
        let embeddings = self.encode_image(image_path).unwrap();
        let (img_w, img_h) = {
            let img = image::open(image_path)?;
            img.dimensions()
        };
        let img = image::open(image_path)?.to_rgba8();
        let embeddings_input = embeddings.0;
        let (resized_w, resized_h) = (embeddings.1, embeddings.2);
        let mut segments = self
            .generate_segments(&embeddings_input, resized_w, resized_h)
            .await?;
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
                Some(SegmentLabel::Building) => colors[0],
                Some(SegmentLabel::Road) => colors[1],
                Some(SegmentLabel::Water) => colors[2],
                Some(SegmentLabel::Vegetation) => colors[3],
                Some(SegmentLabel::Other) | None => colors[4],
            };
            self.apply_mask_to_image(&mut overlay, &segment.mask, color_idx);
        }

        overlay
    }

    fn apply_mask_to_image(&self, img: &mut RgbaImage, mask: &ArrayD<u8>, color: Rgba<u8>) {
        let (width, height) = (img.width() as usize, img.height() as usize);
        let mask_view = mask.view().into_dyn();

        for (y, row) in mask_view.axis_iter(Axis(0)).enumerate() {
            for (x, &value) in row.iter().enumerate() {
                if value > 0 && x < width && y < height {
                    let pixel = img.get_pixel_mut(x as u32, y as u32);
                    pixel.blend(&color);
                }
            }
        }
    }

    async fn generate_mask_for_point(
        &mut self,
        x: f32,
        y: f32,
        img_w: f32,
        img_h: f32,
        embeddings: &Array<f32, IxDyn>,
    ) -> Result<Array<u8, IxDyn>> {
        // Prepare point input (normalized coordinates)
        let point_input = Array::from_shape_vec((1, 1, 2), vec![x / img_w, y / img_h])?.into_dyn();
        let points_as_values = point_input.as_standard_layout();
        let point_labels = Array::from_shape_vec((1, 1), vec![1.0]).unwrap().into_dyn();

        let point_labels = point_labels.as_standard_layout();
        let embeddings = embeddings.as_standard_layout();
        let inputs = ort::inputs![
            //ort::value::TensorRef::from_array_view(&point_input).unwrap(),
            //ort::value::TensorRef::from_array_view(&point_labels).unwrap(),
            ort::value::TensorRef::from_array_view(&embeddings).unwrap(),
        ];

        let outputs = self.decoder.run(inputs).unwrap();
        let output = outputs[0]
            .try_extract_array::<f32>()
            .unwrap()
            // Run YOLOv8 inference
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

    async fn generate_segmented_img(
        &mut self,
        input_img_path: &str,
        //output_img_path: &str
    ) -> Result<RgbaImage> {
        let img = image::open(input_img_path)?;
        let (img_w, img_h) = img.dimensions();

        let (embeddings) = self.encode_image(input_img_path).unwrap();
        let embeddings = embeddings;
        let (resized_w, resized_h) = (embeddings.1, embeddings.2);
        let embeddings = embeddings.0;
        let segments = self
            .generate_segments(&embeddings, resized_w as f32, resized_h as f32)
            .await?;
        let mut output_img = RgbaImage::new(resized_w as u32, resized_h as u32);

        self.create_mask_overlay(&mut output_img, &segments);
        //output_img.save(output_img_path)?;
        Ok(output_img)
    }

    //TODO
    async fn generate_segments(
        &mut self,
        embeddings: &Array<f32, IxDyn>,
        resized_w: f32,
        resized_h: f32,
    ) -> Result<Vec<Segment>> {
        let mut segments: Vec<Segment> = Vec::new();
        let grid_size = 10;
        let step_x = resized_w / grid_size as f32;
        let step_y = resized_h / grid_size as f32;
        let mut points = Vec::new();
        for i in 0..grid_size {
            for j in 0..grid_size {
                points.push((i as f32 * step_x, j as f32 * step_y));
            }
        }
        for (x, y) in points {
            let mask = self
                .generate_mask_for_point(x, y, resized_w, resized_h, embeddings)
                .await
                .unwrap();

            if let Some(bbox) = self.mask_to_bbox(&mask, resized_w, resized_h) {
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

    fn encode_image(&mut self, image: &str) -> Option<(Array<f32, IxDyn>, f32, f32)> {
        let img = image::open(image).ok()?;
        let (img_w, img_h) = (img.width() as f32, img.height() as f32);
        let img_resized = img.resize_exact(640, 640, FilterType::CatmullRom);

        // Copy the image pixels to the tensor, normalizing them using mean and standard deviations
        // for each color channel
        let mut input = Array::zeros((1, 3, 640, 640));
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
        //        let input_as_values = &input.as_standard_layout();
        // let encoder_inputs =
        //            vec![Value::from_array(self.encoder.allocator(), input_as_values).ok()?];

        let outputs: SessionOutputs = self
            .encoder
            .run(ort::inputs![
                ort::value::TensorRef::from_array_view(&input).unwrap()
            ])
            .unwrap();

        //let input_as_values = &input.as_standard_layout();
        //let encoder_inputs = vec![Value::from_array(self.encoder.allocator(), input_as_values).ok()?];
        //println!("encoder_inputs: {:?}", encoder_inputs);
        println!("outputs: {:?} ", outputs.len());
        let embeddings = outputs[0]
            .try_extract_array::<f32>()
            .ok()?
            .view()
            .t()
            .reversed_axes()
            .into_owned();
        println!("embeddings: {:?}", embeddings);
        return Some((embeddings, img_h, img_w));
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
    let img_path = "./buildings2.png";
    let mut segmentor = Segmentor::new("./models/sam2.1_large.onnx", 0.5);
    let timestamp = chrono::Utc::now();
    let run_dir_path = "./runs/sam2.1_large";
    let embeddings = segmentor.encode_image(img_path).unwrap();
    //let output_path = format!("{}/output{}.png", run_dir_path, timestamp);
    //TODO: Fix encoding error
    segmentor
        .generate_segmented_img(
            img_path,
            //&output_path
        )
        .await?;

    //let _ = segmentor.encode_image(img_path).await;
    //let (mask_overlay, segments) = segmentor.segment_image(img_path).await?;
    //println!("Segments: {:?}", segments);
    map_render::run();
    Ok(())
}
