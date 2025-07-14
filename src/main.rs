use galileo::Color;
use galileo::galileo_types::cartesian::{Point2, Rect};
use galileo::galileo_types::geometry::{CartesianGeometry2d, Geom, Geometry};
use galileo::galileo_types::impls::{MultiPolygon, Polygon};
use image::GenericImage;
use image::GenericImageView;
use image::imageops::FilterType;
use image::{Pixel, Rgba, RgbaImage};
use ndarray::{Array, ArrayBase, ArrayD, ArrayView1, Axis, IxDyn};
use ort::environment::Environment;
use ort::session::Session;
use ort::session::SessionOutputs;
use ort::session::builder::SessionBuilder;
use ort::value::Tensor;
use ort::value::Value;
use serde::{Deserialize, Deserializer, Serialize};
use std::sync::{Arc, Mutex};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
mod map_render;
//TODO: consult https://github.com/jamjamjon/usls/blob/main/examples/yolo-sam2/main.rs see if usls
//and yolo-obb
//would work

pub mod model;
use model::geometry::{Segment, SegmentLabel, SegmentMask};

const YOLOV8M_URL: &str = "https://cdn.pyke.io/0/pyke:ort-rs/example-models@0.0.0/yolov8m.onnx";
const YOLOV11_OBB_PATH: &str = "./models/yolo11n-obb.onnx";
const YOLOV11_OBB_CLASS_LABELS: [&str; 7] = [
    "building",
    "road",
    "swimming pool",
    "water",
    "vegetation",
    "tennis court",
    "other",
];

struct Segmentor {
    encoder: Session,
    decoder: Session,

    img_width: u32,
    img_height: u32,
    batch: u32,
    iou: f32,
    class_labels: Vec<String>,

    threshold: f32,
}

impl Segmentor {
    pub fn new(
        model_path: &str,
        img_width: u32,
        img_height: u32,
        batch: u32,
        iou: f32,
        class_labels: Vec<String>,
        threshold: f32,
    ) -> Self {
        let encoder = Session::builder()
            .unwrap()
            .commit_from_file(model_path)
            //.commit_from_url(YOLOV8M_URL)
            .unwrap();
        let decoder = Session::builder()
            .unwrap()
            .commit_from_file(model_path)
            //.commit_from_url(YOLOV8M_URL)
            .unwrap();

        Self {
            encoder,
            decoder,
            img_height,
            img_width,
            batch,
            iou,
            class_labels,
            threshold,
        }
    }

    //TODO
    pub fn process_tile(&mut self) {
        unimplemented!()
    }

    async fn classify_segments(&mut self, segments: &mut [Segment]) -> Result<Vec<Segment>> {
        let mut classified_segments: Vec<Segment> = Vec::new();
        //for segment in segments {

        //}
        Ok(classified_segments)
    }

    fn create_mask_overlay(&self, img: &RgbaImage, segments: &[Segment]) -> RgbaImage {
        let mut overlay = img.clone();
        let colors: Vec<Color> = vec![
            Color::rgba(255, 0, 0, 0), // Red - Buildings
            Color::rgba(0, 0, 255, 0), // Blue - Roads
            Color::rgba(0, 255, 255, 0),
            Color::rgba(0, 255, 255, 0),   // Cyan - Water
            Color::rgba(0, 255, 0, 0),     // Green - Vegetation
            Color::rgba(255, 255, 0, 0),   // Yellow - Tennis Court
            Color::rgba(128, 128, 128, 0), // Gray - Other
        ];
        for segment in segments {
            let segment_type = segment.segment_type.as_ref().unwrap();
            let color_idx = match segment_type {
                (SegmentLabel::Building) => colors[0],
                (SegmentLabel::Road) => colors[1],
                (SegmentLabel::Water) => colors[2],
                (SegmentLabel::SwimmingPool) => colors[3],
                (SegmentLabel::Vegetation) => colors[4],
                (SegmentLabel::TennisCourt) => colors[5],
                (SegmentLabel::Other) => colors[6],
            };
            //segment.color = Some(color_idx);

            //self.apply_mask_to_image(&mut overlay, segment_type, color_idx);
        }

        overlay
    }

    fn apply_mask_to_image(&self, img: &mut RgbaImage, mask: &SegmentMask, color: Color) {
        let (width, height) = (img.width() as usize, img.height() as usize);

        /*
         let mask_view = mask.view().into_dyn();

                for (y, row) in mask_view.axis_iter(Axis(0)).enumerate() {
                    for (x, &value) in row.iter().enumerate() {
                        if value > 0 && x < width && y < height {
                            let pixel = img.get_pixel_mut(x as u32, y as u32);
                            pixel.blend(&color);
                        }

                    }
                }
        */
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
        let point_input =
            Array::from_shape_vec((1, 1, 1, 2), vec![x / img_w, y / img_h])?.into_dyn();
        let point_labels = Array::from_shape_vec((1, 1, 1, 1), vec![1.0])
            .unwrap()
            .into_dyn();
        //let embeddings_tensor = Tensor::from_array(ndarray::Array4::<f32>::from_shape_vec(
        //))
        let inputs = ort::inputs![
            //ort::value::TensorRef::from_array_view(&point_input)?,
            //ort::value::TensorRef::from_array_view(&point_labels)?,
            ort::value::TensorRef::from_array_view(embeddings)?,
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

    fn mask_to_bbox(&self, mask: &Array<u8, IxDyn>, img_w: f32, img_h: f32) -> Option<Rect> {
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
            Some(Rect::new(
                min_x.into(),
                min_y.into(),
                max_x.into(),
                max_y.into(),
            ))
        } else {
            None
        }
    }

    fn prediction_to_segment(
        &self,
        prediction: &Array<f32, IxDyn>,
        scale_factor: f32,
    ) -> Option<Segment> {
        let bbox = prediction.slice(ndarray::s![0..4]);
        let class_scores = prediction.slice(ndarray::s![4..4 + self.class_labels.len()]);
        let (id, &confidence) = class_scores
            .iter()
            .enumerate()
            .reduce(|max, x| if x.1 > max.1 { x } else { max })?;

        if confidence < self.threshold {
            return None;
        }

        let [cx, cy, w, h] = [bbox[0], bbox[1], bbox[2], bbox[3]];
        let (x, y) = ((cx - w / 2.) * scale_factor, (cy - h / 2.) * scale_factor);
        let (w, h) = (w * scale_factor, h * scale_factor);

        let bbox_rect = Rect::new(x.into(), y.into(), (x + w).into(), (y + h).into());

        let segment_type = self
            .class_labels
            .get(id)
            .and_then(|label| match label.as_str() {
                "building" => Some(SegmentLabel::Building),
                "road" => Some(SegmentLabel::Road),
                "swimming pool" => Some(SegmentLabel::SwimmingPool),
                "water" => Some(SegmentLabel::Water),
                "vegetation" => Some(SegmentLabel::Vegetation),
                "tennis court" => Some(SegmentLabel::TennisCourt),
                _ => None,
            });
        Some(Segment {
            bbox: bbox_rect,
            segment_type,
            color: None,
            confidence: Some(confidence),
            mask: None, // Placeholder for mask
        })
    }

    async fn detect_obb(
        &mut self,
        input_img_path: &str,
        //output_img_path: &str
    ) -> Result<RgbaImage> {
        let img = image::open(input_img_path)?;

        let (embeddings) = self.encode_image(input_img_path).unwrap();
        let (predictions) = &embeddings.0;
        let scale_factor = (1024. / self.img_width as f32).min(1024. / self.img_height as f32);

        println!("predictions shape: {:?}", predictions.shape());
        let segment_iter = predictions
            .axis_iter(Axis(0))
            //.flat_map(|batch| batch.axis_iter(Axis(1))).collect::<Vec<_>>()
            .filter_map(|pred| self.prediction_to_segment(&pred.to_owned(), scale_factor));
        let segments = segment_iter.collect::<Vec<_>>();

        let mut output_img = RgbaImage::new(1024, 1024);

        output_img.copy_from(&img, 0, 0).unwrap();
        output_img = self.letterbox(&output_img.into(), 1024, 1024).0.into();
        self.create_mask_overlay(&mut output_img, &segments);

        /*
               self.create_mask_overlay(&mut output_img, &segments);

                output_img.save(output_img_path)?;
        */
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
                /*
                                let segment = Segment {
                                    bbox,
                                    geometry: MultiPolygon::from(Polygon::from(bbox)).into(),
                                    segment_type: None, // Placeholder for segment type
                                    color: None,        // Placeholder for color
                                    confidence: None,   // Placeholder for confidence
                                    mask: Some(SegmentMask::from(mask)),
                                };
                                segments.push(segment);
                */
            }
        }

        Ok(segments)
    }
    fn letterbox(
        &self,
        img: &image::DynamicImage,
        width: u32,
        height: u32,
    ) -> (image::DynamicImage, f32, f32) {
        let (w, h) = (img.width() as f32, img.height() as f32);
        let scale = (width as f32 / w).min(height as f32 / h);
        let new_w = (w * scale).round() as u32;
        let new_h = (h * scale).round() as u32;

        let resized = img.resize_exact(new_w, new_h, FilterType::Triangle);

        let pad_left = ((width - new_w) as f32 / 2.0).round() as u32;
        let pad_top = ((height - new_h) as f32 / 2.0).round() as u32;

        let mut canvas = image::DynamicImage::new_rgb8(width, height);
        canvas.copy_from(&resized, pad_left, pad_top).unwrap();

        (canvas, pad_left as f32, pad_top as f32)
    }

    //TODO:
    //SESSION TYPED encoder tx img

    fn encode_image(&mut self, image: &str) -> Option<(Array<f32, IxDyn>, f32, f32)> {
        let img = image::open(image).ok()?;
        let (img_w, img_h) = (img.width() as f32, img.height() as f32);
        let (resized, pad_left, pad_top) = self.letterbox(&img, 1024, 1024);
        let (resized_w, resized_h) = (resized.width() as f32, resized.height() as f32);

        // Copy the image pixels to the tensor, normalizing them using mean and standard deviations
        // for each color channel
        let mut input = Array::zeros((1, 3, 1024, 1024));
        for (x, y, rgb) in resized.pixels() {
            let x = x as usize;
            let y = y as usize;
            let [r, g, b, _] = rgb.0;
            input[[0, 0, y, x]] = r as f32 / 255.0;
            input[[0, 1, y, x]] = g as f32 / 255.0;
            input[[0, 2, y, x]] = b as f32 / 255.0;
        }
        let now = std::time::Instant::now();
        let outputs: SessionOutputs = self
            .encoder
            .run(ort::inputs![
                ort::value::TensorRef::from_array_view(&input).unwrap()
            ])
            .unwrap();
        println!(
            "outputs: {:?}, inference time: {:?} ",
            outputs.len(),
            now.elapsed()
        );
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
}

//use ort::Result;
#[tokio::main]
pub async fn main() -> Result<()> {
    let img_path = "./buildings4.png";
    let img = image::open(img_path)?;
    let (img_w, img_h) = img.dimensions();
    let mut segmentor = Segmentor::new(
        YOLOV11_OBB_PATH,
        img_w,
        img_h,
        1,
        0.5,
        YOLOV11_OBB_CLASS_LABELS
            .iter()
            .map(|s| s.to_string())
            .collect(),
        0.5,
    );
    let timestamp = chrono::Utc::now();

    //TODO: Fix encoding error
    let output_img = segmentor
        .detect_obb(
            img_path,
            //&output_path
        )
        .await?;
    output_img.save(format!("./output/{}output.png", timestamp))?;

    map_render::run();
    Ok(())
}
