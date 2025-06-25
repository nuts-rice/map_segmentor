use image::GenericImage;
use image::GenericImageView;
use image::imageops::FilterType;
use image::{Pixel, Rgba, RgbaImage};
use ndarray::{Array, ArrayBase, ArrayD, Axis, IxDyn};
use ort::environment::Environment;
use ort::session::Session;
use ort::session::SessionOutputs;
use ort::session::builder::SessionBuilder;
use ort::value::Tensor;
use ort::value::Value;
use std::sync::{Arc, Mutex};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
mod map_render;
//TODO: consult https://github.com/jamjamjon/usls/blob/main/examples/yolo-sam2/main.rs see if usls
//and yolo-obb
//would work
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SegmentLabel {
    Building,
    Road,
    Water,
    SwimmingPool,
    Vegetation,
    TennisCourt,
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
        let colors = [
            Rgba([255, 0, 0, 128]), // Red - Buildings
            Rgba([0, 0, 255, 128]), // Blue - Roads
            Rgba([0, 255, 255, 128]),
            Rgba([0, 255, 255, 128]),   // Cyan - Water
            Rgba([0, 255, 0, 128]),     // Green - Vegetation
            Rgba([255, 255, 0, 128]),   // Yellow - Tennis Court
            Rgba([128, 128, 128, 128]), // Gray - Other
        ];
        for segment in segments {
            let color_idx = match segment.segment_type {
                Some(SegmentLabel::Building) => colors[0],
                Some(SegmentLabel::Road) => colors[1],
                Some(SegmentLabel::Water) => colors[2],
                Some(SegmentLabel::SwimmingPool) => colors[3],
                Some(SegmentLabel::Vegetation) => colors[4],
                Some(SegmentLabel::TennisCourt) => colors[5],
                Some(SegmentLabel::Other) | None => colors[6],
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

    async fn detect_obb(
        &mut self,
        input_img_path: &str,
        //output_img_path: &str
    ) -> Result<RgbaImage> {
        let img = image::open(input_img_path)?;

        let (embeddings) = self.encode_image(input_img_path).unwrap();
        let (predictions) = &embeddings.0;
        println!("predictions shape: {:?}", predictions.shape());
        let mut data: Vec<Segment> =  Vec::new();
        for (idx, row) in predictions.axis_iter(Axis(0)).enumerate() {
            let original_width = self.img_width as f32;
            let original_height = self.img_height as f32;
            let ratio = (1024. / original_width).min(1024. / original_height);
            for pred in row.axis_iter(Axis(1)) {
                let bbox = pred.slice(ndarray::s![0..4]);

                let class_scores = pred.slice(ndarray::s![4..4 + self.class_labels.len()]);
                let (id, &confidence) = class_scores
                    .iter()
                    .enumerate()
                    .reduce(|max, x| if x.1 > max.1 { x } else { max })
                    .unwrap();

                if confidence < self.threshold {
                    continue;
                }
                let cx = bbox[0] / ratio;
                let cy = bbox[1] / ratio;
                let w = bbox[2] / ratio;
                let h = bbox[3] / ratio;
                let x = cx - w / 2.;
                let y = cy - h / 2.;
                let bb = BbCoords {
                    x1: x.max(0.0).min(original_width),
                    y1: y.max(0.0).min(original_height),
                    x2: x + w,
                    y2: y + h,
                };
                let segment = Segment {
                    coords: bb,
                    segment_type: self.class_labels.get(id).and_then(|label| {
                        match label.as_str() {
                            "building" => Some(SegmentLabel::Building),
                            "road" => Some(SegmentLabel::Road),
                            "swimming pool" => Some(SegmentLabel::SwimmingPool),
                            "water" => Some(SegmentLabel::Water),
                            "vegetation" => Some(SegmentLabel::Vegetation),
                            "tennis court" => Some(SegmentLabel::TennisCourt),
                            _ => None,
                        }
                    }),

                    confidence: Some(confidence),
                    mask: Array::zeros((1, 1, 1024, 1024)).into_dyn(), // Placeholder for mask
                                                                       //self
                                                                       //    .generate_mask_for_point(cx, cy, original_width, original_height, &embeddings.0)
                                                                       //    .await?,
                };
                data.push(segment);
            }
            println!("Segment {}: {:?}", idx, data);
            
        }


        let mut output_img = RgbaImage::new(1024, 1024);

        output_img.copy_from(&img, 0, 0).unwrap();
        output_img = self.letterbox(&output_img.into(), 1024, 1024).0.into();
        self.create_mask_overlay(&mut output_img, &data);

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
