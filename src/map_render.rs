use crate::BbCoords;
use eframe::CreationContext;
use egui::FontDefinitions;
use galileo::control::{EventPropagation, MouseButton, UserEvent, UserEventHandler};
use galileo::galileo_types::cartesian::{Point2, Rect};
use galileo::galileo_types::geo::Crs;
use galileo::layer::VectorTileLayer;
use galileo::layer::feature_layer::{FeatureLayer, FeatureLayerOptions};
use galileo::layer::raster_tile_layer::RasterTileLayer;
use galileo::layer::raster_tile_layer::RasterTileLayerBuilder;
use galileo::layer::vector_tile_layer::VectorTileLayerBuilder;
use galileo::layer::vector_tile_layer::style::VectorTileStyle;
use galileo::render::text::RustybuzzRasterizer;
use galileo::render::text::text_service::TextService;
use galileo::symbol::ImagePointSymbol;
use galileo::tile_schema::{TileIndex, TileSchema, VerticalDirection};
use galileo::{Lod, Map, MapBuilder};
use galileo_egui::EguiMap;
use galileo_egui::EguiMapState;
use crate::Segment;
use parking_lot::RwLock;
use std::sync::Arc;

const MAPTILER_SAT_URL: &str = "https://api.maptiler.com/tiles/satellite-v2/{z}/{x}/{y}.jpg";
const ORIGIN: galileo::galileo_types::cartesian::Point2 =
    galileo::galileo_types::cartesian::Point2::new(-20037508.342787, 20037508.342787);

struct ColoredMask {
    width: u32,
    height: u32,
}

struct MapApp {
    map: EguiMapState,
    sat_layer: Arc<RwLock<RasterTileLayer>>,
}

impl eframe::App for MapApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            EguiMap::new(&mut self.map).show_ui(ui);
        });

        egui::Window::new("Buttons")
            .title_bar(false)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    /*                    if ui.button("Default style").clicked() {
                                            self.set_style(default_style());
                                        }
                                        if ui.button("Gray style").clicked() {
                                            self.set_style(gray_style());
                                        }
                    */
                });
            });
    }
}
impl MapApp {
    fn new(
        map: Map,
        layer: Arc<RwLock<RasterTileLayer>>,
        cc: &CreationContext,
        handler: impl UserEventHandler + 'static,
    ) -> Self {
        let fonts = FontDefinitions::default();
        let provider = RustybuzzRasterizer::default();

        let text_service = TextService::initialize(provider);
        for font in fonts.font_data.values() {
            text_service.load_font(Arc::new(font.font.to_vec()));
        }

        Self {
            map: EguiMapState::new(
                map,
                cc.egui_ctx.clone(),
                cc.wgpu_render_state.clone().expect("no render state"),
                [Box::new(handler) as Box<dyn UserEventHandler>],
            ),
            sat_layer: layer,
        }
    }
    /*fn set_style(&mut self, style: RasterTileStyle) {
        let mut layer = self.sat_layer.write();
        if style != *layer.style() {
            layer.update_style(style);
            self.map.request_redraw();
        }
    }
    */
}

pub fn run() {
    galileo_egui::InitBuilder::new(create_map())
        .init()
        .expect("Failed to init map");
}

pub fn build_obb_layer() -> Arc<RwLock<RasterTileLayer>> {
    unimplemented!()
}

struct SegmentMask{}
pub fn overlay_segmented_layer(obb_img: &RgbaImage, segments) -> Arc<RwLock<FeatureLayer<Point2, Segment, SegmentMask, CartesianSpace2d>>> {
    
}

pub fn download_sat_tile() -> Arc<RwLock<RasterTileLayer>> {
    let api_key = std::env::var("VT_API_KEY")
        .expect("Set MapTiler API key in VT_API_KEY environment variable");
    const ORIGIN: Point2 = Point2::new(-20037508.342787, 20037508.342787);
    const TOP_RESOLUTION: f64 = 156543.03392800014 / 4.0;
    let mut lods = vec![Lod::new(TOP_RESOLUTION, 0).expect("invalid config")];
    for i in 1..16 {
        lods.push(
            Lod::new(lods[(i - 1) as usize].resolution() / 2.0, i).expect("invalid tile schema"),
        );
    }

    let tile_schema = TileSchema {
        origin: ORIGIN,
        bounds: Rect::new(
            -20037508.342787,
            -20037508.342787,
            20037508.342787,
            20037508.342787,
        ),
        lods: lods.into_iter().collect(),
        tile_width: 1024,
        tile_height: 1024,
        y_direction: VerticalDirection::TopToBottom,
        crs: Crs::EPSG3857,
    };
    let layer = RasterTileLayerBuilder::new_rest(move |&index: &TileIndex| {
        MAPTILER_SAT_URL
            .replace("{z}", &index.z.to_string())
            .replace("{x}", &index.x.to_string())
            .replace("{y}", &index.y.to_string())
            + &format!("?key={}", api_key)
    })
    .with_file_cache_checked(".satellite_cache")
    .with_tile_schema(tile_schema)
    .build()
    .expect("failed to create layer");
    let layer = Arc::new(RwLock::new(layer));
    layer
}

fn get_zoom() -> u8 {
    unimplemented!()
}

fn create_map() -> Map {
    let sat_layer = download_sat_tile();
    let sat_layer_clone = sat_layer.clone();

    /*    let raster_layer = RasterTileLayerBuilder::new_osm()
            .with_file_cache_checked(".tile_cache")
            .build()
            .expect("failed to create layer");
    */
    MapBuilder::default()
        .with_latlon(39.05514242773056, -108.52035286223932)
        .with_resolution(150.0)
        .with_layer(sat_layer)
        .build()
}
