use crate::BbCoords;

use galileo::layer::feature_layer::{FeatureLayer, FeatureLayerOptions};
use galileo::layer::raster_tile_layer::RasterTileLayerBuilder;
use galileo::symbol::ImagePointSymbol;

use galileo::{Map, MapBuilder};
struct ColoredMask {
    width: u32,
    height: u32,
}

pub fn run() {
    galileo_egui::init(create_map(), []).expect("failed to initialize");
}

fn create_map() -> Map {
    let raster_layer = RasterTileLayerBuilder::new_osm()
        .with_file_cache_checked(".tile_cache")
        .build()
        .expect("failed to create layer");
    MapBuilder::default()
        .with_latlon(39.069067, -108.536016)
        .with_resolution(50.0)
        .with_layer(raster_layer)
        .build()
}
