use galileo::galileo_types::cartesian::{CartesianPoint2d, Point2, Point3, Rect};
use galileo::galileo_types::geo::{GeoPoint, NewGeoPoint, Projection};
use galileo::layer::feature_layer::Feature;
use serde::{Deserialize, Deserializer, Serialize};

use galileo::Color;
use galileo::layer::feature_layer::symbol::{SimplePolygonSymbol, Symbol};
use galileo::render::render_bundle::RenderBundle;

use galileo::galileo_types::geometry::{CartesianGeometry2d, Geom, Geometry};
use galileo::galileo_types::impls::{MultiPolygon, Polygon};
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
pub enum SegmentLabel {
    Building,
    Road,
    Water,
    SwimmingPool,
    Vegetation,
    TennisCourt,
    Other,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Segment {
    pub color: Option<Color>,
    pub bbox: Rect,
    pub segment_type: Option<SegmentLabel>,
    pub confidence: Option<f32>,
    pub mask: Option<SegmentMask>,
    pub is_hidden: bool,
}

//}
/*
fn des_geometry<'de, D: Deserializer<'de>>(d: D) -> Result<MultiPolygon<Point2>, D::Error> {
    Ok(Vec::<Polygon<Point2>>::deserialize(d)?.into())
}
*/
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SegmentMask {}

impl SegmentMask {
    fn get_box_symbol(&self, segment: &Segment) -> SimplePolygonSymbol {
        let color = segment.color.unwrap_or(Color::WHITE);
        let bbox = segment.bbox;
        SimplePolygonSymbol::new(color)
    }
}

impl Symbol<Segment> for SegmentMask {
    fn render(
        &self,
        feature: &Segment,
        geometry: &Geom<Point3>,
        min_resolution: f64,
        bundle: &mut RenderBundle,
    ) {
        if !feature.is_hidden {
            self.get_box_symbol(feature)
                .render(&(), geometry, min_resolution, bundle);
        }
    }
}
