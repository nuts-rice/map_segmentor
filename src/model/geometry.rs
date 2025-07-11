use galileo::galileo_types::cartesian::{Point2, Rect};
use galileo::galileo_types::geo::{GeoPoint, NewGeoPoint, Projection};
use galileo::layer::feature_layer::Feature;
use serde::{Deserialize, Deserializer, Serialize};

use galileo::Color;
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
    #[serde(deserialize_with = "des_geometry")]
    pub geometry: MultiPolygon<Point2>,
    pub color: Option<Color>,
    pub bbox: Rect,
    pub segment_type: Option<SegmentLabel>,
    pub confidence: Option<f32>,
    pub mask: Option<SegmentMask>,
}

impl Feature for Segment {
    type Geom = Self;
    fn geometry(&self) -> &Self::Geom {
        self
    }
}

impl Geometry for Segment {
    type Point = Point2;

    fn project<P: Projection<InPoint = Self::Point> + ?Sized>(
        &self,
        projection: &P,
    ) -> Option<Geom<P::OutPoint>> {
        self.geometry.project(projection)
    }
}

fn des_geometry<'de, D: Deserializer<'de>>(d: D) -> Result<MultiPolygon<Point2>, D::Error> {
    Ok(Vec::<Polygon<Point2>>::deserialize(d)?.into())
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SegmentMask {}
