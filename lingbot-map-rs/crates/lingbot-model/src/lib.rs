//! # lingbot-model
//!
//! The reconstruction core of the LingBot-Map Rust port: turn a streaming RGB
//! frame (plus retrieved anchor context) into a per-frame **keyframe feature
//! vector** and a **3D point cloud**.
//!
//! Two backends share one [`Reconstructor`] interface:
//!
//! * [`SyntheticReconstructor`] — pure-Rust, deterministic, dependency-light.
//!   Always available; builds for `wasm32`. It derives depth from monocular
//!   shading/parallax cues and projects to 3D. It is *not* the trained model —
//!   it is the synthetic fallback (ADR-0006) that lets the full streaming
//!   pipeline, video/image export, and demos run end-to-end without the
//!   4.63 GB checkpoint.
//! * `transformer::GeometricContextTransformer` (behind the `candle` feature) —
//!   the candle implementation that loads the real safetensors weights.
//!
//! Anchor context (the top-K structurally similar past keyframes retrieved from
//! `lingbot-memory`) is folded into reconstruction as a drift-correction term,
//! mirroring how the Python original uses anchor context inside attention.

use lingbot_tensor::ModelConfig;

mod synthetic;
pub use synthetic::SyntheticReconstructor;

#[cfg(feature = "candle")]
pub mod transformer;

/// An RGBA8 frame.
#[derive(Debug, Clone)]
pub struct Frame {
    /// Width in pixels.
    pub width: usize,
    /// Height in pixels.
    pub height: usize,
    /// Row-major RGBA8 pixels, length `width * height * 4`.
    pub pixels: Vec<u8>,
}

impl Frame {
    /// Construct, validating the buffer length.
    pub fn new(width: usize, height: usize, pixels: Vec<u8>) -> Self {
        assert_eq!(pixels.len(), width * height * 4, "RGBA buffer size mismatch");
        Self {
            width,
            height,
            pixels,
        }
    }

    /// A solid-color frame (handy for tests / placeholders).
    pub fn solid(width: usize, height: usize, rgba: [u8; 4]) -> Self {
        let mut pixels = Vec::with_capacity(width * height * 4);
        for _ in 0..width * height {
            pixels.extend_from_slice(&rgba);
        }
        Self::new(width, height, pixels)
    }

    /// Luminance (0..1) at pixel (x, y). Rec. 601 weights.
    #[inline]
    pub fn luma(&self, x: usize, y: usize) -> f32 {
        let i = (y * self.width + x) * 4;
        let r = self.pixels[i] as f32;
        let g = self.pixels[i + 1] as f32;
        let b = self.pixels[i + 2] as f32;
        (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
    }

    /// RGB at pixel (x, y).
    #[inline]
    pub fn rgb(&self, x: usize, y: usize) -> [u8; 3] {
        let i = (y * self.width + x) * 4;
        [self.pixels[i], self.pixels[i + 1], self.pixels[i + 2]]
    }
}

/// A 3D point with color.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point {
    /// World-space position.
    pub pos: [f32; 3],
    /// RGB color.
    pub color: [u8; 3],
}

/// A reconstructed 3D point cloud for one frame.
#[derive(Debug, Clone, Default)]
pub struct PointCloud {
    /// The points.
    pub points: Vec<Point>,
}

impl PointCloud {
    /// Number of points.
    pub fn len(&self) -> usize {
        self.points.len()
    }
    /// Whether empty.
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Axis-aligned bounds `(min, max)` over all points, or `None` if empty.
    pub fn bounds(&self) -> Option<([f32; 3], [f32; 3])> {
        let first = self.points.first()?;
        let mut min = first.pos;
        let mut max = first.pos;
        for p in &self.points {
            for a in 0..3 {
                min[a] = min[a].min(p.pos[a]);
                max[a] = max[a].max(p.pos[a]);
            }
        }
        Some((min, max))
    }
}

/// A keyframe feature vector exported to the memory layer.
#[derive(Debug, Clone, PartialEq)]
pub struct KeyframeFeatures(pub Vec<f32>);

impl KeyframeFeatures {
    /// Borrow as a slice (for `lingbot-memory`).
    pub fn as_slice(&self) -> &[f32] {
        &self.0
    }
    /// Dimensionality.
    pub fn dim(&self) -> usize {
        self.0.len()
    }
}

/// Per-frame reconstruction interface shared by all backends.
pub trait Reconstructor {
    /// The configuration in use.
    fn config(&self) -> &ModelConfig;

    /// Process one streaming frame.
    ///
    /// `anchors` are the feature vectors of the top-K structurally similar past
    /// keyframes retrieved from `lingbot-memory` (empty for the first frames).
    /// They are used for drift correction. Returns this frame's keyframe
    /// features (to be inserted into memory) and its 3D point cloud.
    fn process_frame(
        &mut self,
        frame_index: u64,
        frame: &Frame,
        anchors: &[KeyframeFeatures],
    ) -> (KeyframeFeatures, PointCloud);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_luma_and_rgb() {
        let f = Frame::solid(2, 2, [255, 0, 0, 255]);
        assert!((f.luma(0, 0) - 0.299).abs() < 1e-3);
        assert_eq!(f.rgb(1, 1), [255, 0, 0]);
    }

    #[test]
    fn pointcloud_bounds() {
        let pc = PointCloud {
            points: vec![
                Point {
                    pos: [-1.0, 0.0, 2.0],
                    color: [0, 0, 0],
                },
                Point {
                    pos: [3.0, -2.0, 1.0],
                    color: [0, 0, 0],
                },
            ],
        };
        let (min, max) = pc.bounds().unwrap();
        assert_eq!(min, [-1.0, -2.0, 1.0]);
        assert_eq!(max, [3.0, 0.0, 2.0]);
    }
}
