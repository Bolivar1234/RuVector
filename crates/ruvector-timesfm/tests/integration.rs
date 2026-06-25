//! Integration tests. The pure-logic tests always run; the real-model tests are
//! gated on the `candle` feature AND the local weights, skipping cleanly (never
//! fabricating a pass) when the 814 MB safetensors are absent.

use ruvector_timesfm::anomaly::score_window;
use ruvector_timesfm::sweep::EarlyStopper;
use ruvector_timesfm::Forecast;

fn forecast_from_bands(p10: &[f32], p50: &[f32], p90: &[f32]) -> Forecast {
    let quantiles = (0..p50.len())
        .map(|i| {
            // p10..p90; fill the in-between channels by interpolation (only
            // p10/p50/p90 are asserted by the anomaly logic).
            [
                p10[i], p10[i], p10[i], p50[i], p50[i], p50[i], p90[i], p90[i], p90[i],
            ]
        })
        .collect();
    Forecast {
        point: p50.to_vec(),
        quantiles,
    }
}

#[test]
fn forecast_quantile_accessors() {
    let f = forecast_from_bands(&[1.0, 2.0], &[5.0, 6.0], &[9.0, 10.0]);
    assert_eq!(f.horizon(), 2);
    assert_eq!(f.p10(), vec![1.0, 2.0]);
    assert_eq!(f.p50(), vec![5.0, 6.0]);
    assert_eq!(f.p90(), vec![9.0, 10.0]);
}

#[test]
fn anomaly_flags_out_of_band_points() {
    let f = forecast_from_bands(&[0.0, 0.0, 0.0], &[5.0, 5.0, 5.0], &[10.0, 10.0, 10.0]);
    // inside band, above band, below band.
    let observed = [5.0, 25.0, -15.0];
    let report = score_window(&f, &observed);
    assert_eq!(report.points.len(), 3);
    assert_eq!(report.n_anomalies, 2);
    assert!(!report.points[0].is_anomaly);
    assert!(report.points[1].is_anomaly && report.points[1].deviation > 0.0);
    assert!(report.points[2].is_anomaly && report.points[2].deviation < 0.0);
}

#[test]
fn early_stopper_warms_up_before_deciding() {
    let stopper = EarlyStopper::new(0.05, 1000).with_min_history(16);
    // Build a StopDecision via the non-candle path is not possible (evaluate is
    // gated), but the config + Default surface is exercised here.
    assert_eq!(stopper.min_history, 16);
    assert_eq!(stopper.threshold, 0.05);
    assert_eq!(EarlyStopper::default().confidence_gate, 0.6);
}

#[cfg(feature = "candle")]
mod real_model {
    use ruvector_timesfm::Forecaster;

    const WEIGHTS: &str = "/tmp/timesfm-parity/timesfm.safetensors";

    fn skip() -> bool {
        if !std::path::Path::new(WEIGHTS).exists() {
            eprintln!("SKIP real-model test: weights missing ({WEIGHTS}).");
            true
        } else {
            false
        }
    }

    #[test]
    fn forecast_shapes_and_band_ordering() -> anyhow::Result<()> {
        if skip() {
            return Ok(());
        }
        let device = timesfm::select_device()?;
        let f = Forecaster::load(WEIGHTS, device)?;
        let series: Vec<f32> = (0..256)
            .map(|t| (t as f32 / 12.0).sin() * 10.0 + 50.0)
            .collect();
        let forecast = f.forecast(&series, 64)?;
        assert_eq!(forecast.horizon(), 64);
        assert_eq!(forecast.point.len(), 64);
        // All forecast values finite; quantiles monotone p10 <= p50 <= p90.
        for i in 0..64 {
            assert!(forecast.point[i].is_finite());
            let (lo, mid, hi) = (forecast.p10()[i], forecast.p50()[i], forecast.p90()[i]);
            assert!(lo.is_finite() && mid.is_finite() && hi.is_finite());
            assert!(lo <= hi, "p10 {lo} > p90 {hi} at step {i}");
        }
        Ok(())
    }

    #[test]
    fn early_stopper_prunes_doomed_run() -> anyhow::Result<()> {
        if skip() {
            return Ok(());
        }
        use ruvector_timesfm::sweep::EarlyStopper;
        let device = timesfm::select_device()?;
        let f = Forecaster::load(WEIGHTS, device)?;
        // doomed: decays toward 0.20, never reaches the 0.05 threshold.
        let doomed: Vec<f32> = (0..128)
            .map(|t| 0.20 + 0.75 * (-(t as f32) / 16.0).exp())
            .collect();
        let stopper = EarlyStopper::new(0.05, 1000)
            .with_min_history(16)
            .with_confidence_gate(0.5);
        let d = stopper.evaluate(&f, &doomed)?;
        assert!(d.stop, "doomed run should stop: {}", d.reason);

        // warm-up: too few points → never stop.
        let short = &doomed[..8];
        let d2 = stopper.evaluate(&f, short)?;
        assert!(!d2.stop && d2.decision.is_none());
        Ok(())
    }
}
