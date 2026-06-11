//! Edge cost budgets: operator-declared watts and sampled peak memory.
//!
//! The two cost terms of [`crate::scoring::edge_score`] that the harness
//! cannot derive from the model itself — power (no portable sensor exists,
//! so the operator declares it) and memory (RSS-sampled on Linux, supplied
//! by the operator elsewhere or when inference runs out-of-process).

use serde::{Deserialize, Serialize};

/// Operator-declared power budget for one run.
///
/// There is no portable power sensor, so watts are **supplied by the
/// operator** from the device's known package power, e.g. Raspberry Pi 5
/// ~8 W, Apple M4 Pro package ~30 W. The label names the hardware/config so
/// Pareto points from different devices stay distinguishable.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgePowerProfile {
    /// Device/configuration label, e.g. `pi5`, `m4-pro`.
    pub label: String,
    /// Declared package power in watts (finite, strictly positive).
    pub watts: f64,
}

impl EdgePowerProfile {
    /// Validated profile: non-empty label, finite positive watts.
    pub fn new(label: impl Into<String>, watts: f64) -> anyhow::Result<Self> {
        let label = label.into();
        if label.trim().is_empty() {
            anyhow::bail!("power profile label must be non-empty");
        }
        if !watts.is_finite() || watts <= 0.0 {
            anyhow::bail!("watts must be a finite positive measurement, got {watts}");
        }
        Ok(Self { label, watts })
    }

    /// Raspberry Pi 5 proxy budget (~8 W package).
    pub fn pi5_proxy() -> Self {
        Self { label: "pi5-proxy".to_string(), watts: 8.0 }
    }

    /// Apple M4 Pro proxy budget (~30 W package).
    pub fn m4_pro_proxy() -> Self {
        Self { label: "m4-pro-proxy".to_string(), watts: 30.0 }
    }
}

/// Peak-RSS sampler for **this process**.
///
/// On Linux it reads `VmRSS` from `/proc/self/status`. On other platforms
/// (or on parse failure) sampling returns `None` and the report must carry
/// the operator-supplied
/// [`crate::edge::EdgeLoopBenchmark::with_memory_gb_override`] — also the
/// right choice when the model runs in a separate server process (vLLM),
/// whose footprint this sampler cannot see.
///
/// Peak tracking: the benchmark samples before the first task and after
/// every task, keeping the maximum.
#[derive(Debug, Default)]
pub struct MemorySampler {
    peak_bytes: Option<u64>,
}

impl MemorySampler {
    pub fn new() -> Self {
        Self::default()
    }

    /// Current RSS of this process in bytes; `None` off-Linux or on failure.
    pub fn rss_bytes() -> Option<u64> {
        #[cfg(target_os = "linux")]
        {
            parse_vmrss_bytes(&std::fs::read_to_string("/proc/self/status").ok()?)
        }
        #[cfg(not(target_os = "linux"))]
        {
            None
        }
    }

    /// Take one sample, fold it into the running peak, and return it.
    pub fn sample(&mut self) -> Option<u64> {
        let rss = Self::rss_bytes()?;
        self.peak_bytes = Some(self.peak_bytes.map_or(rss, |p| p.max(rss)));
        Some(rss)
    }

    /// Highest RSS observed so far, in bytes.
    pub fn peak_bytes(&self) -> Option<u64> {
        self.peak_bytes
    }

    /// Highest RSS observed so far, in GB (GiB, 1024^3 bytes).
    pub fn peak_gb(&self) -> Option<f64> {
        self.peak_bytes.map(|b| b as f64 / (1024.0 * 1024.0 * 1024.0))
    }
}

/// Parse the `VmRSS:` line (kB) of `/proc/self/status` into bytes.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
fn parse_vmrss_bytes(status: &str) -> Option<u64> {
    let line = status.lines().find(|l| l.starts_with("VmRSS:"))?;
    let kb: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kb * 1024)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn power_profile_validation_and_proxies() {
        assert!(EdgePowerProfile::new("pi5", 8.0).is_ok());
        assert!(EdgePowerProfile::new("  ", 8.0).is_err());
        assert!(EdgePowerProfile::new("pi5", 0.0).is_err());
        assert!(EdgePowerProfile::new("pi5", -1.0).is_err());
        assert!(EdgePowerProfile::new("pi5", f64::NAN).is_err());
        assert_eq!(EdgePowerProfile::pi5_proxy().watts, 8.0);
        assert_eq!(EdgePowerProfile::m4_pro_proxy().watts, 30.0);
    }

    #[test]
    fn vmrss_parsing() {
        let status = "Name:\tedge\nVmPeak:\t  9999 kB\nVmRSS:\t  2048 kB\nThreads:\t1\n";
        assert_eq!(parse_vmrss_bytes(status), Some(2048 * 1024));
        assert_eq!(parse_vmrss_bytes("Name:\tedge\n"), None);
        assert_eq!(parse_vmrss_bytes("VmRSS:\tgarbage kB\n"), None);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn sampler_returns_some_on_linux_and_tracks_peak() {
        let mut sampler = MemorySampler::new();
        assert!(sampler.peak_bytes().is_none());
        let rss = sampler.sample().expect("VmRSS must sample on linux");
        assert!(rss > 0);
        let peak = sampler.peak_bytes().unwrap();
        assert!(peak >= rss);
        sampler.sample().unwrap();
        assert!(sampler.peak_bytes().unwrap() >= peak, "peak never decreases");
        assert!(sampler.peak_gb().unwrap() > 0.0);
    }
}
