//! PhotonLayer WASM bindings (ADR-260 §7.3). Stub — filled in by the swarm.

#![allow(dead_code)]

use wasm_bindgen::prelude::*;

/// Crate version, exported to verify the WASM module loads.
#[wasm_bindgen]
pub fn photonlayer_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

#[cfg(test)]
mod tests {
    #[test]
    fn smoke() {
        assert!(!super::photonlayer_version().is_empty());
    }
}
