//! PhotonLayer experiment memory on RuVector (ADR-260 §5, §11–§15).
//!
//! NOTE: stub — filled in by the implementation swarm. See ADR-260.

#![allow(dead_code)]

/// Placeholder so the crate compiles as a workspace member before the
/// experiment-memory implementation lands.
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(test)]
mod tests {
    #[test]
    fn smoke() {
        assert!(!super::version().is_empty());
    }
}
