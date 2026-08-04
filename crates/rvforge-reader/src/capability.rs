//! The install-time capability contract (requirements P6, ADR-286).
//!
//! A [`CapabilityCard`] is derived from the signed `CapabilityManifest`
//! described in `docs/research/rvf-forge/registry-model.md`. Two rules govern
//! the derivation and both are enforced here rather than in the UI:
//!
//! - **Default deny.** Every one of the fifteen ADR-286 capability classes that
//!   the manifest does not request is rendered in the "cannot" list. A manifest
//!   that cannot be read produces a card where everything is denied.
//! - **No vague prose.** Broad scopes (`all-files`, `*`, `unrestricted`) are
//!   rejected at derivation time, as are banned phrases such as "access your
//!   computer". The card cannot render a permission it could not describe
//!   specifically.

use serde::{Deserialize, Serialize};

/// The fifteen ADR-286 capability classes.
pub const CAPABILITY_CLASSES: [&str; 15] = [
    "memory",
    "filesystem",
    "network",
    "model",
    "mcp",
    "process",
    "clock",
    "randomness",
    "gpu",
    "sensor",
    "display",
    "audio",
    "clipboard",
    "persistent-state",
    "inter-agent-messaging",
];

/// Scope tokens that describe too much to render honestly. ADR-294 §8 routes
/// these to manual review; requirements P6 forbids rendering them as prose.
const VAGUE_SCOPES: [&str; 12] = [
    "all",
    "all-files",
    "any",
    "anything",
    "arbitrary",
    "broad",
    "everything",
    "full",
    "full-access",
    "system",
    "unrestricted",
    "your-computer",
];

/// Phrases that must never reach the user, whatever the manifest says.
const BANNED_PHRASES: [&str; 6] = [
    "access your computer",
    "access to your computer",
    "control your computer",
    "full access",
    "unrestricted access",
    "all your files",
];

#[derive(Debug, Clone, Deserialize)]
pub struct CapabilityManifest {
    #[serde(rename = "schemaVersion")]
    pub schema_version: u32,
    #[serde(rename = "manifestId")]
    pub manifest_id: String,
    #[serde(rename = "defaultPolicy")]
    pub default_policy: String,
    #[serde(default)]
    pub requests: Vec<CapabilityRequest>,
    #[serde(default)]
    pub denials: Vec<String>,
    #[serde(default, rename = "manualReviewTriggers")]
    pub manual_review_triggers: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CapabilityRequest {
    pub class: String,
    pub scope: String,
    #[serde(default)]
    pub rationale: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum CardStatus {
    /// Derived from a manifest that parsed and passed the P6 rules.
    Derived,
    /// No manifest could be read. Nothing is granted.
    ManifestUnavailable,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CardLine {
    pub class: String,
    pub scope: Option<String>,
    /// The exact sentence shown to the user.
    pub text: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CapabilityCard {
    pub status: CardStatus,
    pub manifest_id: Option<String>,
    pub default_policy: String,
    /// "This agent requests" — one line per granted capability.
    pub requests: Vec<CardLine>,
    /// "This agent cannot" — explicit denials plus every unrequested class.
    pub cannot: Vec<CardLine>,
    pub manual_review_triggers: Vec<String>,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CapabilityError {
    /// ADR-286 requires a closed baseline.
    DefaultPolicyNotDeny(String),
    UnknownClass(String),
    VagueScope {
        class: String,
        scope: String,
    },
    BannedPhrase {
        class: String,
        phrase: String,
    },
    /// A class appears in both `requests` and `denials`.
    Contradiction(String),
    Parse(String),
}

impl std::fmt::Display for CapabilityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CapabilityError::DefaultPolicyNotDeny(p) => {
                write!(f, "defaultPolicy is '{p}', expected 'deny'")
            }
            CapabilityError::UnknownClass(c) => {
                write!(f, "'{c}' is not an ADR-286 capability class")
            }
            CapabilityError::VagueScope { class, scope } => write!(
                f,
                "scope '{scope}' on class '{class}' is too broad to describe specifically"
            ),
            CapabilityError::BannedPhrase { class, phrase } => {
                write!(
                    f,
                    "class '{class}' rationale contains banned phrase '{phrase}'"
                )
            }
            CapabilityError::Contradiction(c) => {
                write!(f, "class '{c}' is both requested and denied")
            }
            CapabilityError::Parse(e) => write!(f, "capability manifest parse error: {e}"),
        }
    }
}

impl std::error::Error for CapabilityError {}

impl CapabilityCard {
    /// The card shown when no manifest is available: nothing granted,
    /// every class denied.
    pub fn manifest_unavailable(reason: &str) -> Self {
        Self {
            status: CardStatus::ManifestUnavailable,
            manifest_id: None,
            default_policy: "deny".to_string(),
            requests: Vec::new(),
            cannot: CAPABILITY_CLASSES
                .iter()
                .map(|c| CardLine {
                    class: (*c).to_string(),
                    scope: None,
                    text: denial_text(c),
                })
                .collect(),
            manual_review_triggers: Vec::new(),
            notes: vec![
                reason.to_string(),
                "No capability is granted while the manifest is unreadable (default deny)."
                    .to_string(),
            ],
        }
    }

    /// Derive a card from manifest JSON.
    pub fn from_manifest_json(json: &str) -> Result<Self, CapabilityError> {
        let manifest: CapabilityManifest =
            serde_json::from_str(json).map_err(|e| CapabilityError::Parse(e.to_string()))?;
        Self::from_manifest(&manifest)
    }

    /// Derive a card from a parsed manifest, enforcing the P6 rules.
    pub fn from_manifest(manifest: &CapabilityManifest) -> Result<Self, CapabilityError> {
        if manifest.default_policy != "deny" {
            return Err(CapabilityError::DefaultPolicyNotDeny(
                manifest.default_policy.clone(),
            ));
        }

        let mut requests = Vec::new();
        let mut requested_classes: Vec<String> = Vec::new();

        for req in &manifest.requests {
            let class = req.class.to_ascii_lowercase();
            if !CAPABILITY_CLASSES.contains(&class.as_str()) {
                return Err(CapabilityError::UnknownClass(req.class.clone()));
            }
            check_scope(&class, &req.scope)?;
            if let Some(rationale) = &req.rationale {
                check_phrases(&class, rationale)?;
            }
            let text = request_text(&class, &req.scope);
            check_phrases(&class, &text)?;

            requested_classes.push(class.clone());
            requests.push(CardLine {
                class,
                scope: Some(req.scope.clone()),
                text,
            });
        }

        let mut cannot = Vec::new();
        let mut seen_denials: Vec<String> = Vec::new();

        for denial in &manifest.denials {
            let token = denial.to_ascii_lowercase();
            if requested_classes.contains(&token) {
                return Err(CapabilityError::Contradiction(token));
            }
            if seen_denials.contains(&token) {
                continue;
            }
            seen_denials.push(token.clone());
            cannot.push(CardLine {
                class: token.clone(),
                scope: None,
                text: denial_text(&token),
            });
        }

        // Default deny: everything not requested is denied, whether or not the
        // publisher bothered to list it.
        for class in CAPABILITY_CLASSES {
            if requested_classes.iter().any(|c| c == class)
                || seen_denials.iter().any(|d| d == class)
            {
                continue;
            }
            cannot.push(CardLine {
                class: class.to_string(),
                scope: None,
                text: denial_text(class),
            });
        }

        Ok(Self {
            status: CardStatus::Derived,
            manifest_id: Some(manifest.manifest_id.clone()),
            default_policy: manifest.default_policy.clone(),
            requests,
            cannot,
            manual_review_triggers: manifest.manual_review_triggers.clone(),
            notes: Vec::new(),
        })
    }
}

fn check_scope(class: &str, scope: &str) -> Result<(), CapabilityError> {
    let normalized = scope.trim().to_ascii_lowercase();
    let vague = normalized.is_empty()
        || normalized.contains('*')
        || VAGUE_SCOPES.contains(&normalized.as_str())
        || VAGUE_SCOPES.iter().any(|v| {
            normalized.starts_with(&format!("{v} ")) || normalized.starts_with(&format!("{v}-"))
        });
    if vague {
        return Err(CapabilityError::VagueScope {
            class: class.to_string(),
            scope: scope.to_string(),
        });
    }
    Ok(())
}

fn check_phrases(class: &str, text: &str) -> Result<(), CapabilityError> {
    let lowered = text.to_ascii_lowercase();
    for phrase in BANNED_PHRASES {
        if lowered.contains(phrase) {
            return Err(CapabilityError::BannedPhrase {
                class: class.to_string(),
                phrase: phrase.to_string(),
            });
        }
    }
    Ok(())
}

/// Per-class, per-scope sentence. Every branch names the specific class and
/// the specific scope; there is no generic fallback that could read as
/// "access your computer".
fn request_text(class: &str, scope: &str) -> String {
    match (class, scope) {
        ("filesystem", "user-selected") => {
            "Read the files and folders you select, and nothing else".to_string()
        }
        ("persistent-state", "encrypted-local") => {
            "Keep encrypted memory on this computer between sessions".to_string()
        }
        ("model", "embedded") | ("model", "local") => {
            "Run its bundled model on this computer".to_string()
        }
        ("memory", s) => format!("Use up to {s} of memory"),
        ("filesystem", s) => format!("Read files under: {s}"),
        ("network", s) => format!("Open network connections to: {s}"),
        ("model", s) => format!("Run a model located at: {s}"),
        ("mcp", s) => format!("Call MCP tools limited to: {s}"),
        ("process", s) => format!("Start processes limited to: {s}"),
        ("clock", s) => format!("Read the runtime clock: {s}"),
        ("randomness", s) => format!("Draw randomness from: {s}"),
        ("gpu", s) => format!("Use the GPU for: {s}"),
        ("sensor", s) => format!("Read sensor data from: {s}"),
        ("display", s) => format!("Draw to the display: {s}"),
        ("audio", s) => format!("Use audio limited to: {s}"),
        ("clipboard", s) => format!("Use the clipboard limited to: {s}"),
        ("persistent-state", s) => format!("Store persistent state as: {s}"),
        ("inter-agent-messaging", s) => format!("Exchange messages with: {s}"),
        (c, s) => format!("Use the '{c}' capability limited to: {s}"),
    }
}

/// Denial sentences. Tokens outside the fifteen classes appear in real
/// manifests (`microphone`, `background`), so they get specific phrasing too.
fn denial_text(token: &str) -> String {
    match token {
        "network" => "Access the internet".to_string(),
        "filesystem" => "Read folders you have not selected".to_string(),
        "process" => "Start other programs".to_string(),
        "background" => "Run in the background".to_string(),
        "external-model-providers" => "Contact external model providers".to_string(),
        "microphone" => "Use the microphone".to_string(),
        "camera" => "Use the camera".to_string(),
        "audio" => "Record or play audio".to_string(),
        "clipboard" => "Read or write your clipboard".to_string(),
        "gpu" => "Use the GPU".to_string(),
        "sensor" => "Read sensors on this computer".to_string(),
        "display" => "Draw outside its own window".to_string(),
        "mcp" => "Call MCP tools".to_string(),
        "model" => "Run a model".to_string(),
        "memory" => "Allocate memory beyond its declared quota".to_string(),
        "clock" => "Read the system clock".to_string(),
        "randomness" => "Draw randomness outside the runtime".to_string(),
        "persistent-state" => "Keep memory between sessions".to_string(),
        "inter-agent-messaging" => "Exchange messages with other agents".to_string(),
        other => format!("Use the '{other}' capability"),
    }
}
