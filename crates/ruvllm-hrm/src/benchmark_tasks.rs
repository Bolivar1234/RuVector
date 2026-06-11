//! Built-in offline task sets for the ADR-199 acceptance harness.
//!
//! Small but real: 20 tasks per category (the ADR specifies 100 controlled
//! tasks per category for the production gate; these built-ins keep the
//! harness runnable offline and the report shape identical).

use crate::benchmark::{AcceptanceTask, Expected, ReasoningTask, TaskCategory};
use crate::router::Route;

/// `(input, expected JSON)` — Extract-mode inputs in the `fields:` key=value
/// form the mock backend can also solve deterministically.
const JSON_CASES: &[(&str, &str)] = &[
    ("fields: name=Alice; age=30", r#"{"name":"Alice","age":30}"#),
    ("fields: city=Oslo; population=709037", r#"{"city":"Oslo","population":709037}"#),
    ("fields: sku=AB-12; qty=4; in_stock=true", r#"{"sku":"AB-12","qty":4,"in_stock":true}"#),
    ("fields: sensor=th-7; celsius=21.5", r#"{"sensor":"th-7","celsius":21.5}"#),
    ("fields: user=dana; logins=17", r#"{"user":"dana","logins":17}"#),
    ("fields: host=node-3; up=false", r#"{"host":"node-3","up":false}"#),
    ("fields: order_id=1042; total=99.95", r#"{"order_id":1042,"total":99.95}"#),
    ("fields: lang=rust; year=2015", r#"{"lang":"rust","year":2015}"#),
    ("fields: id=7; label=alpha; weight=0.5", r#"{"id":7,"label":"alpha","weight":0.5}"#),
    ("fields: track=R1; cycles=16", r#"{"track":"R1","cycles":16}"#),
    ("fields: model=HRM-Text-1B; params=1000000000", r#"{"model":"HRM-Text-1B","params":1000000000}"#),
    ("fields: bucket=logs; objects=88123", r#"{"bucket":"logs","objects":88123}"#),
    ("fields: region=us-central1; zones=3", r#"{"region":"us-central1","zones":3}"#),
    ("fields: branch=main; ahead=2; behind=0", r#"{"branch":"main","ahead":2,"behind":0}"#),
    ("fields: device=pi5; watts=7.2", r#"{"device":"pi5","watts":7.2}"#),
    ("fields: queue=ingest; depth=512", r#"{"queue":"ingest","depth":512}"#),
    ("fields: test=acceptance; passed=true", r#"{"test":"acceptance","passed":true}"#),
    ("fields: kv=two-tier; bits=3", r#"{"kv":"two-tier","bits":3}"#),
    ("fields: dataset=gsm8k; subset=200", r#"{"dataset":"gsm8k","subset":200}"#),
    ("fields: phase=A; gate=parity", r#"{"phase":"A","gate":"parity"}"#),
];

/// Plan-mode goals; usefulness judged by the harness heuristic (offline) or
/// a stronger model (online).
const PLAN_CASES: &[&str] = &[
    "Migrate the user database from MySQL to Postgres without downtime",
    "Add JWT authentication to the public REST API",
    "Reduce p99 latency of the search endpoint below 100ms",
    "Set up nightly backups for the vector store",
    "Roll out the new tokenizer to all edge devices",
    "Write integration tests for the payments service",
    "Upgrade the cluster to the next Kubernetes minor version",
    "Index 10 million documents into the HNSW store",
    "Profile and fix the memory leak in the ingest worker",
    "Move the CI pipeline from Jenkins to GitHub Actions",
    "Introduce feature flags for the recommendation engine",
    "Shard the metrics database by tenant",
    "Implement rate limiting on the inference gateway",
    "Localize the dashboard UI into three languages",
    "Replace polling with webhooks in the sync service",
    "Cut the Docker image size of the model server in half",
    "Add canary deployments to the release workflow",
    "Encrypt all backups at rest with customer-managed keys",
    "Build a load-test harness for the chat endpoint",
    "Document the public SDK and publish versioned docs",
];

/// `(claim, evidence, expected_pass)` contradiction pairs.
const CONTRADICTION_CASES: &[(&str, &str, bool)] = &[
    ("The server is online.", "The server is not online.", false),
    ("The build is passing.", "The build is not passing.", false),
    ("The disk is not full.", "The disk is full.", false),
    ("The deploy is finished.", "The deploy is not finished.", false),
    ("The token was not revoked.", "The token was revoked.", false),
    ("The test suite is green.", "The test suite is not green.", false),
    ("The sky is green.", "The sky is blue.", false),
    ("The cache is empty.", "The cache is warm.", false),
    ("The region is us-east1.", "The region is us-central1.", false),
    ("The answer is 41.", "The answer is 42.", false),
    ("The quota is exhausted.", "(none)", false),
    ("The license is GPL.", "(none)", false),
    ("The server is online.", "The server is online.", true),
    ("The disk is full.", "The disk is full.", true),
    ("The answer is 42.", "The answer is 42.", true),
    ("The region is us-central1.", "The region is us-central1.", true),
    ("Paris is the capital of France.", "France is in Europe.", true),
    ("The cache holds 100 entries.", "The cache holds 100 entries.", true),
    ("Rust ships a borrow checker.", "Rust ships a borrow checker.", true),
    ("The model has 1B parameters.", "The model has 1B parameters.", true),
];

/// `(intent, canonical route wire-form)` routing labels. The wire form must
/// parse via [`Route::from_wire`].
const ROUTING_CASES: &[(&str, &str)] = &[
    ("Calculate the sum of 19 and 23", "tool:calculator"),
    ("Compute 144 divided by 12", "tool:calculator"),
    ("What is the average of 3, 5 and 9?", "tool:calculator"),
    ("Multiply 7 by 8 for me", "tool:calculator"),
    ("What's the weather in Oslo tomorrow?", "tool:weather"),
    ("Get the forecast for Berlin this weekend", "tool:weather"),
    ("Search the knowledge base for similar incidents", "vector_search"),
    ("Find similar documents to this contract", "vector_search"),
    ("Retrieve the nearest neighbors for this embedding", "vector_search"),
    ("Look up related memories about caching strategies", "vector_search"),
    ("Deploy the new release to staging", "workflow:ops"),
    ("Run the nightly backup job now", "workflow:ops"),
    ("Trigger the CI workflow for branch main", "workflow:ops"),
    ("Start the data ingestion pipeline", "workflow:ops"),
    ("Design the architecture for a multi-region database", "remote:frontier"),
    ("Provide a legal analysis of this license clause", "remote:frontier"),
    ("Write a formal proof of this theorem", "remote:frontier"),
    ("Summarize this paragraph in two sentences", "local_model:ruvltra"),
    ("Explain what a hash map is", "local_model:ruvltra"),
    ("Draft a short status update for the team", "local_model:ruvltra"),
];

/// Short Direct-mode prompts for the latency category.
const LATENCY_CASES: &[&str] = &[
    "What color is the clear daytime sky?",
    "Name the capital of France.",
    "How many days are in a week?",
    "What is the chemical symbol for water?",
    "Name a primary color.",
    "How many legs does a spider have?",
    "What planet do humans live on?",
    "What is the opposite of hot?",
    "How many minutes are in an hour?",
    "Name the largest ocean.",
    "What language is this crate written in?",
    "How many sides does a triangle have?",
    "What is the first month of the year?",
    "Name a noble gas.",
    "What is the plural of mouse?",
    "How many continents are there?",
    "What is the freezing point of water in Celsius?",
    "Name the star at the center of our solar system.",
    "What is two squared?",
    "What direction does the sun rise from?",
];

/// `(problem, expected integer)` reasoning tasks for the R1 cycle sweep and
/// the R5 best-of-N experiment (SynthCot mode, greedy).
const REASONING_CASES: &[(&str, i64)] = &[
    ("What is 17 + 25?", 42),
    ("What is 6 times 7?", 42),
    ("What is 100 - 58?", 42),
    ("What is 13 + 29?", 42),
    ("What is 240 + 16?", 256),
    ("What is 12 times 12?", 144),
    ("What is 90 - 48?", 42),
    ("What is 31 + 64?", 95),
    ("What is 8 times 9?", 72),
    ("What is 500 - 123?", 377),
    ("What is 44 + 56?", 100),
    ("What is 15 times 4?", 60),
    ("What is 81 - 19?", 62),
    ("What is 7 plus 35?", 42),
    ("What is 11 times 11?", 121),
    ("What is 200 - 75?", 125),
    ("What is 64 + 64?", 128),
    ("What is 25 times 3?", 75),
    ("What is 1000 - 488?", 512),
    ("What is 21 + 21?", 42),
];

pub(crate) fn json_extraction_tasks() -> Vec<AcceptanceTask> {
    JSON_CASES
        .iter()
        .map(|(input, expected)| AcceptanceTask {
            category: TaskCategory::JsonExtraction,
            input: (*input).to_string(),
            evidence: Vec::new(),
            expected: Expected::Json(
                serde_json::from_str(expected).expect("built-in expected JSON is valid"),
            ),
        })
        .collect()
}

pub(crate) fn plan_tasks() -> Vec<AcceptanceTask> {
    PLAN_CASES
        .iter()
        .map(|input| AcceptanceTask {
            category: TaskCategory::PlanUsefulness,
            input: (*input).to_string(),
            evidence: Vec::new(),
            expected: Expected::PlanHeuristic,
        })
        .collect()
}

pub(crate) fn contradiction_tasks() -> Vec<AcceptanceTask> {
    CONTRADICTION_CASES
        .iter()
        .map(|(claim, evidence, pass)| AcceptanceTask {
            category: TaskCategory::ContradictionDetection,
            input: (*claim).to_string(),
            evidence: vec![(*evidence).to_string()],
            expected: Expected::Verdict { pass: *pass },
        })
        .collect()
}

pub(crate) fn routing_tasks() -> Vec<AcceptanceTask> {
    ROUTING_CASES
        .iter()
        .map(|(intent, wire)| AcceptanceTask {
            category: TaskCategory::RoutingAccuracy,
            input: (*intent).to_string(),
            evidence: Vec::new(),
            expected: Expected::Route(
                Route::from_wire(wire).expect("built-in route labels are canonical"),
            ),
        })
        .collect()
}

pub(crate) fn latency_tasks() -> Vec<AcceptanceTask> {
    LATENCY_CASES
        .iter()
        .map(|input| AcceptanceTask {
            category: TaskCategory::Latency,
            input: (*input).to_string(),
            evidence: Vec::new(),
            expected: Expected::None,
        })
        .collect()
}

pub(crate) fn reasoning_tasks() -> Vec<ReasoningTask> {
    REASONING_CASES
        .iter()
        .map(|(problem, expected)| ReasoningTask {
            problem: (*problem).to_string(),
            expected: *expected,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn task_sets_meet_minimum_sizes() {
        assert!(json_extraction_tasks().len() >= 20);
        assert!(plan_tasks().len() >= 20);
        assert!(contradiction_tasks().len() >= 20);
        assert!(routing_tasks().len() >= 20);
        assert!(latency_tasks().len() >= 20);
        assert!(reasoning_tasks().len() >= 20);
    }

    #[test]
    fn routing_labels_match_keyword_table() {
        // Labels must agree with the shared deterministic keyword router,
        // otherwise the offline benchmark gates the wrong thing.
        for (intent, wire) in ROUTING_CASES {
            assert_eq!(
                crate::router::keyword_route(intent).as_wire(),
                *wire,
                "keyword table drifted for intent: {intent}"
            );
        }
    }
}
