# ADR-282 research gate

This directory is the trusted, dependency-free implementation of the
pre-pull-request gate. Candidate code is evaluated in a separate contained
process; these scripts validate its data and never import candidate modules.

```bash
python3 scripts/research-gate/research_gate.py validate-manifest \
  research-manifest.json --expect-sha "$CANDIDATE_SHA"
python3 scripts/research-gate/research_gate.py evaluate \
  research-manifest.json raw-results.json --output evaluation.json
python3 -m unittest discover -s scripts/research-gate/tests -v
```

The evaluator requires paired confirmation runs over the exact preregistered
seed list. It enforces one primary resource budget, full memory accounting,
deterministic selection counts, real-data/production-topology declarations,
and a canonical ADR-281 embedding-space identity.

The candidate workflow has read-only repository permission and no secrets.
Generated code runs in a networkless, resource-limited, unprivileged
container. A separate trusted job creates the GitHub artifact attestation,
and a separate default-branch promotion workflow owns the pull-request token.
The attestation job is dependency-gated on both preflight and candidate jobs;
it records their actual `needs.<job>.result` values. The candidate job can
only succeed after its sequential scoped CI, methodology validator, raw-result
consistency check, and confirmation evaluator all succeed, so the attested
contained-gate outcome is mechanically tied to those steps rather than a
candidate assertion.

Trusted nightly orchestration should call `.github/workflows/research-candidate.yml`
through `workflow_call` with an immutable branch head SHA. Manual execution
uses the identical `workflow_dispatch` input contract. A candidate-controlled
workflow must never be the caller.

Configure the `research-gate-override` protected Environment with required
reviewers from `@ruvnet/research-gate`, enable “Prevent self-review,” and set
the repository variable `RESEARCH_GATE_CODEOWNERS` to the comma-separated
GitHub logins of that team’s current members. The override workflow reads the
GitHub environment review history, records the actual approving reviewer,
requires that reviewer to have `maintain` or `admin`, and checks the reviewer
against this mirrored CODEOWNERS membership list. Comments and labels are not
accepted as overrides.

The exception is deliberately narrow. A trusted preflight queries check runs
and commit statuses for the exact current `main` SHA. A separately attested
override may convert only those `base/...` failures to `authorized-red`; it
cannot alter candidate containment, scoped CI, methodology, confirmation,
artifact, or attestation outcomes. The override expires within 72 hours and
is revalidated during promotion against the exact base SHA, candidate SHA,
failure set, and branch head.

The nightly (03:17 UTC) and manually runnable, default-branch
`research-nightly-dispatch` workflow discovers immutable heads under
`research/candidate/**` and `research/nightly/**`, de-duplicates them by
ref/SHA, resolves the current base SHA, and dispatches the named
default-branch `research-candidate` workflow. Candidate branches never supply
the dispatch workflow definition or its Actions token. The named run can
then trigger `research-promote`. If the base is red,
the first run fails closed, a reviewer creates a protected override, and a
trusted operator reruns `research-candidate` with that override workflow run
ID.

GitHub artifact retention is a delivery cache, not the durable evidence
store. Production deployment must copy indexed artifacts to versioned
write-once/object-lock storage for the schema's 365-day, 2555-day, or
permanent retention class.
