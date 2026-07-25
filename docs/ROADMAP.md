# memory-gate-rs — Roadmap

**Status:** Living (updated 2026-07-25 from measured state + open GitHub signals)  
**North star:** Thin, tero-first persistent memory facade for agents — domain-scoped learn/retrieve that stays honest about what is verified vs experimental.

Companions: [CURRENT-STATE.md](./CURRENT-STATE.md) · [DEVELOPMENT-PATH.md](./DEVELOPMENT-PATH.md) · [WAVE_C_ACCEPTANCE.md](./WAVE_C_ACCEPTANCE.md) · [DEBT.md](./DEBT.md) · [AGENTS.md](../AGENTS.md)

No delivery dates below unless justified by an external schedule. Items are ordered roughly by how blocked or policy-relevant they are, not by vanity priority.

---

## Done (do not re-plan)

| Item | Evidence |
|------|----------|
| Core gateway + in-memory path | Tests + `cargo run --example basic_usage` (see CURRENT-STATE) |
| M1 domain extensions + prefix parse | PR #26, `AgentDomain` tests |
| Wave B multi-model catalog + fail-closed binding | PR #34 |
| Wave C golden harness + pure recall metrics | PR #35; fixture + `#[ignore]` integration test |
| Integration facade `join/mg-facade` | PR #37 |
| Local gate script `./scripts/check.sh` | PR #27 |
| DEBT-001 documented | PR #40 / `docs/DEBT.md` |
| Fleet CI/security pack + badges | PRs #38–#46, #51 |

---

## Near-term — unblocked by process / honesty fixes

### R1. Semver 0.x compliance

- **What:** Align package version with fleet policy (0.x until a human authorizes 1.x). Tree is `1.0.0` with tags `v1.0.0` / `v1.0.1` while crates.io has **no** package.
- **Why it matters:** 1.x signals maturity and registry presence that do not exist; confuses consumers and violates BRANCH-AND-RELEASE-CONTRACT §3.
- **Unblocks on:** Merge (or supersede) open PR **#53** (`chore/semver-0x-compliance`) after human review; **do not** invent a new 1.x cut.
- **Note:** This docs PR does **not** change versions (docs-only).

### R2. Registry / docs badges honesty

- **What:** Either publish to crates.io + docs.rs **or** stop implying publication (README badges currently point at non-existent crate — measured 404 / “does not exist”).
- **Why it matters:** Stale badges are believed (contract §4 / §5a).
- **Unblocks on:** Human decision to publish under 0.x **or** a docs/badge change PR; cargo publish credentials and a deliberate release.

### R3. Legacy `CI` workflow beta red on `main`

- **What:** At `b9881f3`, workflow **CI** fails on job `Test (beta)` step **Build** while fleet-ci succeeds.
- **Why it matters:** Mixed signals train people to ignore red; beta matrix may be unmaintained.
- **Unblocks on:** Investigation of beta toolchain failure (source/CI change — not docs); decide whether legacy `ci.yml` is retired in favor of fleet-ci only (open PR **#52** may relate).
- **Evidence:** https://github.com/tzervas/memory-gate-rs/actions/runs/29865297829

### R4. REUSE residual path coverage

- **What:** Issue **#42** / `REUSE-DEBT.md` — not all paths covered after P24f bootstrap.
- **Why it matters:** License compliance completeness for fleet.
- **Unblocks on:** Incremental `REUSE.toml` / SPDX work (code/license files — separate non-docs PR).

---

## Product / quality — real capability gaps

### R5. Run and gate Wave C golden on demand (still not default CI)

- **What:** `cargo test --features sqlite-vec --test golden_recall -- --ignored` with model download; keep default CI free of network/model cost.
- **Why it matters:** Semantic regression protection exists as harness but was **UNVERIFIED** in the PM measurement run.
- **Unblocks on:** Operator time + disk/network for FastEmbed weights; optional self-hosted job with cache (CI design choice).
- **Acceptance already written:** [WAVE_C_ACCEPTANCE.md](./WAVE_C_ACCEPTANCE.md).

### R6. sqlite-vec ignored suite (embed path)

- **What:** ~11 sqlite-vec tests marked `requires embedding model download` stay ignored in normal `cargo test --all-features`.
- **Why it matters:** Meta/fail-closed tests pass without models; store/retrieve/domain paths do not get continuous signal.
- **Unblocks on:** Shared model cache in CI or documented operator pre-merge checklist enforced by humans/agents.

### R7. Live Qdrant verification

- **What:** Five Qdrant tests are `#[ignore]`; no Criterion Qdrant bench in-tree (README already states this).
- **Why it matters:** “Production vector DB” path is compile-verified only in default gates.
- **Unblocks on:** Qdrant instance (`QDRANT_URL`), enabling ignored tests, optional bench target.

### R8. Metrics feature behavioral tests

- **What:** `metrics` compiles; scrape/export path not exercised in measured suite.
- **Why it matters:** Observability claims without a test are easy to break silently.
- **Unblocks on:** Small unit/integration test with the Prometheus exporter (source change).

### R9. Facade + tero L1 end-to-end test

- **What:** `for_tero_learn` / metadata helpers exist; no test proved tero index → learn → domain-scoped retrieve in this measurement.
- **Why it matters:** Mint “tero-first” story is architectural; without a test it can drift.
- **Unblocks on:** Fixture-based test using static metadata (no live MCP required) or a marked integration test with local `docs/tero-index`.

### R10. `vsa-accel` / DEBT-001 closure

- **What:** Optional `trit-vsa` 0.3.0 pin; experimental; not in `full`.
- **Why it matters:** Accel feature is advertised as experimental with fleet pin conflict vs sibling 0.2.0.
- **Unblocks on:** Criteria in [DEBT.md](./DEBT.md) — unified trit-vsa line, tests/benches green, README install without caveat.

---

## Integration / workspace (proposed, not committed)

These appear in AGENTS.md / mint history as direction. Treat as **proposed** unless a PR lands.

| Item | Why | Would unblock |
|------|-----|----------------|
| Lang docs dual-index (M3 in older notes) | LangRust/LangPython domains exist as labels | tero dual-index work + ingestion policy outside this crate |
| MCP binary (WS-14 class) | Agent-native transport | Product decision + scope; Wave C explicitly non-goal |
| CommonMemory / W2 shared schemas | Cross-repo facade with cabal | wsfull orch ownership; this crate supplies domain key + gate API |
| Qdrant Criterion bench | Parity with sqlite-vec benches | Dev time + local Qdrant |
| Aspirational warm retrieve p95 / batch ingest ×N | Perf story | Measurement protocol + non-ignored benches; do not ship as guarantees |

---

## Explicit non-goals (still)

- Inventing release dates or “% complete” scores.
- Proposing a **1.x** crates.io cut from an agent (forbidden by fleet contract).
- Treating Python `memory-gate` as the accuracy owner (Wave C: RS-only).
- Documenting unpublished bench numbers as SLOs.

---

## How to add an item

1. Prefer a real signal: open issue, failing CI, `#[ignore]` gap, DEBT entry, or measured UNVERIFIED.
2. Write **what / why / unblocks on**.
3. Update [CURRENT-STATE.md](./CURRENT-STATE.md) when the item becomes VERIFIED.
