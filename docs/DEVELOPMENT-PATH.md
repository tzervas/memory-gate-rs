# Development path — memory-gate-rs

**Back layer.** How this project got where it is, reconstructed from git history, merged PRs, releases, and in-repo ADRs/debt notes.  
Where a claim is inference rather than a cited artifact, it is marked **inferred from …**.

Companion docs: [CURRENT-STATE.md](./CURRENT-STATE.md) (measured today), [ROADMAP.md](./ROADMAP.md) (forward work).

---

## One-paragraph origin

`memory-gate-rs` is the Rust port of a dynamic **memory learning layer** for AI agents: learn from interactions, store experiences, retrieve relevant context, and consolidate (prune) low-value or stale memories. The design explicitly cites **Complementary Learning Systems (CLS)** — a fast ingest path plus slower consolidation — in crate-level docs (`src/lib.rs`). Initial implementation and v1.0.0 tagging landed in a single calendar day (2026-01-25), then the project was pulled into the broader tzervas **fleet / tero / mint / W2** integration work through mid-2026.

---

## Timeline (evidence-backed)

### 2026-01-25 — Birth and “1.0.0” cut

| Commit / PR | What happened |
|-------------|----------------|
| `41b31a2` | `feat: initial memory-gate-rs implementation` |
| `2e30479` | `feat: add CI/CD, storage backends, and examples` |
| Dependabot PRs #1–#8 | Early action/crate bumps (artifacts, fastembed, rusqlite, criterion, metrics-exporter) |
| `ba20bfb` | `chore: release v1.0.0` |

**Inferred from CHANGELOG + tree:** the 1.0.0 narrative bundled gateway, traits, in-memory / Qdrant / sqlite-vec stores, VSA modules, adapters, agents, and metrics as the “core memory system.” Whether all of that was production-hardened at tag time is **not** independently re-proven here; see CURRENT-STATE for what still fails closed only under optional tests.

**Decision signal:** ship a full-featured Rust port early under a **1.x version label**, with GitHub Releases (later `v1.0.1`) rather than waiting for crates.io. That choice collides with the later fleet rule “0.x.x until a human says otherwise” (open PR #53 exists to renumber).

### Early–mid 2026 — Quiet on product commits (from `main` log density)

After the January burst, `main` history is dominated by Dependabot until the mint/tero wave. **Inferred from `git log`:** dependency maintenance kept the crate compiling while product work concentrated elsewhere (workspace orch / cabal / tero) before landing back here.

### 2026-07-09 — Mint M1 domains + W2 facade scoping (PR #26 and follow-ons)

| Ref | What happened |
|-----|----------------|
| PR **#26** (`feature/mint-m1-domain-facade` → `dev`, then main land) | M1 **AgentDomain** extensions: `Workspace`, `Tero`, `Context`, `MemoryGate`, `LangRust`, `LangPython`; prefix-aware parse (`layer:tero`, `lang:rust`, `repo:…`) |
| `8b6de82` / `50ec84f` / `48899ad` | wsfull wave state, tero reindex, AGENTS/README kickoff docs |
| PR **#27** | Hygiene: `scripts/check.sh`, stub `docs/ROADMAP.md` |

**Decision — domain as the integration key:** rather than a heavy new MCP binary inside this crate, M1 treats **domain filters on existing retrieve/learn** as the thin facade for tero-first + context + gate scoping (“no bloat”). Shared structured response schemas were **escalated to wsfull orch** (documented in AGENTS.md / README M1 notes), not fully owned here.

**Alternative rejected (inferred from README/AGENTS prose + absence of MCP binary):** embedding a full MCP server or duplicating tero’s citation index inside memory-gate. Tero stays Layer-1 cited corpus; gate stays persistent learned memory; context-mcp stays session/RAG.

### 2026-07-15–16 — Wave B embeddings + Wave C golden + integration facade

| PR | Decision / delivery |
|----|---------------------|
| **#34** | Multi-model embedding catalog (`SupportedEmbeddingModel`), batch encode, query embed LRU cache, **fail-closed** store/model/dimension binding, sqlite-vec Criterion benches |
| **#35** | Wave C golden corpus + `eval` recall@k helpers; full golden test `#[ignore]` (model download) |
| **#36** | Stop tracking tero-rs correction design **in this repo** (ownership boundary) |
| **#37** | `facade` module: `join/mg-facade@STABLE` — metadata conventions, `for_tero_learn`, `learn`/`retrieve`/`consolidate_once`, prod path hints |

**Decision — fail closed on model mismatch (PR #34):** reopening a Qdrant collection or SQLite DB with the wrong embedding model/dimension errors instead of silently returning garbage similarities. Rationale is safety of long-lived vector stores; alternative (best-effort re-embed or ignore meta) was rejected in favor of stamped metadata + validation (CHANGELOG Unreleased / Wave B notes).

**Decision — golden accuracy is RS-only (Wave C):** Python `memory-gate` frozen as a mirror for catalog parity; acceptance criteria live in `docs/WAVE_C_ACCEPTANCE.md`. Full golden stays opt-in (`--ignored`) so default CI does not download models.

**Decision — facade is thin wrappers, not a second gateway (PR #37):** consumers (especially tero-rs) get a documented public surface over `MemoryGateway` instead of scraping private modules. Contract name: `join/mg-facade@STABLE`.

### 2026-07-16–21 — Fleet CI, licensing, debt, gate hardening

| PR | Theme |
|----|--------|
| **#30–#31** | Commitizen semver config + manual Commitizen workflow |
| **#32** | Dependabot security advisories |
| **#38–#39** | Self-hosted podman runners; re-enable push/PR triggers (after a dispatch-only period — `de9271e`) |
| **#40** | DEBT-001: document intentional `trit-vsa` **0.3.0** pin for experimental `vsa-accel` |
| **#41** | REUSE bootstrap + SPDX annotations |
| **#43–#46** | Fleet standards pack (badges, issue-close-on-main, workflow_dispatch meta) |
| **#51** | Harden fleet gates for self-hosted catch-up (`b9881f3`, current `main` tip at PM measurement) |

**Decision — `vsa-accel` stays experimental and out of `full` (DEBT-001):** keep crates.io pin `trit-vsa = "0.3.0"` for fleet alignment rather than path-depending sibling 0.2.0. Primary VSA path remains pure Rust modules; accel is optional until pins unify.

**Decision — fleet workflows as the trunk signal:** `fleet-ci.yml` / `fleet-security.yml` badges on README; issue close policy is main-only (`docs/FLEET_STANDARDS.md`). A legacy `CI` workflow still exists and was measured **failing on beta** at `b9881f3` (see CURRENT-STATE).

### Releases

| Tag / Release | Date | Notes |
|---------------|------|--------|
| `v1.0.0` | 2026-01-25 (tag era) | Initial release commit `ba20bfb` |
| `v1.0.1` GitHub Release | 2026-07-10 | “Semver Baseline (M1 + W2 CommonMemory)” |

Neither tag implies crates.io publication (verified absent 2026-07-25).

---

## Architectural decisions that shaped the code

### 1. CLS dual-stream memory (founding)

**Choice:** immediate store path + background consolidation worker (`MemoryGateway`, `GatewayConfig`).  
**Why (from crate docs):** avoid stability–plasticity collapse — learn fast without unbounded retention of noise.  
**Evidence:** `src/lib.rs` module docs; consolidation tests; `start_consolidation` / `run_consolidation_once`.

### 2. Pluggable `KnowledgeStore` + `MemoryAdapter` traits

**Choice:** storage and knowledge transform are traits; default `InMemoryStore` + `PassthroughAdapter`.  
**Why:** test without embeddings; swap Qdrant/sqlite-vec for production-shaped deploys.  
**Trade-off:** in-memory retrieval is **not** semantic — demos can look “smart” only when content substrings match. Vector quality lives behind optional features + model download.

### 3. Domain enum as multi-tenant / multi-layer scope (M1)

**Choice:** extend `AgentDomain` rather than introduce a free-form namespace type for mint integration.  
**Why:** single filter argument on retrieve/learn already existed; prefixes give stringly config without new protocol.  
**Trade-off:** enum growth for every new layer/lang; parse rules must stay documented.

### 4. Embedding catalog + one model per store file/collection (Wave B)

**Choice:** shared stable IDs with Python (`all-minilm-l6-v2`, `bge-small-en-v1.5` default, `bge-base-en-v1.5`); fail closed on mismatch.  
**Why:** two 384-d models are not interchangeable in a single collection.  
**Evidence:** PR #34, CHANGELOG, README embedding section, sqlite-vec meta tests (non-ignored).

### 5. Integration facade vs design docs ownership (PR #36–#37)

**Choice:** keep a small stable facade here; move tero-rs correction design **out** of this repo.  
**Why:** prevent dual sources of truth across the tero/memory-gate boundary.

### 6. Versioning tension (ongoing)

**Choice (historical):** label releases 1.0.0 / 1.0.1.  
**Counter-pressure (fleet):** 0.x until human authorization of 1.x; PR #53 proposes `0.2.0`.  
**Status at PM measurement:** tree still `1.0.0` on `main`; compliance PR open, not merged.

---

## What was deliberately *not* built here (yet)

From Wave C acceptance non-goals, CHANGELOG, and missing code paths:

- MCP binary for memory-gate (called out as WS-14 / out of Wave C scope in `docs/WAVE_C_ACCEPTANCE.md`)
- Owning tero L1 indexing (sibling tero stack + `docs/tero-index` is a **lite corpus index of this repo’s docs**, not the runtime memory store)
- Python package evolution (frozen mirror for embed catalog parity)
- Guaranteed warm-retrieve latency SLOs (aspirational table in Wave C doc only)

---

## How to verify this narrative

```bash
git log --oneline | head -80
gh pr list --state merged --limit 40
gh api /repos/tzervas/memory-gate-rs/releases
git show ba20bfb --stat   # v1.0.0 release commit
git show 4fb2c40          # PR #26 merge
```

For measured “does it work today?”, prefer [CURRENT-STATE.md](./CURRENT-STATE.md) over this archaeology.
