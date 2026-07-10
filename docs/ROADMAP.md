# memory-gate-rs — Product Roadmap

**Status:** Living (2026-07-09)  
**North star:** Thin, tero-first persistent memory facade for agents with domain-scoped W2 integration.

M1 domains + facade complete (see below). Companion: [AGENTS.md](AGENTS.md), [CONTRIBUTING.md](CONTRIBUTING.md), [README.md](README.md).

---

## Phases

### M1 — Domain + Facade Design (done, 2026-07-09 PR #26)

- Extended `AgentDomain` (Workspace, Tero, Context, MemoryGate, LangRust, LangPython) + prefix-aware FromStr ("layer:tero", "lang:rust", "repo:xxx").
- Thin facade: domain filters on `retrieve_context` / ingest for tero-first citations + gate persistent scoped memories (no bloat).
- W2: domain as scoping key for shared StructuredResponse schemas (escalated to wsfull).
- Tero-first queries, cited + read, append-only docs/CHANGELOG.
- Merged dev -> main. See AGENTS.md (Latest + Post #26), README (M1 section), CHANGELOG.

**Status:** M1 complete.

### M2 — (pending)

- TBD per cabal / wsfull waves and mint kickoffs.
- Potential: Lang docs dual-index (M3 note in README), full adapter evolution, VSA/accel stabilization.
- Hygiene + tero updates as part of each tranche.

### Hygiene, Tero, Facade Evolution

- Add `scripts/check.sh` (fmt --check/fix, clippy -D warnings, doc -D warnings, build, test --all-features) per AGENTS local checks expectation.
- `docs/ROADMAP.md` (this file) for further reading.
- Use `/root/git/scripts/update-tero.sh` + tero reindex on changes.
- Branch-guard: chore/feature branches, PRs (to dev per AGENTS; protected main).
- Facade evolution: keep thin, mirror cabal CommonMemory W2, domain scoping.
- Run checks before complete: `./scripts/check.sh` (or `--fix`).

See dev-workflow in workspace docs, guards, wsfull-wave-2026-07-09-compact.md, WORKSPACE_CABAL_TERO_READINESS.md.

---

## W2 Facade Evolution (chore/w2-rollout-docs-wiring)

Append-only extension per plan.md w2-rollout (parallel safe to cabal MVP).

- Facade evolution: keep thin; evolve toward CommonMemory mirror (trait stub if fits; see dev-docs/schemas/common_memory_facade.rs.example).
- Simple CommonMemory mirror stub (Rust sketch, for future facade.rs or in types/docs; mirrors cabal CommonMemoryAdapter + Py):
  ```rust
  // stub (not full impl yet; domain-scoped via existing AgentDomain)
  use crate::types::AgentDomain;
  // use super::structured::StructuredResponse; // when added
  pub trait CommonMemory {
      fn query(&self, domain: AgentDomain, q: &str, opts: Option<serde_json::Value>)
          -> /* StructuredResponse */ String;  // placeholder: returns cited resp
      fn store(&self, domain: AgentDomain, content: &str, meta: Option<serde_json::Value>) -> String;
      fn supported_domains(&self) -> Vec<AgentDomain>;
  }
  // Wire via domain filters on retrieve_context (M1) + tero L1; see cabal facade for contract.
  ```
- Cross: cabal-devmelopner (schemas.py AgentDomain/CommonMemoryAdapter) is Py mirror of M1 here (types.rs). Context-mcp session as W2 consumer. Dev-mcp servers/ docs updated w/ cabal facade ex.
- Tero cites: plan.md:44 (W2 rollout), wsfull-wave-2026-07-09-compact.md, memory types.rs M1 (AgentDomain Tero/Context etc), dev-docs/schemas/.

Next M2: full CommonMemory impl + store wiring.

## W2 code wiring (chore/w2-code-wiring-facade appended)
- CommonMemory trait (W2 mirror) implemented in src/traits.rs (pub, reexported in lib + prelude).
- Compiles (cargo check clean); thin sync contract matching cabal Py (query/store/supported_domains + AgentDomain).
- Integrates M1 domains; placeholder for StructuredResponse (per dev-docs/schemas + cabal schemas.py).
- Tero-first, hygiene (will run check.sh), append-only docs, branch-guard.
- Cross-cites: plan.md w2-rollout (code beyond docs), memory AGENTS/ROADMAP prior stub, cabal facade, wsfull compact.
- Status: code wiring started (trait surface); consumers (gateway/adapters) can impl; full later.
- Verify: cargo test/check; tero hits post regen; cabal can evolve consume.

## Semver Baseline (chore/semver-baseline-v1.0.1)
- Cargo version -> 1.0.1 for baseline after W2 trait work (CommonMemory) + M1.
- Tero-first enforced (script `/root/git/scripts/tero.sh memory-gate-rs text_search version .` + W2 queries).
- Notes appended to CHANGELOG/AGENTS/ROADMAP citing plan.md w2-rollout + M1, local GHCR (ghcr.io/tzervas/memory-gate-rs:1.0.1).
- Hygiene + local build; disjoint worktree optional; tag v1.0.1 + gh release.
- Per AGENTS guards, append-only, signed intent.

---

## Links

- AGENTS.md (Tero rule, local checks, PR flow)
- CONTRIBUTING.md (cargo fmt/clippy/test, doc requirements)
- CHANGELOG.md (M1 + 1.0.1 baseline entries)
- tero index: `docs/tero-index/index.json`
