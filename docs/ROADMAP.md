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

## Links

- AGENTS.md (Tero rule, local checks, PR flow)
- CONTRIBUTING.md (cargo fmt/clippy/test, doc requirements)
- CHANGELOG.md (M1 entries)
- tero index: `docs/tero-index/index.json`
