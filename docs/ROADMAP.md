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

### Wave C — Golden recall + accuracy gate (in progress)

- Fixed corpus + queries (`mg/golden-recall`); mean recall@5 must not drop **>2%** vs baseline in fixture/docs.
- Pure metric unit tests in default CI; full FastEmbed integration via `cargo test --test golden_recall -- --ignored`.
- Local merge gate: `./scripts/check.sh`.
- **Canonical RS only** — Python frozen after embed parity.
- Acceptance detail: [WAVE_C_ACCEPTANCE.md](WAVE_C_ACCEPTANCE.md).

**Status:** Wave C delivery (golden tests/fixtures owned by L1-GOLDEN sibling); docs acceptance criteria on `feat/wave-c-golden-recall`.

### Post–Wave C — Tero + MCP integration (planned)

- **tero-rs**: cited L1 index; optional adapter to memory-gate-rs learn/retrieve.
- **tero-mcp**: tool surface; companion or separate memory MCP (WS-14) TBD.
- Scoping via existing M1 domains (`Tero`, `MemoryGate`, `Context`, lang/repo prefixes).
- Design only until next wave: [TERO_INTEGRATION.md](TERO_INTEGRATION.md). No `tero-rs` / `tero-mcp` crate dependency in Wave C PR.

---

## Links

- AGENTS.md (Tero rule, local checks, PR flow)
- CONTRIBUTING.md (cargo fmt/clippy/test, doc requirements)
- CHANGELOG.md (M1 entries)
- [WAVE_C_ACCEPTANCE.md](WAVE_C_ACCEPTANCE.md), [TERO_INTEGRATION.md](TERO_INTEGRATION.md)
- tero index: `docs/tero-index/index.json`
