
# AGENTS.md — memory-gate-rs

**Use Tero + cabal-devmelopner for work here.**

**Kickoff framework**: root `.claude/kickoffs/mint.md` for integration. Local .claude/kickoffs/.

## Tero (Layer-1 corpus index)

Repo has `docs/tero-index/index.json` (generated/ refreshed via tero-mcp/scripts/generate_lite_index.py).

**Rule:** Use tero queries before large greps or assumptions.
- Grok: tero__text_search / query_by_id (token "local-dev")
- Direct: tero-mcp-lite --index docs/tero-index/index.json
- cabal-devmelopner: auto-detects local index when run from within this tree (or set TERO_INDEX_PATH).

Example:
```bash
cd /root/git/memory-gate-rs
# agent with context:
uv run --project ../cabal-devmelopner cabal-devmelopner "task description here" --use-tero
```

Citations point at file:line — open them.

## Working with cabal-devmelopner agent tool

This project is prepared for integration:
- Tero index committed on chore/tero-index-cabal-ready (and PRable to dev)
- Local auto index support in cabal
- This AGENTS.md

**PR flow (protect main/dev):**
- Create/checkout feature or chore branch
- Make changes (agent will often use working branch)
- PR the branch → `dev` (then dev → main when ready)

## Local checks

Look for:
- scripts/check.sh
- Cargo.toml / pyproject.toml + standard commands (cargo test, uv run pytest, ruff, etc.)

Run checks before considering work complete.

## Further reading

- README.md
- docs/ROADMAP.md or ROADMAP.md (if present)
- docs/ASSESSMENT.md or similar for intent/gaps
- ../cabal-devmelopner/docs/* for agent architecture
- ../tero-mcp for how indexes are built and served

Leave mycelium isolated; all coordination here targets the other repos + cabal.

## Latest (M1 + W2 Facade, 2026-07-09 PR #26)
M1 domains + facade executed on feature/mint-m1-domain-facade. AgentDomain extended (Workspace, Tero, Context, MemoryGate, LangRust, LangPython + prefix-aware FromStr for "layer:tero", "lang:rust", "repo:xxx" unified facade scoping). Thin facade via domain filters on retrieve/ingest (tero-first cited + gate persistent scoped, no bloat). W2 integration: domain as scoping key for shared StructuredResponse schemas (escalated to wsfull). Tero reindex + doc/AGENTS/kickoff updates. PR #26. See dev-workflow, guards (branch/worktree), dev-docs/waves/wsfull-wave-2026-07-09-compact.md (compacted state), WORKSPACE_CABAL_TERO_READINESS.md.
Docs + tero-index in PR. Run pr-review (adapted: M1 domains/tero/W2/C0/dev-workflow/guards). Merge if good.
Update docs + tero in PR process per task. Compact state captured.

## Post #26 merge + propagate (2026-07-09)
- Merged to dev @4fb2c40 (PR#26 feature/mint-m1-domain-facade). pr-review (adapted M1/tero/W2) + gh verification comment posted.
- Dev land to main (incl #26 M1 facade + W2 refs); sync pulls.
- Tero cites + append-only followed. Mirror: M1 AgentDomain/facade parallels cabal CommonMemory W2 facade (domain scoping key).
- Refs: dev-docs/waves/wsfull-wave-2026-07-09-compact.md (wsfull state, C0/M1/W2/local-ollama), WORKSPACE_CABAL_TERO_READINESS.md (leaf orch tranche, parameterization W2), cabal-devmelopner PRs + kickoffs.
- Next: use /root/git/scripts/update-tero.sh ; branch-guard for any further (chore/feature -> dev PRs only).

