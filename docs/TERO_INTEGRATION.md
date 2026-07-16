# Tero integration — follow-on design (post–Wave C)

**Status:** **Future work** — not shipped in Wave C.  
**Purpose:** Document how persistent **memory-gate-rs** memory plugs into **tero** (L1 cited corpus) and **tero-mcp** (tool surface) so integrators (cabal, agents, MCP hosts) do not invent APIs in chat.

Wave C delivers the Rust canonical runtime, golden recall acceptance, and local gates. **This document is the handoff contract for the next wave** only.

---

## Hard prerequisite before any tero ↔ memory-gate integration

**Mycelium is a separate language project.** It must not ride along as an accidental monorepo inside `tero-rs`.

Before implementing memory-gate-rs integration (or expanding tero-mcp consumers):

1. **Inventory** every `mycelium-*` path dependency / workspace member in `tero-rs`.
2. **Decide per crate:** required for tero’s public indexing/API surface, or extraction residue.
3. **Purge wholesale** anything not strictly necessary — prefer **published crates / thin optional deps** over vendoring the Mycelium core language stack.
4. **History hygiene:** if mycelium cores were never needed, they should be **removed from the tero-rs tree** (and ideally from ongoing history via a deliberate cleanup PR; do not “capture all mycelium core crates” as the integration path).
5. Capability reuse (e.g. VSA ideas) must be **intentional adapters**, not wholesale crate embedding.

Workspace note (2026-07 inventory snapshot under `/root/git/workspace/tero-rs`): prior strip removed 40 non-dep mycelium crates, but **~16 `mycelium-*` crates remain vendored** as path deps (`mycelium-doc`, `mycelium-vsa`, and transitive language cores). Re-validate necessity before integration work; default posture is **rip out unless proven required**.

---

## Component roles

| Layer | Repo / surface | Responsibility |
|-------|----------------|----------------|
| **Tero L1** | `tero-rs` (library), `docs/tero-index/` in each repo | Cited, structured **corpus index** — search returns answer + resolvable citations (`file:line`), not long-term learned agent memory |
| **Memory gate** | `memory-gate-rs` | **Persistent learned memories** — `learn_from_interaction`, `retrieve_context`, consolidation, vector backends (`sqlite-vec`, `qdrant`) |
| **MCP** | `tero-mcp` (and future thin memory MCP if WS-14 ships) | **Tool surface** for hosts — tero query tools today; memory store/retrieve tools are a separate optional binary |

**Canonical runtime after parity:** memory-gate-rs only for new memory features; Python `memory-gate` frozen except critical fixes.

---

## Scoping with existing domains (M1)

M1 already extended `AgentDomain` for unified facade filtering. Use these for integration sketches — **no new enum variants required for the follow-on wave**:

| Domain | Use when |
|--------|----------|
| `AgentDomain::Tero` | Memories derived from **tero L1** hits (cited snippets, summaries ingested after a tero search) |
| `AgentDomain::MemoryGate` | Operational knowledge about the memory layer itself (policies, consolidation notes, adapter behavior) |
| `AgentDomain::Context` | Session/RAG items from context-mcp (parallel path to tero-first) |
| `AgentDomain::LangRust` / `LangPython` | Lang-doc scoped learn/retrieve (dual-index evolution per roadmap) |

Prefix-aware `FromStr` (`layer:tero`, `lang:rust`, `repo:…`) remains the scoping key for `retrieve_context(..., Some(domain))`.

---

## Suggested flows

### Flow 1 — Tero-first search → optional learn

1. Agent calls **tero-mcp** (or `tero-rs`) with a task query; receives **StructuredResponse**-style payload: short answer + citations.
2. Agent (or cabal facade) optionally calls `MemoryGateway::learn_from_interaction` with:
   - `content`: distilled fact + citation pointers (not full repo dump),
   - `domain`: `AgentDomain::Tero` (or `Lang*` if the hit was lang-scoped),
   - `importance` / feedback from user or orchestrator.
3. Later tasks use `retrieve_context(query, limit, Some(AgentDomain::Tero))` to pull **persistent** scoped memory without re-querying the full L1 index for every step.

**Invariant:** Tero remains authoritative for **citations and corpus structure**; the gate stores **agent-durable** distillations, not a second copy of the index.

### Flow 2 — Gate retrieve before act

1. Before tool use or codegen, orchestrator calls `retrieve_context(query, limit, domain_filter)` with the appropriate `AgentDomain` (e.g. `Tero`, `Infrastructure`, `Workspace`).
2. Retrieved `LearningContext` items are injected into the agent prompt or cabal W2 shared schema (domain as scoping key).
3. New outcomes from the act loop feed back via `learn_from_interaction` on the same domain.

### Flow 3 — Index refresh vs memory

- **Tero index** refresh (`generate_lite_index.py`, `update-tero.sh`) updates L1; it does **not** automatically rewrite gate vectors.
- Reconciliation policy (open): re-learn on index version bump, TTL stale memories, or explicit `delete_experience` — **owned by tero-rs adapter design**.

---

## Non-goals for Wave C (and this PR)

- No `Cargo.toml` dependency on `tero-rs` or `tero-mcp` from `memory-gate-rs`
- No MCP binary in memory-gate-rs tree (WS-14 optional later)
- No implementation of cabal/orchestrator glue — document patterns only; cabal **CommonMemory** W2 facade remains the orchestration mirror

---

## Integration targets (next wave)

| Target | Intended relationship |
|--------|------------------------|
| **tero-rs** | Rust API for L1 query + citations; optional **adapter trait** to call memory-gate-rs for learn/retrieve |
| **tero-mcp** | Exposes tero tools to MCP hosts; may gain **companion** memory tools or delegate to a memory-gate MCP server |

Bulletin follow-on (from `mg/golden-recall`): after Wave C merges, wire persistent store behind tero-rs / tero-mcp with domain `Tero` / `MemoryGate` and cited L1 reads → learn/retrieve.

---

## Open questions (for tero-rs / cabal owners)

1. **Adapter trait ownership** — Should `MemoryGateway` implement a tero-defined `PersistentMemory` trait, or should tero-rs own a thin `MemoryGateClient` wrapper around the public `MemoryGateway` API? (Avoid circular crate deps; prefer trait in the **orchestration** crate or a small `memory-gate-tero-bridge` crate.)
2. **Citation → content shape** — Exact `LearningContext.content` and `metadata` keys for tero citation IDs, index generation, and replay.
3. **Dedup** — When the same tero `query_by_id` is learned twice, merge by key, bump importance, or append?
4. **Auth / tenancy** — Per-workspace store paths vs single collection with `AgentDomain::Workspace` + metadata (multi-tenant MCP).
5. **MCP surface split** — Single server vs tero-mcp + memory-gate-mcp registration in dev-mcp index.

---

## References

- [README.md](../README.md) — M1 domain + facade section, `AgentDomain::Tero`
- [WAVE_C_ACCEPTANCE.md](WAVE_C_ACCEPTANCE.md) — what ships before integration work starts
- [AGENTS.md](../AGENTS.md) — tero-first rule, `docs/tero-index/index.json`
- Workspace: `work/bulletins/mg-golden-recall.md`, cabal W2 / `WORKSPACE_CABAL_TERO_READINESS.md`