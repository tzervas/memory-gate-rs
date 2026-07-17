# Technical debt register (memory-gate-rs)

## DEBT-001 — `vsa-accel` optional `trit-vsa` pin (0.3.0 vs standalone 0.2.0)

**Status:** Accepted (documented)  
**Feature:** `vsa-accel` (`Cargo.toml` optional `trit-vsa`)  
**Date:** 2026-07-16

### Situation

| Context | `trit-vsa` version | Source |
|---------|-------------------|--------|
| `memory-gate-rs` `vsa-accel` (optional, crates.io consumers) | **0.3.0** | `Cargo.toml` + `Cargo.lock` |
| Workspace sibling `trit-vsa` (standalone repo / path dev) | **0.2.0** | `/trit-vsa` package version |
| `vsa-optim-rs` (standalone path sibling) | **0.2.0** | `path = "../trit-vsa"` |
| `rust-ai-core` / rust-ai monorepo fleet | **0.3** | Published alignment for accelerated stack |

The `vsa-accel` feature is **EXPERIMENTAL** and not in `default` or `full` features. Primary VSA path remains `vsa-native` (pure Rust, no `trit-vsa`).

### Decision

**Keep optional dependency at `trit-vsa = "0.3.0"`** (crates.io) for fleet consistency with `rust-ai-core` and related published crates, rather than downgrading to `0.2.0` or adding a `path` dependency in this repo.

Rationale:

1. **Published crate boundary** — `memory-gate-rs` on crates.io cannot rely on `path = "../trit-vsa"`; optional accel must resolve from the registry for downstream users who enable `vsa-accel`.
2. **Lock stability** — Pinning `0.2.0` from crates.io would diverge from the current lock (`0.3.0` checksum in `Cargo.lock`) and from fleet pins documented in `trit-vsa/pin` bulletin work; local `path` substitution builds but does not define the published contract.
3. **Low blast radius** — No production code path requires `vsa-accel` today; feature exists for future rust-ai acceleration once dependencies stabilize.

### Resolution criteria (close DEBT-001)

- [ ] `trit-vsa` **0.2.x** and **0.3.x** lines reconciled on crates.io (single supported line or explicit migration bulletin).
- [ ] `vsa-accel` integration tests and benchmarks green against the chosen pin.
- [ ] README / feature table updated with stable install instructions (no “awaiting stable releases” caveat).

### References

- `Cargo.toml` — `vsa-accel = ["dep:trit-vsa"]`, `trit-vsa = { version = "0.3.0", optional = true }`
- Sibling: `vsa-optim-rs` uses path `0.2.0` for standalone optimization work (intentional split until fleet unification).