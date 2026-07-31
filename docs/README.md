# Documentation index — memory-gate-rs

Project-management and deep-back docs. Front-door quickstart lives in the root [README.md](../README.md).

## Project management (start here)

| Doc | Layer | Question it answers |
|-----|--------|---------------------|
| [DEVELOPMENT-PATH.md](./DEVELOPMENT-PATH.md) | Back | How we got here — decisions, PRs, alternatives |
| [CURRENT-STATE.md](./CURRENT-STATE.md) | Back | What works **today**, measured, VERIFIED/UNVERIFIED |
| [ROADMAP.md](./ROADMAP.md) | Back | Planned work + what would unblock each item |

## Product / quality acceptance

| Doc | Role |
|-----|------|
| [WAVE_C_ACCEPTANCE.md](./WAVE_C_ACCEPTANCE.md) | Golden recall@k criteria; RS-only accuracy ownership |
| [DEBT.md](./DEBT.md) | Accepted technical debt (e.g. DEBT-001 `vsa-accel` pin) |
| [FLEET_STANDARDS.md](./FLEET_STANDARDS.md) | Fleet CI/security/issue-close conventions for this repo |

## Architecture decisions

| Doc | Role |
|-----|------|
| [adr/0001-fail-closed-embedding-model-binding.md](./adr/0001-fail-closed-embedding-model-binding.md) | Why stores reject model/dimension mismatch |

## Agent / corpus tooling

| Path | Role |
|------|------|
| [../AGENTS.md](../AGENTS.md) | Agent workflow, tero index usage, mint/W2 notes |
| [tero-index/](./tero-index/) | Layer-1 lite index over this repo’s docs (not the runtime memory store) |

## Contributing / license debt

| Path | Role |
|------|------|
| [../CONTRIBUTING.md](../CONTRIBUTING.md) | Dev setup, doc style, check expectations |
| [../REUSE-DEBT.md](../REUSE-DEBT.md) | Residual REUSE coverage (see also issue #42) |
| [../CHANGELOG.md](../CHANGELOG.md) | Release-oriented history (may lag; prefer git + CURRENT-STATE for truth) |
