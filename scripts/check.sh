#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
MODE="${1:-}"
export CARGO_TERM_COLOR="${CARGO_TERM_COLOR:-always}"
export RUST_BACKTRACE="${RUST_BACKTRACE:-1}"
TOOLCHAIN="${RUSTUP_TOOLCHAIN:-stable}"
CARGO=(cargo)
if command -v rustup >/dev/null 2>&1; then
  rustup component add rustfmt clippy --toolchain "$TOOLCHAIN" >/dev/null 2>&1 || true
  CARGO=(cargo "+$TOOLCHAIN")
fi

if [[ "$MODE" == "--fix" ]]; then
  "${CARGO[@]}" fmt
else
  "${CARGO[@]}" fmt --check
fi
"${CARGO[@]}" clippy --all-targets --all-features -- -D warnings
RUSTDOCFLAGS="-D warnings" "${CARGO[@]}" doc --all-features --no-deps
"${CARGO[@]}" build --all-features
"${CARGO[@]}" test --all-features --verbose

echo "OK: memory-gate-rs checks passed"
