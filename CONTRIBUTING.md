# Contributing to memory-gate-rs

Thank you for your interest in contributing to memory-gate-rs! This document provides
guidelines and information for contributors.

## Code of Conduct

This project follows the [Rust Code of Conduct](https://www.rust-lang.org/policies/code-of-conduct).
Please be respectful and constructive in all interactions.

## Getting Started

### Prerequisites

- Rust 1.92 or later (check with `rustc --version`)
- Cargo (comes with Rust)
- Git

### Development Setup

```bash
# Clone the repository
git clone https://github.com/tzervas/memory-gate-rs.git
cd memory-gate-rs

# Build with all features
cargo build --all-features

# Run tests
cargo test --all-features

# Run clippy (we enforce clippy::pedantic)
cargo clippy --all-features

# Build documentation
cargo doc --all-features --no-deps --open
```

## Development Guidelines

### Code Style

We use `rustfmt` for formatting and `clippy` with strict lints:

```bash
# Format code
cargo fmt

# Check lints (must pass with no warnings)
cargo clippy --all-features
```

The project enables these lint levels in `src/lib.rs`:
- `clippy::all`
- `clippy::pedantic`
- `clippy::nursery`
- `missing_docs`
- `rustdoc::broken_intra_doc_links`

### Documentation Requirements

**Undocumented code is incomplete code.**

All public items must have documentation with:

1. **Summary line** - Brief description of what the item does
2. **Why section** - Explain the rationale behind design decisions
3. **Example** - Working code example (tested via `cargo test --doc`)
4. **Arguments/Returns/Errors** - Document all parameters and return values

Use Google-style docstring format:

```rust
/// Creates a new learning context with the specified content and domain.
///
/// # Why
///
/// Learning contexts are the atomic unit of memory in the system. By requiring
/// both content and domain at construction time, we ensure every memory has
/// the minimum metadata needed for effective retrieval and filtering.
///
/// # Arguments
///
/// * `content` - The knowledge content to store. Should be concise but complete.
/// * `domain` - The operational domain for filtering. Use `AgentDomain::General`
///   if the memory applies across domains.
///
/// # Example
///
/// ```
/// use memory_gate_rs::{LearningContext, AgentDomain};
///
/// let ctx = LearningContext::new(
///     "Restarting nginx resolves 502 errors",
///     AgentDomain::Infrastructure,
/// );
/// ```
#[must_use]
pub fn new(content: impl Into<String>, domain: AgentDomain) -> Self {
    // ...
}
```

### Testing

- Write unit tests for all new functionality
- Integration tests go in `tests/`
- Benchmark performance-critical code in `benches/`
- All tests must pass: `cargo test --all-features`

### Commit Messages

Use conventional commit format:

```
type(scope): description

[optional body]

[optional footer]
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `chore`

Examples:
- `feat(vsa): add resonator network factorization`
- `fix(storage): handle empty query results`
- `docs(readme): update quick start example`

### Pull Requests

1. Fork the repository
2. Create a feature branch from `main`
3. Make your changes with appropriate tests and documentation
4. Run the full test suite: `cargo test --all-features`
5. Run clippy: `cargo clippy --all-features`
6. Run rustfmt: `cargo fmt`
7. Submit a PR with a clear description of the changes

PR titles should follow the conventional commit format.

## Feature Development

### Adding a New Storage Backend

1. Create a new module in `src/storage/`
2. Implement `KnowledgeStore` trait (required)
3. Optionally implement `VectorStore`, `BatchKnowledgeStore`, `FilterableStore`
4. Add feature flag in `Cargo.toml`
5. Re-export from `src/storage/mod.rs` with `#[cfg(feature = "...")]`
6. Add integration tests in `tests/`
7. Document in crate-level docs and README

### Adding a New Adapter

1. Create a new module in `src/adapters/`
2. Implement `MemoryAdapter<LearningContext>` trait
3. Re-export from `src/adapters/mod.rs`
4. Add example in `examples/`

## Questions?

- Open an issue for bugs or feature requests
- Use discussions for general questions
- Check existing issues before creating new ones

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
