# Contributing to libfca

Short rules for a numerics-heavy codebase.

## Ground truth

- **The integer arithmetic is the spec.** Results must be identical on every
  platform, every release. No floats in render paths, no architecture-branching
  math without a golden test for every branch.
- **Golden tests are the contract.** `tests/golden.rs` embeds bit-exact outputs
  of the original C implementation. If your change alters any of them, that is
  a breaking change: regenerate the goldens deliberately, say so loudly in the
  PR, and bump the minor version.
- Determinism over cleverness: fixed point, explicit shifts, no UB, no
  fast-math.

## Dev environment

A stable Rust toolchain (1.80+) is everything:

```bash
cargo test
```

To regenerate the golden constants you also need a C++ compiler (the generator
compiles the original headers; see the maintainer scripts `fca_golden.cpp` and
`fca_make_golden_tests.py`).

## Before you open a PR

- [ ] `cargo fmt --check`
- [ ] `cargo clippy --all-targets -- -D warnings`
- [ ] `cargo test` (goldens green — or regenerated on purpose)
- [ ] New behaviour has a test
- [ ] README/README.ru.md updated if the CLI surface, pipeline or docs changed

## PR style

Small, single-purpose PRs with a conventional-commit title (`feat:`, `fix:`,
`docs:`, `ci:`, `refactor:`). The PR template asks for the type and the
output-compatibility impact — fill it in and you are done.
