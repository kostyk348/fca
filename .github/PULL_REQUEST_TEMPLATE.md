## What this changes

<!-- One paragraph. Link the issue if there is one: "Closes #12" -->

## Type

- [ ] Bug fix
- [ ] Feature
- [ ] Documentation
- [ ] CI / build / tooling
- [ ] Refactor (no behaviour change)

## Output compatibility

- [ ] Bit-identical (goldens untouched)
- [ ] Output changes on purpose (goldens regenerated, breaking — explain)

## Checks

- [ ] `cargo fmt --check`
- [ ] `cargo clippy --all-targets -- -D warnings`
- [ ] `cargo test` (golden tests green)
- [ ] `fca_video` smoke: gray / yuv444 / 16-bit modes still produce correct sizes
