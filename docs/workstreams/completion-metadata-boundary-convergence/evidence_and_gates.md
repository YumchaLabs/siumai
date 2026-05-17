# Completion Metadata Boundary Convergence - Evidence And Gates

Last updated: 2026-05-17

## Required Gates

- `cargo fmt -p siumai-protocol-openai -p siumai-provider-openai -p siumai-provider-openai-compatible`
- `cargo nextest run -p siumai-protocol-openai --all-features --no-fail-fast`
- `cargo nextest run -p siumai-provider-openai --features openai --no-fail-fast`
- `cargo nextest run -p siumai-provider-openai-compatible --all-features --no-fail-fast`

## Evidence

- `cargo fmt -p siumai-protocol-openai -p siumai-provider-openai -p siumai-provider-openai-compatible`
  passed on 2026-05-17.
- `cargo nextest run -p siumai-protocol-openai --all-features completion_metadata --no-fail-fast`
  passed: 3 tests.
- `cargo nextest run -p siumai-provider-openai --features openai openai_completion --no-fail-fast`
  passed: 6 tests.
- `cargo nextest run -p siumai-provider-openai-compatible --all-features completion_ --no-fail-fast`
  passed: 10 tests.
- `cargo nextest run -p siumai-protocol-openai --all-features --no-fail-fast`
  passed: 440 tests.
- `cargo nextest run -p siumai-provider-openai --features openai --no-fail-fast`
  passed: 125 tests.
- `cargo nextest run -p siumai-provider-openai-compatible --all-features --no-fail-fast`
  passed: 213 tests.
- `git diff --check` passed on 2026-05-17 with LF/CRLF warnings only.

## Notes

- If `siumai-provider-openai --features openai` is too broad for this lane, run focused native
  completion tests first, then broaden before closeout.
- Windows `git diff --check` may report LF/CRLF warnings; actionable whitespace errors still need
  fixing.
