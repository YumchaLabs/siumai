# OpenAI-Compatible Usage Policy Alignment - Evidence And Gates

Status: Completed
Last updated: 2026-05-18

## Planned Gates

- `cargo fmt -p siumai-protocol-openai -p siumai-provider-openai-compatible -- --check`
- `cargo nextest run -p siumai-protocol-openai --all-features --no-fail-fast`
- `cargo nextest run -p siumai-provider-openai-compatible --all-features --no-fail-fast`
- `git diff --check`

## Evidence Log

- 2026-05-18: Opened lane from issue #20 follow-on analysis.
- 2026-05-18: `cargo check -p siumai-protocol-openai --all-features` passed.
- 2026-05-18: `cargo check -p siumai-provider-openai-compatible --all-features` passed.
- 2026-05-18: `cargo nextest run -p siumai-protocol-openai --all-features usage --no-fail-fast`
  passed, 63 passed / 387 skipped.
- 2026-05-18: `cargo nextest run -p siumai-provider-openai-compatible --all-features siliconflow --no-fail-fast`
  passed, 6 passed / 233 skipped.
- 2026-05-18: `cargo fmt -p siumai-protocol-openai -p siumai-provider-openai-compatible -- --check`
  passed.
- 2026-05-18: `cargo nextest run -p siumai-protocol-openai --all-features --no-fail-fast`
  passed, 450 passed.
- 2026-05-18: `cargo nextest run -p siumai-provider-openai-compatible --all-features --no-fail-fast`
  passed, 239 passed.
- 2026-05-18: `git diff --check` passed with Windows LF-to-CRLF warnings only.

## Known Environment Notes

- On this Windows environment, `cargo fmt --all -- --check` may fail with `os error 206` because
  the workspace command line is too long. Use package-scoped formatting gates for changed crates.
