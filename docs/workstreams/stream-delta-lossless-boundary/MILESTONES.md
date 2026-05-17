# Stream Delta Lossless Boundary - Milestones

Status: Closed
Last updated: 2026-05-17

## M0 - Scope And Evidence Freeze

Exit criteria:

- Problem and target state are explicit.
- Non-goals are explicit.
- Vercel AI SDK reference files are linked.
- First proof target is chosen.

Primary evidence:

- `docs/workstreams/stream-delta-lossless-boundary/DESIGN.md`
- `docs/workstreams/stream-delta-lossless-boundary/TODO.md`

## M1 - Lossless Extractor Semantics

Exit criteria:

- Generated text/reasoning extraction is visibly separate from control string extraction.
- Whitespace-only generated deltas are preserved by tests.
- Empty-string behavior is either preserved intentionally or aligned with field-presence semantics.

Primary gate:

```bash
cargo nextest run -p siumai-protocol-openai --all-features
```

Result: passed, 431 tests.

## M2 - Shared Stream Factory Guard

Exit criteria:

- Shared stream factory tests prove whitespace-bearing JSON frames reach the converter.
- Raw empty SSE data remains ignored as framing.

Primary gate:

```bash
cargo nextest run -p siumai-core streaming::factory
```

Result: passed, 5 tests.

## M3 - Provider Surface Regression Gate

Exit criteria:

- OpenAI-compatible provider public path has regression coverage for whitespace-only streamed
  content/reasoning.
- Provider gate passes without relying on raw protocol-only tests.

Primary gate:

```bash
cargo nextest run -p siumai-provider-openai-compatible --all-features
```

Result: passed, 213 tests.

## M4 - Closeout

Exit criteria:

- Gate set is recorded.
- Remaining provider-wide audit is completed, explicitly deferred, or split.
- `WORKSTREAM.json` status is updated.

Result: closed with no split follow-on.
