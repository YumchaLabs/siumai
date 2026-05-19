# ContentPart Compatibility Namespace Break — Evidence And Gates

Status: Closed
Last updated: 2026-05-19

## Gate Selection

Use fresh command evidence before marking any task complete.

Baseline gates:

```powershell
cargo fmt --check -p siumai-spec -p siumai-core -p siumai
cargo nextest run -p siumai-spec --no-default-features --test content_projection_boundary_test --no-fail-fast
cargo nextest run -p siumai --test public_surface_imports_test --no-fail-fast
cargo nextest run -p siumai --test facade_architecture_boundary_test --no-fail-fast
```

Response/protocol gates when protocol response paths move:

```powershell
cargo nextest run -p siumai-protocol-openai --all-features --no-fail-fast
cargo nextest run -p siumai-protocol-anthropic --all-features --no-fail-fast
cargo nextest run -p siumai-protocol-gemini --all-features --no-fail-fast
cargo nextest run -p siumai-bridge --features openai,anthropic,google response --no-fail-fast
```

Facade/provider parity gates when public imports change:

```powershell
cargo nextest run -p siumai --test provider_public_path_parity_test --no-fail-fast
cargo nextest run -p siumai --features openai,anthropic,google,xai,groq --test public_surface_imports_test --no-fail-fast
```

## Evidence Log

### 2026-05-19 — CPN-010 workstream opened

Claim verified: a dedicated follow-up workstream now exists for the breaking public `ContentPart`
compatibility namespace move and response-side adapter deepening.

Inputs reviewed:

- `docs/adr/0008-legacy-content-part-compatibility-boundary.md`
- `docs/workstreams/fearless-content-part-boundary-split/`
- `docs/workstreams/fearless-module-deepening/FMD-080-content-part-directional-boundary-decision.md`
- current `siumai-spec` content projection tests and `siumai` facade architecture guards

Initial validation to run after docs are written:

```powershell
Get-Content docs/workstreams/content-part-compat-namespace-break/WORKSTREAM.json -Raw |
  ConvertFrom-Json | Select-Object title,status,active_task,next_task,updated | Format-List
Select-String -Path docs/workstreams/content-part-compat-namespace-break/TODO.md -Pattern 'CPN-020'
```

Result: PASS. `WORKSTREAM.json` parsed with `status=active`, `active_task=CPN-020`, and
`next_task=CPN-030`; TODO contains CPN-020 as the first executable task.

Broader Rust gates skipped for CPN-010 because this task only opens planning docs. First code gate
belongs to CPN-020.

### 2026-05-19 — CPN-020 compatibility namespace established

Claim verified: legacy chat content carriers now have an explicit compatibility namespace and the
facade/migration docs recommend that namespace for `ContentPart`.

Implementation summary:

- `siumai-spec::types::compat::content::*` and `siumai-spec::types::chat::compat::*` re-export the
  legacy chat content carrier set.
- `siumai_core::compat::content::*`, `siumai::compat::content::*`, and
  `siumai::prelude::compat::content::*` expose the same compatibility namespace at public facade
  layers.
- `docs/architecture/public-surface.md` and `docs/migration/migration-0.11.0-beta.7.md` now teach
  `siumai::compat::content::ContentPart` for migration code and steer new code toward directional
  prompt/generated-output parts.
- `siumai/tests/public_surface_imports_test.rs` compiles an explicit compat import example.
- `siumai/tests/facade_architecture_boundary_test.rs` guards the facade/module/docs paths.
- `docs/workstreams/fearless-spec-core-boundary-convergence/content-part-construction-audit.md`
  was refreshed for the new bridge request helper scan hits that the full facade guard surfaced.

Decision: CPN-020 deliberately keeps old-path aliases available so the namespace can be introduced
with clear migration docs first. CPN-030 owns the breaking public-prelude decision: remove legacy
`ContentPart` from recommended `prelude::unified` exports or prove any retained alias is
migration-only.

Fresh validation:

```powershell
cargo fmt --check -p siumai-spec -p siumai-core -p siumai
```

Result: PASS.

```powershell
cargo nextest run -p siumai-spec --no-default-features --test content_projection_boundary_test --no-fail-fast
```

Result: PASS. 8 tests run, 8 passed, 0 skipped.

```powershell
cargo nextest run -p siumai --test public_surface_imports_test --no-fail-fast
```

Result: PASS. 21 tests run, 21 passed, 0 skipped.

```powershell
cargo nextest run -p siumai --test facade_architecture_boundary_test --no-fail-fast
```

Result: PASS after refreshing the construction audit. 26 tests run, 26 passed, 0 skipped.

Iteration notes:

- A first targeted public-surface assertion exposed that `ToolResultOutput::content(...)` renders a
  JSON string through `to_string_lossy()`. The test was corrected to assert the JSON payload rather
  than raw text.
- One attempted nextest command omitted `--test` before `content_projection_boundary_test` and
  selected no tests. The workstream gate list was corrected and the proper command above passed.
- The first full facade architecture run failed because the scan guard found two newly introduced
  bridge request files absent from the historic audit. They were classified as request-side
  compatibility constructors / Gemini request normalization, then the full facade guard passed.

Review: self-review found no blocking issue in CPN-020. The compatibility namespace is additive;
the only intentional remaining architecture question is deferred to CPN-030, not hidden in this
task.

### 2026-05-19 — CPN-030 stable unified prelude cleanup

Claim verified: legacy `ContentPart` is no longer part of the recommended stable
`siumai::prelude::unified::*` surface; migration code must import it from the explicit compatibility
namespace.

Implementation summary:

- Removed `ContentPart` from the explicit `siumai_core::types::{...}` export list inside
  `siumai::prelude::unified`.
- Kept compatibility imports available through `siumai::compat::content::ContentPart` and
  `siumai::prelude::compat::content::*`.
- Updated public-surface tests so legacy provider-extension examples that still need
  `ContentPart` opt into `siumai::compat::content::ContentPart`.
- Added a public-surface compile guard for stable unified imports without `ContentPart`.
- Added a facade architecture guard that parses the `prelude::unified` module and rejects a
  `ContentPart` identifier there.
- Updated `docs/architecture/public-surface.md` to state that `prelude::unified` no longer exports
  legacy `ContentPart`.

Fresh validation:

```powershell
cargo nextest run -p siumai --test public_surface_imports_test --no-fail-fast
```

Result: PASS. 22 tests run, 22 passed, 0 skipped.

```powershell
cargo nextest run -p siumai --test facade_architecture_boundary_test --no-fail-fast
```

Result: PASS. 27 tests run, 27 passed, 0 skipped.

```powershell
cargo fmt --check -p siumai
```

Result: PASS.

Iteration notes:

- The first public-surface compile run failed after removing the unified export because several
  provider-extension tests were implicitly relying on `prelude::unified::*` for legacy
  `ContentPart`. Those examples were migrated to explicit `siumai::compat::content::ContentPart`,
  which is the desired public rule.
- The first facade architecture run failed because the broad compatibility-alias guard used a raw
  substring check for `ContentPart`, which also matched directional names such as
  `GenerateTextContentPart`. A dedicated identifier-based guard now owns the exact legacy
  `ContentPart` assertion.

Review: self-review found no blocking issue. The stable-prelude break is intentional and now has
both compile and source-architecture evidence.

### 2026-05-19 — CPN-040 response compatibility adapter deepened

Claim verified: response-side legacy `ContentPart` -> generated-output projection is now owned by a
named adapter module rather than the broad generated-output shape module, with behavior preserved.

Implementation summary:

- Added `siumai-spec/src/types/ai_sdk/response_compat_projection.rs` as the explicit response
  compatibility seam.
- Moved `project_response_content_part_to_generate_text_content_part`,
  `project_response_content_to_generate_text_content_parts`, and
  `project_chat_response_to_generate_text_content_parts` into that module.
- Kept the public API unchanged by re-exporting the module from `siumai-spec/src/types/ai_sdk/mod.rs`.
- Left `GenerateTextContentPart` and `GenerateTextContentPartProjectionError` in
  `generate_text.rs`, so output-shape definitions stay separate from legacy response projection.
- Updated spec/facade source guards to prove `generate_text.rs` no longer owns local
  `ContentPart::*` response mapping and the adapter documents that it ignores request options.

Fresh validation:

```powershell
cargo nextest run -p siumai-spec --no-default-features --test content_projection_boundary_test --no-fail-fast
```

Result: PASS. 8 tests run, 8 passed, 0 skipped.

```powershell
cargo nextest run -p siumai --test facade_architecture_boundary_test --no-fail-fast
```

Result: PASS. 27 tests run, 27 passed, 0 skipped.

```powershell
cargo nextest run -p siumai --test public_surface_imports_test --no-fail-fast
```

Result: PASS. 22 tests run, 22 passed, 0 skipped.

```powershell
cargo fmt --check -p siumai-spec -p siumai
```

Result: PASS.

Review: self-review found no blocking issue. CPN-040 is intentionally a seam/deep-module refactor;
it does not migrate protocol parsers yet. CPN-050 should choose one high-value protocol/bridge
response path if a real external adapter migration remains beneficial.

### 2026-05-19 — CPN-050 protocol response path split decision

Claim verified: no narrow protocol/bridge response path can adopt the generated-output adapter in
this workstream without broadening into a parser/encoder redesign.

Inspection performed:

```powershell
rg -n "project_response_content_part_to_generate_text_content_part|project_chat_response_to_generate_text_content_parts|GenerateTextContentPart|ContentPart::|provider_metadata:|provider_options:" siumai-bridge/src/response siumai-bridge/src/stream siumai-protocol-openai/src siumai-protocol-anthropic/src siumai-protocol-gemini/src -g "*.rs"
```

Result: the high-value response paths are still protocol-native serializers/parsers over
`ChatResponse` and legacy `ContentPart`:

- `siumai-protocol-openai/src/standards/openai/json_response.rs` serializes OpenAI Responses output
  items directly from `ContentPart` plus OpenAI typed provider metadata.
- `siumai-protocol-anthropic/src/standards/anthropic/json_response.rs` and
  `siumai-protocol-gemini/src/standards/gemini/json_response.rs` similarly target provider-native
  wire shapes, not generated-output parts.
- `siumai-bridge/src/response/*` delegates response serialization through protocol encoders and
  inspects loss/metadata preservation; it is not a generated-output projection path.
- `siumai-bridge/src/stream/openai_responses_parts_bridge.rs` is stream-event/replay oriented and
  does not produce non-V4 `GenerateTextContentPart` output.

Decision: CPN-050 is intentionally deferred/split. The adapter shipped in CPN-040 is the correct
facade/spec seam today. Migrating protocol parsers should wait until a protocol response model can
carry generated-output parts natively or until a concrete gateway use-case needs that projection.
Forcing one protocol path through `GenerateTextContentPart` now would add lossy round-trips and
increase coupling rather than deepen the module boundary.

Validation:

- No additional protocol package gates were run for CPN-050 because no protocol code changed in
  this task. The relevant CPN-040 gates remain the fresh code evidence for the adapter seam.
- The audit note in
  `docs/workstreams/fearless-spec-core-boundary-convergence/content-part-construction-audit.md` was
  updated to point at `response_compat_projection.rs`.

Review: self-review found this is a valid scope-control decision, not an implementation miss.
Proceed to CPN-060 migration/docs/source-guard closeout updates.

### 2026-05-19 — CPN-060 migration and source-guard closeout

Claim verified: migration docs, architecture docs, and source guards now prevent new work from
presenting legacy `ContentPart` as canonical.

Implementation summary:

- `docs/migration/migration-0.11.0-beta.7.md` now includes the stable-prelude removal in the TL;DR,
  a before/after import block, explicit `siumai::compat::content::ContentPart`, replacement
  directional request/response content families, and named response adapter behavior.
- `docs/architecture/public-surface.md` now states near the recommended prelude that legacy
  `ContentPart` is not part of `prelude::unified` and gives the replacement request/response
  families.
- `siumai/tests/facade_architecture_boundary_test.rs` now guards the migration doc's response
  adapter explanation and public-surface replacement guidance.
- `siumai-spec/tests/content_projection_boundary_test.rs` now guards that
  `response_compat_projection.rs` may mention request `providerOptions` only as documented ignored
  input and must not construct `provider_options:` fields.

Fresh validation:

```powershell
cargo nextest run -p siumai-spec --no-default-features --test content_projection_boundary_test --no-fail-fast
```

Result: PASS. 8 tests run, 8 passed, 0 skipped.

```powershell
cargo nextest run -p siumai --test facade_architecture_boundary_test --no-fail-fast
```

Result: PASS. 27 tests run, 27 passed, 0 skipped.

```powershell
cargo fmt --check -p siumai-spec -p siumai
```

Result: PASS.

Review: self-review found no blocking issue. CPN-070 closeout is next.

### 2026-05-19 — CPN-070 closeout verification

Claim verified: the workstream target state is complete. Legacy `ContentPart` has an explicit
compatibility namespace, the recommended stable prelude no longer exports it, response-side
generated-output projection is adapter-owned, docs teach the migration path, and parser-wide
generated-output migration is split rather than widened into this lane.

Fresh final gates:

```powershell
cargo fmt --check -p siumai-spec -p siumai-core -p siumai
```

Result: PASS.

```powershell
cargo nextest run -p siumai-spec --no-default-features --test content_projection_boundary_test --no-fail-fast
```

Result: PASS. 8 tests run, 8 passed, 0 skipped.

```powershell
cargo nextest run -p siumai --test public_surface_imports_test --no-fail-fast
```

Result: PASS. 22 tests run, 22 passed, 0 skipped.

```powershell
cargo nextest run -p siumai --test facade_architecture_boundary_test --no-fail-fast
```

Result: PASS. 27 tests run, 27 passed, 0 skipped.

```powershell
cargo nextest run -p siumai-bridge --features openai,anthropic,google response --no-fail-fast
```

Result: PASS. 53 tests run, 53 passed, 56 skipped.

```powershell
cargo nextest run -p siumai --features openai,anthropic,google,xai,groq --test public_surface_imports_test --no-fail-fast
```

Result: PASS. 27 tests run, 27 passed, 0 skipped.

Follow-on split:

- Protocol/parser generated-output migration remains a separate future lane. CPN-050 inspected the
  high-value response paths and found they still serialize provider-native `ChatResponse` /
  `ContentPart` shapes directly. A future lane should start only when a protocol response model can
  carry generated-output parts natively or a concrete gateway use-case needs that projection.

Review: closeout self-review found no blocking issues. `WORKSTREAM.json` status is closed and the
task ledger is complete, with CPN-050 intentionally deferred/split.
