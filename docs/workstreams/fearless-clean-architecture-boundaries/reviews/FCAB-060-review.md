# FCAB-060 Review

Status: Accepted
Date: 2026-05-21
Scope: promoted OpenAI-compatible vendor cleanup, focused on TogetherAI provider-owned image
runtime and public extension exports.

## Workstream Compliance

- No blocking findings.
- The task goal is satisfied for this slice: TogetherAI image runtime responsibilities no longer
  live in the registry factory. The registry now builds and projects provider-owned clients while
  keeping provider resolution and build override wiring local.
- The change aligns with FCAB-M2 ownership:
  - shared OpenAI-compatible text/audio execution stays in the OpenAI-compatible runtime;
  - TogetherAI provider-owned image and rerank surfaces live in `siumai-provider-togetherai`;
  - facade exports are scoped under `siumai::provider_ext::togetherai`.
- Groq, xAI, and DeepSeek were audited through the FCAB-060 registry gate and remain provider-owned
  wrapper/metadata/typed-option surfaces around the shared runtime; no additional long-tail vendor
  split is required for this task.
- Workstream docs, evidence, journal, review, handoff, milestone, and TODO ledger were updated.

## Code Quality

- No blocking findings.
- `siumai-provider-togetherai::providers::togetherai::image` is a real provider runtime module, not
  a pass-through shim. It owns body construction, provider-option merging, size parsing, edit
  validation, warning creation, response parsing, HTTP execution, metadata, and capability traits.
- `build_togetherai_json_headers` removes duplicated rerank/image header construction and keeps
  TogetherAI auth/header quirks provider-owned.
- `TogetherAiBuilder::build_image_model` gives users a provider-owned image entry point while
  preserving existing auth, base URL, HTTP config, interceptors, fetch transport, and retry options.
- Source guards cover the architectural seam from both sides:
  - provider module must expose and own `TogetherAiImageClient`;
  - registry factory must not reintroduce image runtime structs, request body builders, response
    parsers, or direct `execute_json_request` image calls.

## Missing Gates

- No missing FCAB-060 task-local gates.
- The public path parity gate was run with `--no-default-features --features togetherai` so it
  proves the TogetherAI module without compiling unrelated default OpenAI parity modules.
  The broader default-feature version currently fails on existing directional-content references to
  `siumai::prelude::unified::ContentPart`; FCAB-070 owns that export cleanup.

## Residual Risk

- FCAB-070 must resolve the existing public-path `ContentPart` compile issue by tightening the
  directional content/facade boundary.
- Native OpenAI completion conversion remains intentionally provider-owned until a separate
  native-provider seam task decides otherwise.
- Provider-utils/core isolation, bridge target adapter ownership, facade finalization, and family
  taxonomy remain in later FCAB milestones.
