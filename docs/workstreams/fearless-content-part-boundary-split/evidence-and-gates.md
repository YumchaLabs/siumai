# Fearless ContentPart Boundary Split - Evidence And Gates

Last updated: 2026-05-16

## Evidence Anchors

- Existing direct construction audit:
  `docs/workstreams/fearless-spec-core-boundary-convergence/content-part-construction-audit.md`
- Spec request/response projection guard:
  `siumai-spec/tests/content_projection_boundary_test.rs`
- Facade audit guard:
  `siumai/tests/facade_architecture_boundary_test.rs`
- Core provider-map guard:
  `siumai-core/tests/core_provider_boundary_test.rs`

## Required Gates

- `cargo fmt --package siumai-spec --package siumai-core --package siumai-bridge --package siumai --check`
- `cargo nextest run -p siumai-spec --test content_projection_boundary_test --no-default-features --no-fail-fast`
- `cargo nextest run -p siumai-core --test core_provider_boundary_test --no-default-features --no-fail-fast`
- `cargo nextest run -p siumai --test facade_architecture_boundary_test --features openai,anthropic,google --no-default-features --no-fail-fast`
- `git diff --check`

## Validation Log

- CPB-020:
  - `cargo fmt --package siumai --check`
  - `cargo nextest run -p siumai --test facade_architecture_boundary_test content_part_provider_map_audit_covers_high_value_production_hits --features openai,anthropic,google --no-default-features --no-fail-fast`
  - `cargo nextest run -p siumai-spec --test content_projection_boundary_test --no-default-features --no-fail-fast`
- CPB-030:
  - `cargo fmt --package siumai-provider-google-vertex --check`
  - `cargo nextest run -p siumai-provider-google-vertex --lib vertex_gemini_image_request_content_construction_is_centralized build_image_chat_request_routes_prompt_through_request_text_adapter image_input_part_maps_provider_options_without_provider_metadata --features google-vertex --no-default-features --no-fail-fast`
  - `cargo nextest run -p siumai-spec --test content_projection_boundary_test --no-default-features --no-fail-fast`
- CPB-040:
  - `cargo fmt --package siumai-core --check`
  - `cargo nextest run -p siumai-core --lib stream_processor_routes_text_response_parts_through_response_adapter terminal_text_extraction_uses_response_text_adapter_defaults --no-default-features --no-fail-fast`
  - `cargo nextest run -p siumai-core --test core_provider_boundary_test --no-default-features --no-fail-fast`
  - `cargo nextest run -p siumai-spec --test content_projection_boundary_test --no-default-features --no-fail-fast`
- CPB-050:
  - `cargo fmt --package siumai-spec --package siumai-core --package siumai-provider-google-vertex --package siumai --check`
  - `cargo fmt --package siumai-spec --package siumai-core --package siumai-bridge --package siumai --check`
  - `cargo nextest run -p siumai-spec --test content_projection_boundary_test --no-default-features --no-fail-fast`
  - `cargo nextest run -p siumai-core --test core_provider_boundary_test --no-default-features --no-fail-fast`
  - `cargo nextest run -p siumai --test facade_architecture_boundary_test --features openai,anthropic,google --no-default-features --no-fail-fast`
  - `git diff --check`
