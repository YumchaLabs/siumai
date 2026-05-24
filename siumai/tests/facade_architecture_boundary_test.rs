use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

fn crate_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
}

fn workspace_root() -> PathBuf {
    crate_root()
        .parent()
        .expect("facade crate should live under workspace root")
        .to_path_buf()
}

fn read_source(relative_path: &str) -> String {
    fs::read_to_string(crate_root().join(relative_path)).expect("read source file")
}

fn normalized_workspace_path(workspace_root: &Path, path: &Path) -> String {
    path.strip_prefix(workspace_root)
        .expect("source path under workspace root")
        .iter()
        .map(|component| component.to_string_lossy())
        .collect::<Vec<_>>()
        .join("/")
}

fn rust_sources_under(relative_path: &str) -> Vec<PathBuf> {
    let mut pending = vec![crate_root().join(relative_path)];
    let mut sources = Vec::new();

    while let Some(path) = pending.pop() {
        for entry in fs::read_dir(&path).expect("read source directory") {
            let entry = entry.expect("source directory entry");
            let path = entry.path();
            if path.is_dir() {
                pending.push(path);
            } else if path.extension().and_then(|extension| extension.to_str()) == Some("rs") {
                sources.push(path);
            }
        }
    }

    sources
}

fn workspace_rust_sources_under(workspace_root: &Path, relative_path: &str) -> Vec<PathBuf> {
    let mut pending = vec![workspace_root.join(relative_path)];
    let mut sources = Vec::new();

    while let Some(path) = pending.pop() {
        for entry in fs::read_dir(&path).expect("read workspace source directory") {
            let entry = entry.expect("workspace source directory entry");
            let path = entry.path();
            if path.is_dir() {
                pending.push(path);
            } else if path.extension().and_then(|extension| extension.to_str()) == Some("rs") {
                sources.push(path);
            }
        }
    }

    sources
}

fn prelude_source() -> String {
    public_namespace_module_source("prelude")
}

fn prelude_unified_source(prelude_rs: &str) -> &str {
    let unified_start = prelude_rs
        .find("pub mod unified {")
        .expect("unified prelude module");
    let compat_start = prelude_rs[unified_start..]
        .find("pub mod compat {")
        .expect("compat prelude module");
    &prelude_rs[unified_start..unified_start + compat_start]
}

fn public_namespace_module_source(module: &str) -> String {
    let lib_rs = read_source("src/lib.rs");
    let macros_start = lib_rs
        .find("// Macros moved to a dedicated module for cleanliness")
        .expect("macros module marker");
    let root_source = &lib_rs[..macros_start];
    assert!(
        root_source
            .lines()
            .map(str::trim)
            .any(|line| line == format!("pub mod {module};")),
        "facade root should declare `siumai::{module}` as a named module"
    );
    assert!(
        !root_source.contains(&format!("pub mod {module} {{")),
        "facade root should not inline `siumai::{module}` implementation"
    );

    read_source(&format!("src/{module}.rs"))
}

fn experimental_source() -> String {
    public_namespace_module_source("experimental")
}

fn source_identifiers(source: &str) -> BTreeSet<String> {
    source
        .split(|ch: char| !(ch == '_' || ch.is_ascii_alphanumeric()))
        .filter(|identifier| !identifier.is_empty())
        .map(str::to_owned)
        .collect()
}

fn is_audited_unified_identifier(identifier: &str) -> bool {
    matches!(identifier, "CancelHandle")
}

#[test]
fn facade_keeps_provider_extension_bodies_out_of_lib_rs() {
    let lib_rs = read_source("src/lib.rs");
    let provider_ext_rs = read_source("src/provider_ext.rs");

    assert!(
        lib_rs.contains("pub mod provider_ext;"),
        "siumai/src/lib.rs should declare provider_ext as an external module"
    );
    assert!(
        lib_rs.contains("pub use crate::provider_ext as providers;"),
        "siumai::providers should stay a thin alias for provider_ext"
    );
    assert!(
        !lib_rs.contains("pub mod provider_ext {"),
        "provider extension bodies must stay out of siumai/src/lib.rs"
    );
    assert!(
        !lib_rs.contains(
            "pub use siumai_provider_openai_compatible::siumai_for_each_openai_compatible_provider;"
        ),
        "the OpenAI-compatible provider-list macro is provider-owned and should not be re-exported from the facade root"
    );

    for provider in ["openai", "anthropic", "gemini", "google_vertex", "xai"] {
        let declaration = format!("pub mod {provider};");
        assert!(
            provider_ext_rs.contains(&declaration),
            "provider_ext.rs should own the {provider} module declaration"
        );
    }
}

#[test]
fn gemini_model_catalog_stays_out_of_provider_reexport_glue() {
    let gemini_rs = read_source("src/provider_ext/gemini.rs");
    let gemini_models_rs = read_source("src/provider_ext/gemini/models.rs");

    assert!(
        gemini_rs.contains("pub mod models;"),
        "Gemini provider extension should declare the model catalog as a dedicated module"
    );
    assert!(
        !gemini_rs.contains("pub mod models {"),
        "Gemini model-id catalog should not be inlined in provider_ext/gemini.rs"
    );
    let model_reexport_line = gemini_rs
        .lines()
        .find(|line| line.trim_start().starts_with("pub use models::{"))
        .expect("Gemini provider extension should re-export model-id groups");
    for family_module in [
        "agents",
        "chat",
        "embedding",
        "image",
        "interactions",
        "model_sets",
        "video",
    ] {
        assert!(
            model_reexport_line.contains(family_module),
            "Gemini model-id group path `{family_module}` should remain available from provider_ext::gemini"
        );
        let declaration = format!("pub mod {family_module}");
        assert!(
            gemini_models_rs.contains(&declaration),
            "Gemini model catalog should expose the {family_module} model group"
        );
    }
}

#[test]
fn provider_ext_legacy_params_are_explicitly_scoped() {
    let provider_modules = [
        (
            "openai",
            [
                "OpenAiParams",
                "OpenAiParamsBuilder",
                "FunctionChoice",
                "ResponseFormat",
                "ToolChoice",
            ]
            .as_slice(),
        ),
        ("anthropic", ["AnthropicParams", "CacheControl"].as_slice()),
        (
            "gemini",
            [
                "GeminiParams",
                "GeminiParamsBuilder",
                "GenerationConfig",
                "SafetyCategory",
                "SafetySetting",
                "SafetyThreshold",
            ]
            .as_slice(),
        ),
    ];

    for (provider, legacy_names) in provider_modules {
        let source = read_source(&format!("src/provider_ext/{provider}.rs"));
        let legacy_start = source
            .find("pub mod legacy_params")
            .unwrap_or_else(|| panic!("provider_ext::{provider} should expose legacy_params"));
        let legacy_source = &source[legacy_start..];
        let root_source = &source[..legacy_start];
        let root_identifiers = source_identifiers(root_source);

        for legacy_name in legacy_names {
            assert!(
                source_identifiers(legacy_source).contains(*legacy_name),
                "provider_ext::{provider} should keep legacy parameter `{legacy_name}` inside legacy_params"
            );
            assert!(
                !root_identifiers.contains(*legacy_name),
                "provider_ext::{provider} must not flatten legacy parameter `{legacy_name}` at the provider root"
            );
        }
    }
}

#[test]
fn google_provider_ext_remains_the_google_package_alias() {
    let google_rs = read_source("src/provider_ext/google.rs");
    let public_surface_doc =
        fs::read_to_string(workspace_root().join("docs/architecture/public-surface.md"))
            .expect("read public surface doc");

    assert!(
        google_rs.contains("pub fn google() -> super::gemini::GeminiBuilder")
            && google_rs.contains("pub fn create_google() -> super::gemini::GeminiBuilder")
            && google_rs
                .contains("pub fn create_google_generative_ai() -> super::gemini::GeminiBuilder"),
        "provider_ext::google should own Google-named builder helpers over the Gemini runtime"
    );
    assert!(
        google_rs.contains("pub use super::gemini::*;"),
        "provider_ext::google should stay a package-level alias of the audited Gemini/Google surface instead of duplicating every re-export"
    );
    assert!(
        public_surface_doc.contains(
            "`siumai::provider_ext::google` is the Google package facade over the Gemini runtime"
        ) && public_surface_doc.contains("`siumai::provider_ext::google::legacy_params::*`"),
        "public-surface.md should document the Google alias relationship and its legacy_params path"
    );
}

#[test]
fn openai_compatible_provider_ext_low_level_aliases_are_documented() {
    let cases = [
        ("deepinfra", "DeepInfraClient", "DeepInfraConfig"),
        ("fireworks", "FireworksClient", "FireworksConfig"),
        ("mistral", "MistralClient", "MistralConfig"),
        ("moonshotai", "MoonshotAIClient", "MoonshotAIConfig"),
        ("perplexity", "PerplexityClient", "PerplexityConfig"),
        (
            "google_vertex_xai",
            "GoogleVertexXaiClient",
            "GoogleVertexXaiConfig",
        ),
        (
            "vertex_maas",
            "GoogleVertexMaasClient",
            "GoogleVertexMaasConfig",
        ),
    ];
    let public_surface_doc =
        fs::read_to_string(workspace_root().join("docs/architecture/public-surface.md"))
            .expect("read public surface doc");
    let normalized_public_surface_doc = public_surface_doc
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");

    for (provider, client_name, config_name) in cases {
        let source = read_source(&format!("src/provider_ext/{provider}.rs"));
        let builder_name = provider;
        let create_name = format!("create_{provider}");

        assert!(
            source.contains("Lower-level") && source.contains("compat client/config aliases"),
            "provider_ext::{provider} should label {client_name}/{config_name} as lower-level compat aliases"
        );
        assert!(
            source.contains(client_name) && source.contains(config_name),
            "provider_ext::{provider} should expose audited low-level compat names for migration"
        );
        assert!(
            source.contains(&format!("pub fn {builder_name}()"))
                && source.contains(&format!("pub fn {create_name}()"))
                && source.contains("SiumaiBuilder::new()"),
            "provider_ext::{provider} should keep package-level builder helpers next to compat aliases"
        );
    }

    assert!(
        normalized_public_surface_doc
            .contains("OpenAI-compatible provider extension modules may expose")
            && normalized_public_surface_doc.contains("`*Client` / `*Config` compat aliases")
            && normalized_public_surface_doc
                .contains("`provider()` and `create_provider()` builder helpers"),
        "public-surface.md should document why these low-level aliases remain visible"
    );
}

#[test]
fn experimental_bridge_is_owned_by_bridge_crate_and_reexported_by_facade() {
    let lib_rs = read_source("src/lib.rs");
    let experimental_rs = experimental_source();
    let bridge_crate_lib = fs::read_to_string(crate_root().join("../siumai-bridge/src/lib.rs"))
        .expect("read siumai-bridge lib.rs");

    assert!(
        !lib_rs.contains("mod experimental_bridge;"),
        "siumai facade should not own the bridge implementation module"
    );
    assert!(
        !lib_rs.contains("pub mod experimental_bridge;"),
        "experimental_bridge should not become a top-level public facade module"
    );
    assert!(
        experimental_rs.contains("pub use siumai_bridge::*;"),
        "siumai::experimental::bridge should re-export the dedicated bridge crate"
    );

    assert!(
        !crate_root().join("src/experimental_bridge.rs").exists(),
        "bridge implementation file should not live in the facade crate"
    );
    assert!(
        !crate_root().join("src/experimental_bridge").exists(),
        "bridge implementation directory should not live in the facade crate"
    );

    assert!(
        bridge_crate_lib.contains("This crate owns gateway/protocol conversion code")
            && bridge_crate_lib.contains("siumai-extras")
            && bridge_crate_lib.contains("siumai-core"),
        "siumai-bridge should document why the bridge lives outside the facade and core crates"
    );
}

#[test]
fn fearless_clean_architecture_inventory_tracks_current_guard_surfaces() {
    let inventory = fs::read_to_string(
        workspace_root()
            .join("docs/workstreams/fearless-clean-architecture-boundaries/seam-inventory.md"),
    )
    .expect("read FCAB seam inventory");

    for required in [
        "ProviderFactory",
        "LlmClient",
        "ContentPart",
        "OpenAI-compatible",
        "provider-utils",
        "BridgeTarget",
        "siumai-registry/src/registry/entry/boundary_tests.rs",
        "siumai-core/tests/core_provider_boundary_test.rs",
        "siumai/tests/facade_architecture_boundary_test.rs",
        "siumai/tests/public_surface_imports_test.rs",
        "siumai-protocol-openai/tests/openai_compat_boundary_test.rs",
        "siumai-provider-openai-compatible/src/providers/openai_compatible/openai_client/tests.rs",
        "siumai-bridge/src/request/normalize.rs",
        "siumai-core/src/utils",
        "siumai-core/src/streaming",
        "siumai-core/src/execution",
        "siumai-core/src/retry",
        "siumai-core/src/encoding",
    ] {
        assert!(
            inventory.contains(required),
            "FCAB seam inventory should track `{required}` so later refactors keep the architecture guard set discoverable"
        );
    }
}

#[test]
fn generate_text_projection_delegates_content_part_mapping_to_spec() {
    let text_rs = read_source("src/text.rs");
    let spec_generate_text_rs =
        fs::read_to_string(workspace_root().join("siumai-spec/src/types/ai_sdk/generate_text.rs"))
            .expect("read spec generate_text source");
    let spec_response_adapter_rs = fs::read_to_string(
        workspace_root().join("siumai-spec/src/types/ai_sdk/response_compat_projection.rs"),
    )
    .expect("read spec response compatibility adapter source");
    let spec_ai_sdk_mod_rs =
        fs::read_to_string(workspace_root().join("siumai-spec/src/types/ai_sdk/mod.rs"))
            .expect("read spec ai_sdk module source");
    let projection_start = text_rs
        .find("fn project_generate_text_content_part")
        .expect("generate_text content projection function");
    let fallback_start = text_rs
        .find("fn project_generate_text_legacy_compat_content_part")
        .expect("generate_text legacy compatibility fallback function");
    let push_start = text_rs
        .find("fn push_text_output")
        .expect("generate_text projection push helper");

    let projection_fn = &text_rs[projection_start..fallback_start];
    assert!(
        projection_fn.contains("project_response_content_part_to_generate_text_content_part"),
        "facade generate_text should delegate response ContentPart projection to siumai-spec"
    );
    assert!(
        spec_ai_sdk_mod_rs.contains("mod response_compat_projection;")
            && spec_ai_sdk_mod_rs.contains("pub use response_compat_projection::*;"),
        "siumai-spec should expose legacy ContentPart response projection through a named response compatibility adapter module"
    );
    assert!(
        !spec_generate_text_rs.contains("ContentPart::Text")
            && !spec_generate_text_rs.contains("ContentPart::ToolResult")
            && spec_response_adapter_rs.contains("ContentPart::Text")
            && spec_response_adapter_rs.contains("ContentPart::ToolResult")
            && spec_response_adapter_rs.contains("ignores request options"),
        "legacy ContentPart response mapping should live in response_compat_projection.rs, not the broad generate_text.rs output-shape module"
    );

    for forbidden_local_mapping in [
        "ContentPart::Text",
        "ContentPart::Custom",
        "ContentPart::File",
        "ContentPart::Reasoning",
        "ContentPart::ReasoningFile",
        "ContentPart::Source",
        "ContentPart::ToolCall",
    ] {
        assert!(
            !projection_fn.contains(forbidden_local_mapping),
            "facade generate_text should not reintroduce local `{forbidden_local_mapping}` output mapping"
        );
    }

    let fallback_fn = &text_rs[fallback_start..push_start];
    assert!(
        fallback_fn.contains("ContentPart::ToolResult") && fallback_fn.contains("input.is_none()"),
        "the only facade-local content projection fallback should be the documented legacy tool-result-without-input path"
    );
}

#[test]
fn facade_macros_only_create_request_side_empty_provider_options() {
    let macros_rs = read_source("src/macros.rs");

    for forbidden in [
        "provider_metadata",
        "ProviderMetadata",
        "ProviderMetadataMap",
    ] {
        assert!(
            !macros_rs.contains(forbidden),
            "facade macros build request messages and must not populate response metadata `{forbidden}`"
        );
    }

    let unexpected_provider_options = macros_rs
        .lines()
        .map(str::trim)
        .filter(|line| line.contains("provider_options:"))
        .filter(|line| !line.contains("ProviderOptionsMap::default()"))
        .collect::<Vec<_>>();
    assert!(
        unexpected_provider_options.is_empty(),
        "facade macros may only initialize request provider_options with empty defaults: {unexpected_provider_options:?}"
    );

    let unexpected_content_part_lines = macros_rs
        .lines()
        .map(str::trim)
        .filter(|line| line.contains("ContentPart::"))
        .filter(|line| !line.contains("ContentPart::tool_result_text"))
        .collect::<Vec<_>>();
    assert!(
        unexpected_content_part_lines.is_empty(),
        "facade macros should not become a local ContentPart projection surface: {unexpected_content_part_lines:?}"
    );

    assert!(
        !macros_rs.contains(".cache_control("),
        "facade macros should not reintroduce the removed ChatMessageBuilder::cache_control(...) path"
    );

    let lib_rs = read_source("src/lib.rs");
    assert!(
        lib_rs.contains("fn with_anthropic_cache_control"),
        "facade macros should route legacy cache-control macro support through a narrow private helper"
    );
}

#[test]
fn facade_audio_and_structured_helpers_do_not_read_request_provider_options() {
    for relative_path in [
        "src/speech.rs",
        "src/transcription.rs",
        "src/structured_output.rs",
    ] {
        let source = read_source(relative_path);
        let production_source = source
            .split("#[cfg(test)]")
            .next()
            .expect("production source section");

        assert!(
            production_source.contains("provider_metadata"),
            "{relative_path} should remain a response metadata projection helper"
        );

        for forbidden in [
            "provider_options",
            "providerOptions",
            "ProviderOptionsMap",
            "ContentPart::",
        ] {
            assert!(
                !production_source.contains(forbidden),
                "{relative_path} must not read request provider options or reintroduce local ContentPart mapping"
            );
        }
    }
}

#[test]
fn facade_video_metadata_projection_avoids_legacy_request_provider_options() {
    let source = read_source("src/video.rs");
    let production_source = source
        .split("#[cfg(test)]")
        .next()
        .expect("production source section");

    assert!(
        production_source.contains("model.polling_options(request)")
            && production_source.contains("build_call_provider_metadata")
            && production_source.contains("merge_provider_metadata"),
        "facade video should keep high-level polling options separate from response metadata aggregation"
    );

    for forbidden in [
        "provider_options_map",
        "ProviderOptionsMap",
        "ContentPart::",
    ] {
        assert!(
            !production_source.contains(forbidden),
            "facade video helpers must not depend on legacy request provider option maps or ContentPart projection"
        );
    }
}

#[test]
fn content_part_provider_map_audit_covers_high_value_production_hits() {
    let workspace_root = workspace_root();
    let audit = fs::read_to_string(workspace_root.join(
        "docs/workstreams/fearless-spec-core-boundary-convergence/content-part-construction-audit.md",
    ))
    .expect("read Track C content-part construction audit");
    let refreshed_audit =
        fs::read_to_string(workspace_root.join(
            "docs/workstreams/fearless-content-part-boundary-split/direct-content-part-scan.md",
        ))
        .expect("read ContentPart boundary split scan");
    let fcab_audit = fs::read_to_string(workspace_root.join(
        "docs/workstreams/fearless-clean-architecture-boundaries/content-part-adapter-audit.md",
    ))
    .expect("read FCAB content-part adapter audit");

    let target_dirs = [
        "siumai-core/src",
        "siumai-bridge/src",
        "siumai-protocol-openai/src",
        "siumai-protocol-anthropic/src",
        "siumai-protocol-gemini/src",
        "siumai-provider-amazon-bedrock/src",
        "siumai-provider-anthropic/src",
        "siumai-provider-google-vertex/src",
        "siumai-provider-minimaxi/src",
        "siumai-provider-openai/src",
        "siumai-provider-gemini/src",
        "siumai/src",
    ];

    let mut missing = Vec::new();
    for relative_dir in target_dirs {
        for path in workspace_rust_sources_under(&workspace_root, relative_dir) {
            let relative_path = normalized_workspace_path(&workspace_root, &path);
            let source = fs::read_to_string(&path)
                .unwrap_or_else(|error| panic!("read {relative_path}: {error}"));

            let has_content_or_provider_map_hit = source.contains("ContentPart::")
                || source.contains("provider_metadata:")
                || source.contains("provider_options:");
            if !has_content_or_provider_map_hit {
                continue;
            }

            if audit.contains(&relative_path)
                || refreshed_audit.contains(&relative_path)
                || fcab_audit.contains(&relative_path)
            {
                continue;
            }

            missing.push(relative_path);
        }
    }

    missing.sort();
    missing.dedup();

    assert!(
        missing.is_empty(),
        "ContentPart boundary audits must classify every production source that directly constructs ContentPart or provider maps before the path is accepted: {missing:#?}"
    );
}

#[test]
fn stable_unified_prelude_excludes_compatibility_construction_aliases() {
    let lib_rs = read_source("src/lib.rs");
    let prelude_rs = prelude_source();
    let unified_source = prelude_unified_source(&prelude_rs);
    let architecture_audit = fs::read_to_string(
        workspace_root()
            .join("docs/workstreams/fearless-architecture-convergence/compatibility-audit.md"),
    )
    .expect("read architecture compatibility audit");

    for forbidden in [
        "pub use crate::Provider;",
        "pub use crate::provider::Siumai;",
        "pub use crate::compat::{Siumai, SiumaiBuilder};",
        "experimental_generate_image",
        "experimental_generate_speech",
        "experimental_transcribe",
        "experimental_generate_video",
        "StreamingToolCallDelta",
        "StreamingToolCallFunctionDelta",
        "StreamingToolCallTracker",
        "StreamingToolCallTrackerOptions",
        "StreamingToolCallTypeValidation",
        "CallSettings",
        "Experimental_GenerateImageResult",
        "Experimental_GeneratedImage",
        "Experimental_LanguageModelStreamPart",
        "Experimental_SpeechResult",
        "Experimental_TranscriptionResult",
        "ExperimentalLanguageModelStreamPart",
        "experimental_filter_active_tools",
        "step_count_is",
    ] {
        assert!(
            !unified_source.contains(forbidden),
            "prelude::unified should not export compatibility-only surface `{forbidden}`"
        );
    }

    assert!(
        lib_rs.contains("pub mod prelude;")
            && prelude_rs.contains("pub mod compat {")
            && prelude_rs.contains("pub use crate::compat::{")
            && prelude_rs.contains("StreamingToolCallTracker")
            && prelude_rs.contains("Experimental_GenerateImageResult")
            && prelude_rs.contains("pub mod content")
            && prelude_rs.contains("pub use crate::compat::content::*")
            && prelude_rs.contains("step_count_is"),
        "compatibility construction and legacy helper aliases should remain explicit under prelude::compat"
    );
    assert!(
        architecture_audit.contains("`siumai::compat` and `prelude::compat` re-export deprecated")
            && architecture_audit.contains(
                "`prelude::unified` must not re-export deprecated experimental helper spellings"
            )
            && architecture_audit.contains("keep, explicit compat only"),
        "architecture compatibility audit should classify deprecated helper aliases as explicit compat-only, not stable unified prelude exports"
    );
}

#[test]
fn legacy_content_part_has_explicit_compat_namespace() {
    let lib_rs = read_source("src/lib.rs");
    let content_rs = public_namespace_module_source("content");
    let compat_rs = read_source("src/compat.rs");
    let public_surface = read_source("tests/public_surface_imports_test.rs");
    let migration_doc =
        fs::read_to_string(workspace_root().join("docs/migration/migration-0.11.0-beta.7.md"))
            .expect("read migration beta.7 doc");

    assert!(
        compat_rs.contains("pub mod content")
            && compat_rs.contains("pub use siumai_core::compat::content::*"),
        "siumai::compat::content should be the facade compatibility namespace for legacy ContentPart"
    );
    assert!(
        lib_rs.contains("pub mod content;")
            && content_rs.contains("pub use crate::compat::content::*"),
        "siumai::prelude::compat::content should re-export the legacy content compatibility namespace"
    );
    assert!(
        public_surface.contains("use siumai::compat::content::{")
            && public_surface.contains("prelude::compat::content"),
        "public import tests should exercise explicit compatibility content imports"
    );
    assert!(
        migration_doc.contains("siumai::compat::content::ContentPart"),
        "migration docs should teach the explicit ContentPart compatibility import"
    );
    assert!(
        migration_doc.contains("project_response_content_part_to_generate_text_content_part")
            && migration_doc.contains("preserves `providerMetadata`")
            && migration_doc.contains("ignores request"),
        "migration docs should explain the named response adapter and provider metadata/options directionality"
    );
}

#[test]
fn stable_unified_prelude_does_not_export_legacy_content_part() {
    let prelude_rs = prelude_source();
    let unified_source = prelude_unified_source(&prelude_rs);
    let public_surface = read_source("tests/public_surface_imports_test.rs");
    let public_surface_doc =
        fs::read_to_string(workspace_root().join("docs/architecture/public-surface.md"))
            .expect("read public surface doc");

    assert!(
        !source_identifiers(unified_source).contains("ContentPart"),
        "prelude::unified should not export the legacy dual-use ContentPart carrier; use siumai::compat::content::ContentPart for migration code"
    );
    assert!(
        public_surface
            .contains("public_surface_legacy_content_part_uses_explicit_compat_namespace")
            && public_surface.contains("use siumai::compat::content::{"),
        "public surface tests should keep legacy ContentPart examples on the explicit compat namespace"
    );
    assert!(
        public_surface_doc.contains("`prelude::unified` no longer exports legacy `ContentPart`")
            && public_surface_doc
                .contains("`siumai::compat::content::{ContentPart, MessageContent}`"),
        "public surface docs should state the stable-prelude ContentPart removal and the explicit migration path"
    );
    assert!(
        public_surface_doc
            .contains("Legacy `ContentPart` is not part of this stable unified prelude")
            && public_surface_doc.contains("Use request-directional prompt parts")
            && public_surface_doc.contains("response-directional generated output parts"),
        "public surface docs should keep replacement request/response content families visible near the recommended prelude"
    );
}

#[test]
fn directional_content_namespaces_are_visible_and_compat_is_explicit() {
    let lib_rs = read_source("src/lib.rs");
    let content_rs = public_namespace_module_source("content");
    let prelude_rs = prelude_source();
    let unified_source = prelude_unified_source(&prelude_rs);
    let spec_types_rs = fs::read_to_string(workspace_root().join("siumai-spec/src/types.rs"))
        .expect("read spec types source");
    let public_surface = read_source("tests/public_surface_imports_test.rs");
    let public_surface_doc =
        fs::read_to_string(workspace_root().join("docs/architecture/public-surface.md"))
            .expect("read public surface doc");
    let migration_doc =
        fs::read_to_string(workspace_root().join("docs/migration/migration-0.11.0-beta.7.md"))
            .expect("read migration beta.7 doc");

    assert!(
        spec_types_rs.contains("pub mod content")
            && spec_types_rs.contains("pub mod prompt")
            && spec_types_rs.contains("pub mod output")
            && spec_types_rs.contains("pub mod compat"),
        "spec/core content exports should be split into prompt, output, and compat namespaces"
    );
    assert!(
        lib_rs.contains("pub mod content;")
            && content_rs.contains("pub mod prompt")
            && content_rs.contains("pub mod output")
            && content_rs.contains("pub mod compat"),
        "facade content exports should be split into a named module with prompt, output, and compat namespaces"
    );

    assert!(
        unified_source.contains("pub use crate::content::prompt")
            && unified_source.contains("pub use crate::content::output")
            && !unified_source.contains("pub use crate::content::compat")
            && !unified_source.contains("pub use crate::compat::content"),
        "prelude::unified should expose prompt/output navigation but not legacy compat content"
    );
    assert!(
        public_surface.contains("public_surface_directional_content_namespaces_compile")
            && public_surface.contains("use siumai::content::{compat, output, prompt}")
            && public_surface.contains(
                "use siumai::prelude::unified::{output as prelude_output, prompt as prelude_prompt}"
            ),
        "public surface compile tests should exercise directional content namespaces"
    );
    assert!(
        public_surface_doc.contains("siumai::content::prompt::*")
            && public_surface_doc.contains("siumai::content::output::*")
            && public_surface_doc.contains("siumai::content::compat::*")
            && migration_doc.contains("siumai::content::prompt")
            && migration_doc.contains("siumai::content::output"),
        "public docs should teach directional content namespaces and the explicit compat namespace"
    );
}

#[test]
fn tests_and_examples_do_not_import_legacy_content_part_from_unified_prelude() {
    let workspace_root = workspace_root();
    let mut offenders = Vec::new();

    for relative_dir in ["siumai/tests", "siumai/examples", "siumai-extras/src"] {
        let root = workspace_root.join(relative_dir);
        if !root.exists() {
            continue;
        }

        for path in workspace_rust_sources_under(&workspace_root, relative_dir) {
            let relative_path = normalized_workspace_path(&workspace_root, &path);
            if relative_path == "siumai/tests/facade_architecture_boundary_test.rs" {
                continue;
            }

            let source = fs::read_to_string(&path)
                .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
            let imports_legacy_content_part_directly =
                source.contains("prelude::unified::ContentPart");
            let imports_legacy_content_part_from_glob_group = source
                .match_indices("use siumai::prelude::unified::{")
                .any(|(start, _)| {
                    source[start..]
                        .find(';')
                        .map(|end| source_identifiers(&source[start..start + end]))
                        .is_some_and(|identifiers| identifiers.contains("ContentPart"))
                });

            if imports_legacy_content_part_directly || imports_legacy_content_part_from_glob_group {
                offenders.push(relative_path);
            }
        }
    }

    offenders.sort();
    assert!(
        offenders.is_empty(),
        "legacy ContentPart examples/tests should use siumai::compat::content::ContentPart, not prelude::unified::ContentPart: {offenders:#?}"
    );
}

#[test]
fn stable_unified_prelude_does_not_mirror_core_streaming_internals() {
    let experimental_rs = experimental_source();
    let prelude_rs = prelude_source();
    let unified_source = prelude_unified_source(&prelude_rs);
    let public_surface_doc =
        fs::read_to_string(crate_root().join("../docs/architecture/public-surface.md"))
            .expect("read public surface doc");

    assert!(
        !unified_source.contains("pub use siumai_core::streaming::*;"),
        "prelude::unified should not mirror the broad siumai-core streaming module"
    );
    assert!(
        !unified_source.contains("pub use crate::parse_json_event_stream;"),
        "prelude::unified should not directly export low-level JSON/SSE parser helpers"
    );
    assert!(
        experimental_rs.contains("pub mod streaming {")
            && experimental_rs.contains("pub use siumai_core::streaming::*;"),
        "siumai::experimental::streaming should remain the explicit advanced facade path for core streaming internals"
    );

    for internal_name in [
        "SseEventConverter",
        "JsonEventConverter",
        "StreamFactory",
        "EventBuilder",
        "StreamProcessor",
        "SseJsonStreamConfig",
        "ChatByteStream",
        "TypedStreamPart",
        "UnsupportedStreamPartBehavior",
        "parse_json_event_stream",
    ] {
        assert!(
            !source_identifiers(unified_source).contains(internal_name),
            "prelude::unified should not export low-level streaming implementation type `{internal_name}`"
        );
    }

    for stable_name in [
        "ChatStream",
        "ChatStreamEvent",
        "ChatStreamPart",
        "ChatStreamHandle",
    ] {
        assert!(
            source_identifiers(unified_source).contains(stable_name),
            "prelude::unified should keep stable stream consumption type `{stable_name}`"
        );
    }

    assert!(
        public_surface_doc.contains(
            "Low-level streaming converters, factories, encoders, and bridge stream parts"
        ) && public_surface_doc.contains("siumai::experimental::streaming"),
        "public-surface.md should document where low-level streaming internals live"
    );
}

#[test]
fn stable_unified_prelude_scopes_low_level_utility_helpers() {
    let lib_rs = read_source("src/lib.rs");
    let prelude_rs = prelude_source();
    let unified_source = prelude_unified_source(&prelude_rs);
    let public_surface_doc =
        fs::read_to_string(crate_root().join("../docs/architecture/public-surface.md"))
            .expect("read public surface doc");

    let low_level_utility_names = [
        "DEFAULT_JSON_GENERIC_SUFFIX",
        "DEFAULT_JSON_SCHEMA_PREFIX",
        "DEFAULT_JSON_SCHEMA_SUFFIX",
        "DEFAULT_MAX_DOWNLOAD_SIZE",
        "Download",
        "DownloadOptions",
        "DownloadedFile",
        "HeaderRecord",
        "JsonInstructionMessageOptions",
        "JsonInstructionOptions",
        "JsonParseResult",
        "LoadApiKeyOptions",
        "LoadOptionalSettingOptions",
        "LoadSettingOptions",
        "SupportedUrlMap",
        "TypeValidationResult",
        "UrlSupportRegex",
        "combine_headers",
        "create_download",
        "download_url",
        "extract_response_headers",
        "inject_json_instruction",
        "inject_json_instruction_into_messages",
        "is_parsable_json",
        "is_provider_reference",
        "is_url_supported",
        "load_api_key",
        "load_optional_setting",
        "load_setting",
        "normalize_header_map",
        "normalize_headers",
        "normalize_optional_headers",
        "parse_json",
        "parse_json_with_schema",
        "parse_provider_options",
        "read_response_with_size_limit",
        "resolve_provider_reference",
        "safe_parse_json",
        "safe_parse_json_with_schema",
        "safe_validate_types",
        "validate_download_url",
        "validate_types",
        "with_user_agent_suffix",
        "without_trailing_slash",
    ];

    for utility_name in low_level_utility_names {
        assert!(
            !source_identifiers(unified_source).contains(utility_name),
            "prelude::unified should not export low-level utility helper `{utility_name}`"
        );
        assert!(
            source_identifiers(&lib_rs).contains(utility_name),
            "the explicit facade root should still export `{utility_name}` for opt-in utility users"
        );
    }

    for demoted_utility_name in [
        "Arrayable",
        "DEFAULT_ID_ALPHABET",
        "DEFAULT_ID_SIZE",
        "DEFAULT_REASONING_BUDGET_PERCENTAGES",
        "ReasoningBudgetOptions",
        "ReasoningLevel",
        "ReasoningLevelConversionError",
        "VERSION",
        "as_array",
        "convert_base64_to_uint8_array",
        "convert_image_model_file_to_data_uri",
        "convert_to_base64",
        "convert_data_content_to_base64_string",
        "convert_data_content_to_uint8_array",
        "convert_uint8_array_to_base64",
        "convert_uint8_array_to_text",
        "cosine_similarity",
        "delay",
        "filter_nullable",
        "get_error_message",
        "get_runtime_environment_user_agent",
        "get_text_from_data_url",
        "is_abort_error",
        "is_custom_reasoning",
        "is_deep_equal_data",
        "is_non_nullable",
        "map_reasoning_to_provider_budget",
        "map_reasoning_to_provider_effort",
        "media_type_to_extension",
        "remove_undefined_entries",
        "strip_file_extension",
    ] {
        assert!(
            !source_identifiers(unified_source).contains(demoted_utility_name),
            "prelude::unified should not export provider-utils helper `{demoted_utility_name}`; use explicit siumai:: root imports"
        );
        assert!(
            source_identifiers(&lib_rs).contains(demoted_utility_name),
            "the explicit facade root should still export `{demoted_utility_name}` for opt-in utility users"
        );
    }

    for stable_utility_name in [
        "generate_id",
        "create_id_generator",
        "IdGenerator",
        "IdGeneratorOptions",
        "json_schema",
        "json_schema_with_validator",
        "lazy_schema",
        "as_schema",
        "as_schema_or_empty",
        "empty_json_schema",
        "filter_active_tools",
        "has_tool_call",
        "is_step_count",
        "is_tool_ui_part",
        "last_assistant_message_is_complete_with_tool_calls",
        "SerialJobExecutor",
        "ToolNameMapping",
        "create_tool_name_mapping",
    ] {
        assert!(
            source_identifiers(unified_source).contains(stable_utility_name),
            "prelude::unified should keep AI SDK application helper `{stable_utility_name}`"
        );
    }

    assert!(
        public_surface_doc.contains("Low-level utility helpers are explicit root imports")
            && public_surface_doc.contains("use siumai::{parse_json, normalize_headers};")
            && public_surface_doc.contains("Provider-utils helpers that remain at the root are backed by `siumai-provider-utils`")
            && public_surface_doc.contains("prelude::unified` does not export broad provider-utils helper groups"),
        "public-surface.md should document scoped low-level utility helper imports"
    );
}

#[test]
fn stable_unified_prelude_scopes_retry_api() {
    let prelude_rs = prelude_source();
    let unified_source = prelude_unified_source(&prelude_rs);
    let public_surface_doc =
        fs::read_to_string(crate_root().join("../docs/architecture/public-surface.md"))
            .expect("read public surface doc");
    let migration_doc =
        fs::read_to_string(crate_root().join("../docs/migration/migration-0.11.0-beta.7.md"))
            .expect("read migration doc");

    assert!(
        !unified_source.contains("pub use crate::retry_api::*;"),
        "prelude::unified should not glob-export the facade retry module"
    );

    for retry_name in [
        "RetryOptions",
        "RetryBackend",
        "RetryPolicy",
        "BackoffRetryExecutor",
        "retry",
        "retry_with",
        "maybe_retry",
        "classify_http_error",
        "backoff_executor_for_provider",
        "backoff_options_for_provider",
        "retry_for_provider",
    ] {
        assert!(
            !source_identifiers(unified_source).contains(retry_name),
            "prelude::unified should not directly export retry API `{retry_name}`; use siumai::retry_api::*"
        );
    }

    assert!(
        public_surface_doc.contains("use siumai::retry_api::*;")
            && public_surface_doc.contains("prelude::unified` should not directly export"),
        "public-surface.md should document retry API as an explicit scoped module"
    );
    assert!(
        migration_doc.contains("use siumai::retry_api::{RetryOptions, RetryPolicy, retry_with};"),
        "migration docs should show the explicit retry_api import path"
    );
}

#[test]
fn facade_retry_api_exports_an_explicit_control_surface() {
    let retry_api_rs = read_source("src/retry_api.rs");
    let public_surface_doc =
        fs::read_to_string(crate_root().join("../docs/architecture/public-surface.md"))
            .expect("read public surface doc");

    assert!(
        !retry_api_rs.contains("pub use siumai_core::retry_api::*;"),
        "siumai::retry_api should not mirror every future core retry helper through a wildcard re-export"
    );

    for retry_name in [
        "BackoffRetryExecutor",
        "RetryBackend",
        "RetryOptions",
        "RetryPolicy",
        "classify_http_error",
        "maybe_retry",
        "retry",
        "retry_with",
        "backoff_executor_for_provider",
        "backoff_options_for_provider",
        "retry_for_provider",
    ] {
        assert!(
            source_identifiers(&retry_api_rs).contains(retry_name),
            "siumai::retry_api should explicitly export stable retry helper `{retry_name}`"
        );
    }

    assert!(
        public_surface_doc.contains("use siumai::retry_api::*;")
            && public_surface_doc.contains("stable retry control surface"),
        "public-surface.md should document retry_api as an explicit stable control surface"
    );
}

#[test]
fn stable_unified_prelude_does_not_mirror_tooling_runtime_module() {
    let prelude_rs = prelude_source();
    let unified_source = prelude_unified_source(&prelude_rs);
    let public_surface_doc =
        fs::read_to_string(crate_root().join("../docs/architecture/public-surface.md"))
            .expect("read public surface doc");

    assert!(
        !unified_source.contains("pub use crate::tooling;"),
        "prelude::unified should not mirror the runtime tooling module; import siumai::tooling::* explicitly"
    );

    for stable_tool_name in [
        "ExecutableTool",
        "ExecutableTools",
        "ToolExecutionOptions",
        "ToolExecutionResult",
        "ToolExecutionStream",
        "ToolModelOutputContext",
        "ToolSet",
        "ToolExecuteFunction",
        "tool",
        "dynamic_tool",
        "execute_tool",
        "is_executable_tool",
        "model_messages_from_chat_messages",
    ] {
        assert!(
            source_identifiers(unified_source).contains(stable_tool_name),
            "prelude::unified should keep AI SDK-style tool helper `{stable_tool_name}`"
        );
    }

    assert!(
        public_surface_doc.contains("use siumai::tooling::*;")
            && public_surface_doc
                .contains("prelude::unified` should not mirror the whole `tooling` module"),
        "public-surface.md should document the explicit tooling module path"
    );
}

#[test]
fn facade_tooling_module_exports_an_explicit_runtime_surface() {
    let tooling_rs = read_source("src/tooling.rs");

    assert!(
        !tooling_rs.contains("pub use siumai_core::tooling::*;"),
        "siumai::tooling should not wildcard-mirror every future core tooling helper"
    );
    assert!(
        tooling_rs.contains("pub use siumai_core::tooling::{"),
        "siumai::tooling should remain a curated facade re-export surface"
    );

    for stable_tooling_name in [
        "ExecutableTool",
        "ExecutableTools",
        "ProviderDefinedToolFactory",
        "ProviderDefinedToolFactoryWithOutputSchema",
        "ProviderExecutedToolFactory",
        "ToolExecuteFn",
        "ToolExecuteFunction",
        "ToolExecuteStreamFn",
        "ToolExecuteValueStream",
        "ToolExecuteWithOptionsFn",
        "ToolExecutionOptions",
        "ToolExecutionResult",
        "ToolExecutionStream",
        "ToolInputAvailableContext",
        "ToolInputAvailableFn",
        "ToolInputDeltaContext",
        "ToolInputDeltaFn",
        "ToolInputStartFn",
        "ToolModelOutputContext",
        "ToolModelOutputFn",
        "ToolNeedsApproval",
        "ToolNeedsApprovalContext",
        "ToolNeedsApprovalFn",
        "ToolRuntimeContext",
        "ToolRuntimeMetadata",
        "ToolSet",
        "create_provider_defined_tool_factory",
        "create_provider_defined_tool_factory_with_output_schema",
        "create_provider_executed_tool_factory",
        "dynamic_tool",
        "execute_tool",
        "is_executable_tool",
        "model_messages_from_chat_messages",
        "tool",
    ] {
        assert!(
            source_identifiers(&tooling_rs).contains(stable_tooling_name),
            "siumai::tooling should explicitly export runtime helper `{stable_tooling_name}`"
        );
    }
}

#[test]
fn facade_ui_module_exports_an_explicit_conversion_surface() {
    let ui_rs = read_source("src/ui.rs");
    let public_surface_doc =
        fs::read_to_string(crate_root().join("../docs/architecture/public-surface.md"))
            .expect("read public surface doc");

    assert!(
        !ui_rs.contains("pub use siumai_core::ui::*;"),
        "siumai::ui should not mirror every future core UI helper through a wildcard re-export"
    );

    for stable_ui_name in [
        "ConvertUiMessagesOptions",
        "SafeValidateUiMessagesResult",
        "SafeValidateUIMessagesResult",
        "UiMessageError",
        "UiSchemaValidator",
        "ValidateUiMessagesSchemaOptions",
        "convert_to_chat_request",
        "convert_to_chat_request_with",
        "convert_to_chat_request_with_tooling",
        "convert_to_model_messages",
        "convert_to_model_messages_with",
        "convert_to_model_messages_with_tooling",
        "safe_validate_ui_messages",
        "safe_validate_ui_messages_with_schemas",
        "validate_ui_messages",
        "validate_ui_messages_with_schemas",
    ] {
        assert!(
            source_identifiers(&ui_rs).contains(stable_ui_name),
            "siumai::ui should explicitly export stable UI conversion helper `{stable_ui_name}`"
        );
    }

    assert!(
        public_surface_doc.contains("use siumai::ui::*;")
            && public_surface_doc.contains("UI message validation and conversion helpers"),
        "public-surface.md should document the explicit UI conversion module path"
    );
}

#[test]
fn stable_unified_prelude_does_not_export_middleware_internals() {
    let lib_rs = read_source("src/lib.rs");
    let experimental_rs = experimental_source();
    let prelude_rs = prelude_source();
    let unified_source = prelude_unified_source(&prelude_rs);
    let public_surface_doc =
        fs::read_to_string(crate_root().join("../docs/architecture/public-surface.md"))
            .expect("read public surface doc");

    for internal_name in [
        "LanguageModelMiddleware",
        "MiddlewareBuilder",
        "NamedMiddleware",
    ] {
        assert!(
            !source_identifiers(unified_source).contains(internal_name),
            "prelude::unified should not export execution middleware implementation type `{internal_name}`"
        );
    }

    assert!(
        lib_rs.contains("pub mod experimental;")
            && experimental_rs.contains("pub mod execution {")
            && experimental_rs.contains("pub use siumai_core::execution::*;")
            && experimental_rs.contains("pub mod client {")
            && experimental_rs
                .contains("pub use crate::compat::client::{ClientWrapper, LlmClient};"),
        "siumai::experimental::execution should remain the explicit advanced facade path for middleware internals"
    );
    assert!(
        public_surface_doc.contains("Execution middleware is also an advanced integration API")
            && public_surface_doc
                .contains("siumai::experimental::execution::middleware::LanguageModelMiddleware"),
        "public-surface.md should document that middleware imports live under experimental execution"
    );
}

#[test]
fn facade_root_and_experimental_exports_are_owner_backed_and_scoped() {
    let lib_rs = read_source("src/lib.rs");
    let experimental_rs = experimental_source();
    let public_surface_doc =
        fs::read_to_string(crate_root().join("../docs/architecture/public-surface.md"))
            .expect("read public surface doc");
    let migration_doc =
        fs::read_to_string(crate_root().join("../docs/migration/migration-0.11.0-beta.7.md"))
            .expect("read migration doc");

    assert!(
        lib_rs.contains("pub use siumai_provider_utils::{")
            && lib_rs.contains("pub use siumai_provider_utils::standards::{ToolNameMapping, create_tool_name_mapping};"),
        "facade root utility helpers should be backed by the provider-utils owner after FCAB-120"
    );
    let root_utility_exports_source = lib_rs[lib_rs
        .find("/// AI SDK-style utility helpers.")
        .expect("root utility helper exports comment")..]
        .split("/// Protocol mapping facade")
        .next()
        .expect("root utility helper exports section");
    let root_core_utils_reexports = root_utility_exports_source
        .lines()
        .map(str::trim)
        .filter(|line| line.starts_with("pub use siumai_core::utils::"))
        .filter(|line| !line.starts_with("pub use siumai_core::utils::*"))
        .collect::<Vec<_>>();
    assert_eq!(
        root_core_utils_reexports,
        vec!["pub use siumai_core::utils::{delay, is_abort_error};"],
        "facade root should not teach old broad core-owned utility paths; delay/is_abort_error are the only audited core-runtime helper exception until their CancelHandle coupling is split"
    );
    assert!(
        !lib_rs.contains(
            "pub use siumai_core::standards::{ToolNameMapping, create_tool_name_mapping};"
        ),
        "facade root should not teach old broad core-owned standard-helper paths"
    );
    assert!(
        !lib_rs.contains(
            "pub use siumai_core::{defaults, execution, observability, params, retry, utils};"
        ) && !experimental_rs.contains(
            "pub use siumai_core::{defaults, execution, observability, params, retry, utils};"
        ),
        "experimental facade should not use a broad grouped core-module mirror"
    );
    let experimental_utils_source = experimental_rs[experimental_rs
        .find("/// Core runtime and provider-utils compatibility utility modules.")
        .expect("experimental utils comment")..]
        .trim();
    assert!(
        experimental_utils_source.contains("pub use siumai_provider_utils::*;")
            && experimental_utils_source.contains("StreamingToolCallTracker")
            && experimental_utils_source.contains("pub use siumai_core::utils::{"),
        "experimental::utils should compose provider-utils helpers with explicit core-owned cancel/compat helpers"
    );
    assert!(
        !experimental_utils_source.contains("pub use siumai_core::utils::*;"),
        "experimental::utils must not mirror the whole siumai-core::utils module"
    );

    for module in [
        "defaults",
        "execution",
        "observability",
        "params",
        "retry",
        "utils",
    ] {
        let module_decl = format!("pub mod {module} {{");
        assert!(
            experimental_rs.contains(&module_decl),
            "experimental::{module} should remain a named advanced module"
        );
    }

    assert!(
        public_surface_doc.contains(
            "Provider-utils helpers that remain at the root are backed by `siumai-provider-utils`"
        ) && public_surface_doc
            .contains("Retained broad exports are limited to explicit namespaces")
            && public_surface_doc.contains("`siumai::protocol::<provider>::*`")
            && public_surface_doc.contains("`siumai::content::{prompt,output,compat}::*`")
            && public_surface_doc.contains("`siumai::prelude::compat::{types,content}::*`"),
        "public-surface.md should justify every retained broad namespace"
    );
    assert!(
        migration_doc.contains("Provider-utils helpers that remain available from the facade root now come from `siumai-provider-utils`")
            && migration_doc.contains("The stable unified prelude keeps only the narrow AI SDK-style helper subset"),
        "migration doc should describe the FCAB-120 provider-utils owner-backed facade tightening"
    );
}

#[test]
fn facade_root_splits_public_namespace_modules() {
    let lib_rs = read_source("src/lib.rs");

    for (module, owner_marker) in [
        (
            "hosted_tools",
            "siumai_protocol_openai::hosted_tools::openai::*",
        ),
        ("protocol", "siumai_protocol_openai::standards::openai::*"),
        ("content", "siumai_core::types::content::prompt::*"),
        ("extensions", "VideoGenerationCapability"),
    ] {
        let source = public_namespace_module_source(module);
        assert!(
            source.contains(owner_marker),
            "siumai::{module} should retain its owner-backed public re-export marker `{owner_marker}`"
        );
    }

    let macros_start = lib_rs
        .find("// Macros moved to a dedicated module for cleanliness")
        .expect("macros module marker");
    for module in [
        "hosted_tools",
        "protocol",
        "content",
        "experimental",
        "extensions",
        "prelude",
    ] {
        let declaration = format!("pub mod {module};");
        assert!(
            lib_rs[..macros_start]
                .lines()
                .map(str::trim)
                .any(|line| line == declaration),
            "facade root should declare `siumai::{module}` before private modules instead of hiding it inside another module"
        );
    }
}

#[test]
fn stable_unified_prelude_keeps_only_audited_compatibility_and_runtime_aliases() {
    let prelude_rs = prelude_source();
    let unified_source = prelude_unified_source(&prelude_rs);
    let compatibility_audit = fs::read_to_string(crate_root().join(
        "../docs/workstreams/fearless-spec-core-boundary-convergence/compatibility-audit.md",
    ))
    .expect("read fearless compatibility audit");

    let expected_audited_identifiers: BTreeSet<String> =
        ["CancelHandle"].into_iter().map(str::to_owned).collect();

    let actual_audited_identifiers: BTreeSet<String> = source_identifiers(unified_source)
        .into_iter()
        .filter(|identifier| is_audited_unified_identifier(identifier))
        .collect();

    assert_eq!(
        actual_audited_identifiers, expected_audited_identifiers,
        "new compatibility, experimental, or runtime-bridge names in prelude::unified must be classified in the compatibility audit before they are exported"
    );

    for identifier in &expected_audited_identifiers {
        let audited_identifier = format!("`{identifier}`");
        assert!(
            compatibility_audit.contains(&audited_identifier),
            "prelude::unified keeps `{identifier}` but compatibility-audit.md does not explain it"
        );
    }
}

#[test]
fn broad_facade_types_path_is_explicit_compat_only() {
    let lib_rs = read_source("src/lib.rs");
    let compat_rs = read_source("src/compat.rs");
    let prelude_rs = prelude_source();
    let unified_source = prelude_unified_source(&prelude_rs);
    let root_types_path = ["siumai", "types"].join("::");
    let root_types_glob = format!("`{root_types_path}::*`");
    let compat_types_glob = "`siumai::compat::types::*`";
    let prelude_compat_types_glob = "`siumai::prelude::compat::types::*`";
    let public_surface_doc =
        fs::read_to_string(crate_root().join("../docs/architecture/public-surface.md"))
            .expect("read public surface doc");
    let migration_doc =
        fs::read_to_string(crate_root().join("../docs/migration/migration-0.11.0-beta.7.md"))
            .expect("read migration doc");
    let compatibility_audit = fs::read_to_string(crate_root().join(
        "../docs/workstreams/fearless-spec-core-boundary-convergence/compatibility-audit.md",
    ))
    .expect("read fearless compatibility audit");

    assert!(
        !lib_rs.lines().any(|line| line.starts_with("pub mod types")),
        "facade root should not reintroduce the broad root type namespace"
    );
    assert!(
        !lib_rs.contains("pub use siumai_core::types::*;"),
        "facade root and stable preludes should not mirror the broad core type namespace"
    );
    assert!(
        compat_rs.contains("pub mod types {")
            && compat_rs.contains("pub use siumai_core::types::*;"),
        "siumai::compat should own the broad type namespace for migration-only imports"
    );
    assert!(
        prelude_rs.contains("pub mod types {")
            && prelude_rs.contains("pub use crate::compat::types::*;"),
        "prelude::compat should expose compat::types without restoring a root facade type module"
    );
    assert!(
        !unified_source.contains("siumai_core::types::*"),
        "prelude::unified should stay a curated type surface and must not mirror the broad historical type path"
    );
    assert!(
        public_surface_doc.contains(&root_types_glob)
            && public_surface_doc.contains("removed historical compatibility path")
            && public_surface_doc.contains(compat_types_glob)
            && public_surface_doc.contains(prelude_compat_types_glob)
            && public_surface_doc.contains("curated explicit list"),
        "public-surface.md should document the root type removal and the explicit compat migration path"
    );
    assert!(
        migration_doc.contains("Root broad type namespace")
            && migration_doc.contains(&root_types_glob)
            && migration_doc.contains(compat_types_glob),
        "migration docs should include the root type namespace removal"
    );
    assert!(
        compatibility_audit.contains("Facade broad type path")
            && compatibility_audit.contains(&root_types_glob)
            && compatibility_audit.contains(compat_types_glob)
            && compatibility_audit.contains("removed from the facade root"),
        "compatibility-audit.md should classify broad type imports as explicit compat-only"
    );

    let forbidden_path = format!("{root_types_path}::");
    let forbidden_use = format!("use {root_types_path}");
    for relative_dir in ["tests", "examples"] {
        for path in rust_sources_under(relative_dir) {
            let relative_path = normalized_workspace_path(&crate_root(), &path);
            if relative_path.ends_with("tests/facade_architecture_boundary_test.rs") {
                continue;
            }
            let source = fs::read_to_string(&path).expect("read facade test/example source");
            assert!(
                !source.contains(&forbidden_path) && !source.contains(&forbidden_use),
                "{relative_path} should import stable types from prelude::unified, extension modules, provider extensions, or explicit compat::types"
            );
        }
    }
}

#[test]
fn root_provider_builder_entry_is_compatibility_classified() {
    let lib_rs = read_source("src/lib.rs");
    let compat_rs = read_source("src/compat.rs");
    let compat_provider_rs = read_source("src/compat/provider.rs");
    let public_surface_doc =
        fs::read_to_string(crate_root().join("../docs/architecture/public-surface.md"))
            .expect("read public surface doc");
    let migration_doc =
        fs::read_to_string(crate_root().join("../docs/migration/migration-0.11.0-beta.7.md"))
            .expect("read migration doc");
    let compatibility_audit = fs::read_to_string(crate_root().join(
        "../docs/workstreams/fearless-spec-core-boundary-convergence/compatibility-audit.md",
    ))
    .expect("read fearless compatibility audit");

    assert!(
        compat_rs.contains("pub use provider::Provider;")
            && compat_provider_rs.contains("pub struct Provider;")
            && compat_provider_rs.contains("impl Provider"),
        "siumai::compat should own the Provider builder construction implementation and explicit compatibility import path"
    );
    assert!(
        !lib_rs.contains("pub use compat::Provider;")
            && !lib_rs.contains("pub struct Provider;")
            && !lib_rs.contains("impl Provider {"),
        "facade root should not export Provider; builder-era construction belongs under siumai::compat::Provider"
    );
    assert!(
        compat_rs.contains("pub use siumai_registry::provider::Siumai;")
            && compat_rs.contains("pub use siumai_registry::provider::SiumaiBuilder;")
            && !compat_rs.contains("pub use crate::provider::Siumai;")
            && !compat_provider_rs.contains("crate::provider::SiumaiBuilder"),
        "compat builder-era imports should bind directly to registry-owned types instead of routing through the facade provider shim"
    );
    assert!(
        !lib_rs.contains("pub mod builder {")
            && !lib_rs.contains("pub use siumai_core::builder::*;")
            && !compat_provider_rs.contains("crate::builder::"),
        "facade root should not expose the legacy core builder module; compat/provider should bind to core builder internals directly"
    );
    assert!(
        compat_rs.contains("pub mod builder {")
            && compat_rs.contains("pub use siumai_core::builder::*;"),
        "siumai::compat::builder should remain the explicit migration path for legacy builder base types"
    );
    assert!(
        !crate_root().join("src/provider/mod.rs").exists(),
        "the historical siumai::provider facade shim should be removed; use siumai::compat or registry paths"
    );
    assert!(
        !lib_rs.contains("pub mod provider;") && !lib_rs.contains("mod provider;"),
        "facade root should not declare the removed siumai::provider shim"
    );
    for path in rust_sources_under("src") {
        let relative_path = path
            .strip_prefix(crate_root())
            .expect("source path under crate root")
            .to_string_lossy()
            .replace('\\', "/");

        let source = fs::read_to_string(&path).expect("read facade source file");
        assert!(
            !source.contains("crate::provider::Siumai")
                && !source.contains("crate::provider::SiumaiBuilder"),
            "{relative_path} should use compat or registry-owned builder types instead of the facade provider shim"
        );
    }
    for relative_dir in ["tests", "examples"] {
        for path in rust_sources_under(relative_dir) {
            let relative_path = path
                .strip_prefix(crate_root())
                .expect("source path under crate root")
                .to_string_lossy()
                .replace('\\', "/");

            if relative_path == "tests/facade_architecture_boundary_test.rs" {
                continue;
            }

            let source = fs::read_to_string(&path).expect("read facade test/example source file");
            assert!(
                !source.contains("use siumai::provider::")
                    && !source.contains("siumai::provider::Siumai")
                    && !source.contains("siumai::provider::SiumaiBuilder"),
                "{relative_path} should use siumai::compat or stable registry paths instead of the historical siumai::provider shim"
            );

            if relative_dir == "tests" {
                assert!(
                    !source.contains("use siumai::Provider")
                        && !source.contains("siumai::Provider::"),
                    "{relative_path} should use siumai::compat::Provider or stable registry paths instead of the removed root siumai::Provider alias"
                );
            }

            if relative_dir == "examples" {
                assert!(
                    !source.contains("use siumai::Provider")
                        && !source.contains("siumai::Provider::"),
                    "{relative_path} should not teach the removed root siumai::Provider alias"
                );
            }
        }
    }
    assert!(
        compat_rs.contains("StreamingToolCallDelta")
            && compat_rs.contains("StreamingToolCallFunctionDelta")
            && compat_rs.contains("StreamingToolCallTracker")
            && compat_rs.contains("StreamingToolCallTrackerOptions")
            && compat_rs.contains("StreamingToolCallTypeValidation"),
        "siumai::compat should own the explicit compatibility import path for legacy streaming tool-call helpers"
    );
    assert!(
        !lib_rs.contains("siumai::Provider") && lib_rs.contains("siumai::compat"),
        "facade root docs should not describe a removed root Provider alias"
    );
    assert!(
        public_surface_doc
            .contains("Provider-specific builder construction is also compatibility-oriented")
            && public_surface_doc.contains("use siumai::compat::Provider;")
            && public_surface_doc.contains("root `siumai::Provider` path has been removed")
            && public_surface_doc.contains("root `siumai::provider::*` shim has been removed"),
        "public-surface.md should steer builder imports through explicit compatibility paths and document root removals"
    );
    assert!(
        public_surface_doc.contains("root `siumai::builder::*` shim has been removed")
            && public_surface_doc.contains("siumai::compat::builder"),
        "public-surface.md should document the removed root builder shim and explicit compat builder path"
    );
    assert!(
        migration_doc.contains("Provider builder entry")
            && migration_doc.contains("root `siumai::Provider` alias was removed")
            && migration_doc.contains("root `siumai::provider::*` shim was removed"),
        "migration docs should classify both removed root builder-era facade paths"
    );
    assert!(
        migration_doc.contains("root")
            && migration_doc.contains("`siumai::builder::*` shim was removed")
            && migration_doc.contains("siumai::compat::builder"),
        "migration docs should classify the removed root builder shim"
    );
    assert!(
        compatibility_audit.contains("### Facade provider builder compatibility entry")
            && compatibility_audit.contains("removed the root")
            && compatibility_audit.contains("`siumai::Provider` re-export")
            && compatibility_audit.contains("`siumai::compat::Provider`"),
        "compatibility-audit.md should explain that Provider builder construction is explicit compat-only"
    );
    assert!(
        compatibility_audit.contains("`siumai::builder::*`")
            && compatibility_audit.contains("removed the root `siumai::builder::*` shim")
            && compatibility_audit.contains("`siumai::compat::builder::*`"),
        "compatibility-audit.md should classify legacy builder base types as compat-only"
    );
    assert!(
        compatibility_audit.contains("`siumai::provider::*`")
            && compatibility_audit.contains("removed the root")
            && compatibility_audit.contains("`siumai::provider::*` shim")
            && compatibility_audit.contains("registry-owned"),
        "compatibility-audit.md should classify siumai::provider as removed builder-era facade surface"
    );
}

#[test]
fn provider_extension_builder_helpers_do_not_route_through_provider_shim() {
    let provider_ext_dir = crate_root().join("src/provider_ext");
    let mut checked_files = Vec::new();

    for entry in fs::read_dir(provider_ext_dir).expect("read provider_ext directory") {
        let entry = entry.expect("provider_ext entry");
        let path = entry.path();
        if path.extension().and_then(|extension| extension.to_str()) != Some("rs") {
            continue;
        }

        let source = fs::read_to_string(&path).expect("read provider_ext source");
        let file_name = path
            .file_name()
            .and_then(|name| name.to_str())
            .expect("provider_ext file name")
            .to_owned();

        assert!(
            !source.contains("crate::provider::SiumaiBuilder"),
            "provider_ext/{file_name} should return registry-owned SiumaiBuilder directly instead of routing through the facade provider shim"
        );
        assert!(
            !source.contains("crate::Provider::"),
            "provider_ext/{file_name} should not call the facade root Provider alias; use provider-owned builders or the explicit compat Provider path"
        );

        if source.contains("-> SiumaiBuilder") {
            assert!(
                source.contains("siumai_registry::provider::SiumaiBuilder"),
                "provider_ext/{file_name} returns SiumaiBuilder but does not import the registry-owned type"
            );
        }

        checked_files.push(file_name);
    }

    assert!(
        checked_files.iter().any(|file| file == "azure.rs")
            && checked_files.iter().any(|file| file == "togetherai.rs")
            && checked_files.iter().any(|file| file == "vertex_maas.rs"),
        "provider_ext source guard should cover provider package builder helpers"
    );
}

#[test]
fn streaming_tool_call_helpers_are_explicit_compat_only() {
    let lib_rs = read_source("src/lib.rs");
    let compat_rs = read_source("src/compat.rs");
    let public_surface_doc =
        fs::read_to_string(crate_root().join("../docs/architecture/public-surface.md"))
            .expect("read public surface doc");
    let migration_doc =
        fs::read_to_string(crate_root().join("../docs/migration/migration-0.11.0-beta.7.md"))
            .expect("read migration doc");
    let compatibility_audit = fs::read_to_string(crate_root().join(
        "../docs/workstreams/fearless-spec-core-boundary-convergence/compatibility-audit.md",
    ))
    .expect("read fearless compatibility audit");

    for helper in [
        "StreamingToolCallDelta",
        "StreamingToolCallFunctionDelta",
        "StreamingToolCallTracker",
        "StreamingToolCallTrackerOptions",
        "StreamingToolCallTypeValidation",
    ] {
        assert!(
            !lib_rs
                .lines()
                .take_while(|line| !line.contains("/// Protocol mapping facade"))
                .any(|line| line.contains(helper)),
            "facade root should not export `{helper}` directly; use siumai::compat or prelude::compat"
        );
        assert!(
            compat_rs.contains(helper),
            "siumai::compat should keep the explicit migration import for `{helper}`"
        );
        assert!(
            public_surface_doc.contains(helper),
            "public-surface.md should classify `{helper}` as compat-only"
        );
    }

    assert!(
        compat_rs.contains("pub use siumai_core::utils::{"),
        "compat should re-export streaming tool-call helpers from their implementation owner without routing through facade root aliases"
    );
    assert!(
        public_surface_doc.contains("StreamingToolCall*` helpers remain available from")
            && public_surface_doc.contains("They are no longer re-exported from the facade root"),
        "public-surface.md should tell users the root aliases were removed"
    );
    assert!(
        migration_doc.contains("StreamingToolCall* helpers")
            && migration_doc.contains("use siumai::compat::{")
            && migration_doc.contains("StreamingToolCallTracker"),
        "migration docs should show the explicit compat import path for streaming tool-call helpers"
    );
    assert!(
        compatibility_audit.contains("Root aliases were removed during the Track F facade cleanup")
            && compatibility_audit.contains("only explicit compat re-exports"),
        "compatibility-audit.md should record the root alias removal"
    );
}

#[test]
fn stable_registry_prelude_exports_factory_signature_types() {
    let lib_rs = read_source("src/lib.rs");
    let prelude_rs = prelude_source();
    let unified_source = prelude_unified_source(&prelude_rs);
    let compat_prelude_source = prelude_rs[prelude_rs
        .find("pub mod compat {")
        .expect("compat prelude module")..]
        .split("pub mod extensions")
        .next()
        .expect("compat prelude tail before extensions");
    let compatibility_audit = fs::read_to_string(crate_root().join(
        "../docs/workstreams/fearless-spec-core-boundary-convergence/compatibility-audit.md",
    ))
    .expect("read fearless compatibility audit");
    let migration_doc =
        fs::read_to_string(crate_root().join("../docs/migration/migration-0.11.0-beta.7.md"))
            .expect("read migration doc");
    let public_surface_doc =
        fs::read_to_string(crate_root().join("../docs/architecture/public-surface.md"))
            .expect("read public surface doc");

    assert!(
        !lib_rs.contains("registry_global")
            && !lib_rs.contains("pub use registry::global as")
            && !unified_source.contains("registry_global"),
        "facade should not keep a root registry_global alias; use registry::global() or prelude::unified::registry::global()"
    );
    assert!(
        !lib_rs.contains("pub mod provider_catalog;")
            && !crate_root().join("src/provider_catalog.rs").exists(),
        "facade root should not mirror siumai-registry provider_catalog; import the registry-owned catalog explicitly"
    );
    assert!(
        !unified_source.contains("pub use crate::registry::ProviderFactory;"),
        "prelude::unified should not export ProviderFactory at the top level; use prelude::unified::registry::*"
    );
    assert!(
        !compat_prelude_source.contains("pub mod registry {")
            && !compat_prelude_source.contains("pub use super::unified::registry::*;"),
        "facade should not keep a historical prelude::registry mirror; use prelude::unified::registry::* or siumai::registry::*"
    );
    assert!(
        unified_source.contains("pub mod registry {")
            && unified_source.contains("ProviderFactory")
            && unified_source.contains("BuildContext")
            && unified_source.contains("ProviderBuildOverrides"),
        "prelude::unified::registry should export ProviderFactory plus the context types required by family-first factory method signatures"
    );
    assert!(
        public_surface_doc
            .contains("`siumai::prelude::unified::registry::*` includes `BuildContext`")
            && public_surface_doc.contains("custom factory implementations"),
        "public-surface.md should document that custom factory signature types are available from the stable registry surface"
    );
    assert!(
        public_surface_doc.contains("`siumai::registry_global` alias has been removed")
            && migration_doc.contains("`siumai::registry_global` alias")
            && migration_doc.contains("removed"),
        "docs should steer users from registry_global to the scoped registry global handle"
    );
    assert!(
        public_surface_doc.contains("`siumai::prelude::registry::*` mirror has been removed")
            && migration_doc.contains("`siumai::prelude::registry::*` mirror")
            && migration_doc.contains("removed"),
        "docs should steer users from the historical prelude registry mirror to prelude::unified::registry"
    );
    assert!(
        public_surface_doc.contains("root `siumai::provider_catalog::*` mirror has been removed")
            && migration_doc.contains("root `siumai::provider_catalog::*` mirror")
            && migration_doc.contains("removed")
            && compatibility_audit.contains("removed root")
            && compatibility_audit.contains("`siumai::provider_catalog::*` mirror"),
        "docs should steer provider catalog users to the registry-owned provider catalog"
    );
}

#[test]
fn stable_unified_prelude_scopes_non_family_upload_helpers() {
    let lib_rs = read_source("src/lib.rs");
    let prelude_rs = prelude_source();
    let unified_source = prelude_unified_source(&prelude_rs);
    let public_surface_doc =
        fs::read_to_string(crate_root().join("../docs/architecture/public-surface.md"))
            .expect("read public surface doc");

    for forbidden in [
        "pub use crate::files::{",
        "pub use crate::skills::{",
        "upload_file, upload_skill",
    ] {
        assert!(
            !unified_source.contains(forbidden),
            "prelude::unified should not directly export non-family upload helper surface `{forbidden}`"
        );
    }

    assert!(
        unified_source.contains("files")
            && unified_source.contains("skills")
            && lib_rs.contains("pub async fn upload_file")
            && lib_rs.contains("pub async fn upload_skill"),
        "explicit upload helper modules and root helpers should remain available while top-level unified stays family-focused"
    );
    assert!(
        public_surface_doc.contains("File and skill upload helpers are stable explicit modules")
            && public_surface_doc.contains("siumai::files::*")
            && public_surface_doc.contains("siumai::skills::*"),
        "public-surface.md should document explicit file/skill upload helper paths"
    );
}

#[test]
fn stable_unified_prelude_keeps_non_family_extension_types_scoped() {
    let lib_rs = read_source("src/lib.rs");
    let extensions_rs = public_namespace_module_source("extensions");
    let prelude_rs = prelude_source();
    let unified_source = prelude_unified_source(&prelude_rs);
    let public_surface_doc =
        fs::read_to_string(crate_root().join("../docs/architecture/public-surface.md"))
            .expect("read public surface doc");

    assert!(
        !unified_source.contains("pub use crate::extensions::*;"),
        "prelude::unified should not mirror the whole non-family extensions module"
    );

    for extension_only_name in [
        "FileManagementCapability",
        "ModelListingCapability",
        "ModerationCapability",
        "MusicGenerationCapability",
        "SkillsCapability",
        "ImageExtras",
        "SpeechExtras",
        "TranscriptionExtras",
        "VideoGenerationCapability",
        "FileDeleteResponse",
        "FileListQuery",
        "FileListResponse",
        "FileObject",
        "FileUploadRequest",
        "ImageEditInput",
        "ImageEditRequest",
        "ImageVariationRequest",
        "ModerationRequest",
        "ModerationResponse",
        "SkillFileContent",
        "SkillProviderMetadata",
        "SkillUploadFile",
        "SkillUploadRequest",
        "SkillUploadResult",
        "VideoGenerationInput",
        "VideoGenerationRequest",
        "VideoGenerationResponse",
        "VideoTaskStatus",
        "VideoTaskStatusResponse",
    ] {
        assert!(
            !source_identifiers(unified_source).contains(extension_only_name),
            "prelude::unified should not directly export non-family extension type `{extension_only_name}`; use siumai::extensions or prelude::extensions"
        );
    }

    assert!(
        lib_rs.contains("pub mod extensions;")
            && extensions_rs.contains("pub use siumai_core::traits::{")
            && extensions_rs.contains("pub mod types")
            && prelude_rs.contains("pub use crate::extensions::*;")
            && public_surface_doc.contains("use siumai::extensions::*;")
            && public_surface_doc.contains("use siumai::extensions::types::*;")
            && public_surface_doc.contains("siumai::prelude::extensions::*"),
        "facade docs and prelude should keep non-family extension imports on explicit extension paths"
    );
}

#[test]
fn family_taxonomy_documents_video_as_stable_and_music_as_extension_only() {
    let lib_rs = read_source("src/lib.rs");
    let prelude_rs = prelude_source();
    let unified_source = prelude_unified_source(&prelude_rs);
    let public_surface_doc =
        fs::read_to_string(crate_root().join("../docs/architecture/public-surface.md"))
            .expect("read public surface doc");
    let capability_surface_doc =
        fs::read_to_string(crate_root().join("../docs/architecture/capability-surface.md"))
            .expect("read capability surface doc");
    let module_split_doc =
        fs::read_to_string(crate_root().join("../docs/architecture/module-split-design.md"))
            .expect("read module split doc");
    let adr_family_policy = fs::read_to_string(
        crate_root().join("../docs/adr/0006-family-model-first-trait-policy.md"),
    )
    .expect("read family policy ADR");
    let migration_doc =
        fs::read_to_string(crate_root().join("../docs/migration/migration-0.11.0-beta.7.md"))
            .expect("read migration doc");
    let registry_entry =
        fs::read_to_string(crate_root().join("../siumai-registry/src/registry/entry.rs"))
            .expect("read registry entry source");
    let registry_handles_mod = fs::read_to_string(
        crate_root().join("../siumai-registry/src/registry/entry/handles/mod.rs"),
    )
    .expect("read registry handles mod source");
    let language_handle = fs::read_to_string(
        crate_root().join("../siumai-registry/src/registry/entry/handles/language.rs"),
    )
    .expect("read registry language handle source");
    let video_handle = fs::read_to_string(
        crate_root().join("../siumai-registry/src/registry/entry/handles/video.rs"),
    )
    .expect("read registry video handle source");
    let core_video = fs::read_to_string(crate_root().join("../siumai-core/src/video.rs"))
        .expect("read core video source");
    let core_music_trait =
        fs::read_to_string(crate_root().join("../siumai-core/src/traits/music.rs"))
            .expect("read core music trait source");

    assert!(
        prelude_rs.contains("seven stable model families")
            && prelude_rs.contains("Language/Embedding/Image/Reranking/Speech/Transcription/Video")
            && lib_rs.contains("Music remains extension-only"),
        "facade docs/comments should name seven stable families and keep music extension-only"
    );
    for stable_video_name in [
        "video",
        "VideoModel",
        "VideoModelV4",
        "GenerateVideoResult",
        "VideoModelProviderMetadata",
        "VideoModelHandle",
    ] {
        assert!(
            source_identifiers(unified_source).contains(stable_video_name),
            "prelude::unified should expose stable video family name `{stable_video_name}`"
        );
    }
    for extension_only_name in ["MusicGenerationCapability", "MusicGenerationRequest"] {
        assert!(
            !source_identifiers(unified_source).contains(extension_only_name),
            "prelude::unified should not expose music extension-only name `{extension_only_name}`"
        );
    }

    assert!(
        public_surface_doc.contains("7 stable model families")
            && public_surface_doc.contains(
                "Language / Embedding / Image / Rerank / Speech (TTS) / Transcription (STT) / Video"
            )
            && public_surface_doc.contains("Music generation remains extension-only"),
        "public-surface.md should document Video as the seventh stable family and Music as extension-only"
    );
    assert!(
        capability_surface_doc.contains("seven model families")
            && capability_surface_doc.contains("7. Video generation")
            && capability_surface_doc.contains("Music generation remains extension-only")
            && capability_surface_doc.contains("no stable")
            && capability_surface_doc.contains("`MusicModel`")
            && capability_surface_doc.contains("`music_model(...)`"),
        "capability-surface.md should make the stable/extension taxonomy explicit"
    );
    assert!(
        module_split_doc.contains("request/response types for the 7 model families")
            && adr_family_policy.contains("`video` are the preferred public families")
            && adr_family_policy.contains("Amendment — 2026-05-21 (FCAB-130)")
            && adr_family_policy.contains("Music remains extension-only")
            && migration_doc.contains("Video is part of the stable family list")
            && migration_doc.contains("music remains extension-only"),
        "architecture, ADR, and migration docs should agree on the seven-family taxonomy"
    );

    assert!(
        core_video.contains("Stable Rust interface for task-oriented video generation models")
            && core_video.contains("pub trait VideoModel")
            && core_video.contains("pub trait VideoModelV4")
            && core_video.contains("impl<T> VideoModel for T"),
        "siumai-core should own a first-class stable VideoModel family contract"
    );
    assert!(
        core_music_trait.contains("pub trait MusicGenerationCapability")
            && !core_music_trait.contains("pub trait MusicModel")
            && !core_music_trait.contains("MusicModelV4"),
        "music should remain a capability trait, not a stable family model"
    );
    assert!(
        registry_entry.contains("pub fn video_model(&self")
            && registry_entry.contains("VideoModelHandle")
            && !registry_entry.contains("pub fn music_model(&self"),
        "registry should expose a stable video_model handle and no music_model handle"
    );
    assert!(
        registry_handles_mod.contains("pub use video::VideoModelHandle;")
            && !registry_handles_mod.contains("MusicModelHandle"),
        "registry handles should export VideoModelHandle only; no MusicModelHandle should exist"
    );
    assert!(
        video_handle.contains("impl VideoGenerationCapability for VideoModelHandle")
            && video_handle.contains("impl crate::traits::ModelMetadata for VideoModelHandle"),
        "VideoModelHandle should be the stable video-family handle and metadata carrier"
    );
    assert!(
        language_handle.contains("impl MusicGenerationCapability for LanguageModelHandle")
            && language_handle.contains("build_music_generation_capability_with_ctx")
            && !language_handle.contains("MusicModelHandle"),
        "music should stay extension-only through language-handle compatibility delegation"
    );
}

#[test]
fn legacy_core_root_modules_do_not_return_to_facade() {
    let lib_rs = read_source("src/lib.rs");
    let public_surface_doc =
        fs::read_to_string(crate_root().join("../docs/architecture/public-surface.md"))
            .expect("read public surface doc");

    for legacy_path in [
        "`siumai::traits::*`",
        "`siumai::error::*`",
        "`siumai::streaming::*`",
    ] {
        assert!(
            public_surface_doc.contains(legacy_path),
            "public-surface.md should keep `{legacy_path}` classified outside the stable facade"
        );
    }

    for forbidden_root_module in ["error", "traits", "streaming"] {
        let module_declaration = format!("pub mod {forbidden_root_module}");
        assert!(
            !lib_rs
                .lines()
                .any(|line| line.starts_with(&module_declaration)),
            "facade root should not reintroduce the legacy `{forbidden_root_module}` module"
        );

        let broad_reexport = format!("pub use siumai_core::{forbidden_root_module}::*;");
        assert!(
            !lib_rs.lines().any(|line| line == broad_reexport),
            "facade root should not broadly re-export `siumai_core::{forbidden_root_module}`"
        );
    }
}

#[test]
fn hosted_tools_facade_reexports_protocol_owned_constructors() {
    let lib_rs = read_source("src/lib.rs");
    let hosted_tools_rs = public_namespace_module_source("hosted_tools");
    let public_surface_doc =
        fs::read_to_string(crate_root().join("../docs/architecture/public-surface.md"))
            .expect("read public surface doc");

    assert!(
        !lib_rs.contains("pub use siumai_core::hosted_tools"),
        "facade hosted_tools should not re-export provider-specific constructors from siumai-core"
    );
    let tools_rs = read_source("src/tools.rs");
    assert!(
        !tools_rs.contains("siumai_core::tools"),
        "facade tools compatibility surface should delegate to protocol/provider-owned catalogs, not siumai-core"
    );

    for expected in [
        "siumai_protocol_openai::hosted_tools::openai::*",
        "siumai_protocol_anthropic::hosted_tools::anthropic::*",
        "siumai_protocol_gemini::hosted_tools::google::*",
    ] {
        assert!(
            hosted_tools_rs.contains(expected),
            "facade hosted_tools should re-export protocol-owned constructor surface `{expected}`"
        );
    }

    for expected in [
        "siumai_protocol_openai::tool_catalog::openai::*",
        "siumai_protocol_anthropic::tool_catalog::anthropic::*",
        "siumai_protocol_gemini::tool_catalog::google::*",
        "siumai_provider_groq::tools::groq::*",
        "siumai_provider_xai::tools::xai::*",
    ] {
        assert!(
            tools_rs.contains(expected),
            "facade tools compatibility surface should re-export provider-owned catalog `{expected}`"
        );
    }

    assert!(
        public_surface_doc.contains("protocol-owned provider-defined tool constructors")
            && public_surface_doc.contains("core only owns the passive `Tool::ProviderDefined`"),
        "public-surface.md should describe hosted tool ownership"
    );
}

#[test]
fn facade_generic_client_paths_are_explicit_compatibility_exports() {
    let lib_rs = read_source("src/lib.rs");
    let experimental_rs = experimental_source();
    let compat_rs = read_source("src/compat.rs");
    let public_surface_doc =
        fs::read_to_string(crate_root().join("../docs/architecture/public-surface.md"))
            .expect("read public surface doc");
    let migration_doc =
        fs::read_to_string(crate_root().join("../docs/migration/migration-0.11.0-beta.7.md"))
            .expect("read migration doc");

    assert!(
        compat_rs.contains("pub mod client")
            && compat_rs
                .contains("pub use siumai_core::compat::client::{ClientWrapper, LlmClient};"),
        "siumai::compat::client should be the explicit facade migration path for generic clients"
    );
    assert!(
        lib_rs.contains("pub mod experimental;")
            && experimental_rs.contains("pub mod client {")
            && experimental_rs
                .contains("pub use crate::compat::client::{ClientWrapper, LlmClient};"),
        "siumai::experimental::client should remain as an advanced alias to compat::client"
    );
    assert!(
        !lib_rs.contains("pub use siumai_core::client::{ClientWrapper, LlmClient};")
            && !lib_rs.contains("pub use siumai_core::client::*;")
            && !experimental_rs
                .contains("pub use siumai_core::client::{ClientWrapper, LlmClient};")
            && !experimental_rs.contains("pub use siumai_core::client::*;"),
        "facade root/experimental code should not point directly at siumai_core::client; route through explicit compat::client"
    );
    assert!(
        public_surface_doc.contains("siumai::compat::client::{LlmClient, ClientWrapper}")
            && migration_doc.contains("siumai::compat::client::{LlmClient, ClientWrapper}"),
        "public and migration docs should name the explicit generic-client compatibility import path"
    );
}
