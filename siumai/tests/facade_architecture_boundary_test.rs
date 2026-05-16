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

fn prelude_unified_source(lib_rs: &str) -> &str {
    let unified_start = lib_rs
        .find("pub mod unified {")
        .expect("unified prelude module");
    let compat_start = lib_rs[unified_start..]
        .find("pub mod compat {")
        .expect("compat prelude module");
    &lib_rs[unified_start..unified_start + compat_start]
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

fn is_low_priority_content_part_audit_path(relative_path: &str) -> bool {
    relative_path.ends_with("/tests.rs")
        || relative_path.contains("/provider_options/")
        || relative_path.contains("/provider_ext/")
        || relative_path.ends_with("/mod.rs")
        || relative_path.ends_with("/builder.rs")
        || relative_path.ends_with("/config.rs")
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
    assert!(
        gemini_rs.contains("pub use models::{chat, embedding, image, model_sets, video};"),
        "Gemini model-id group paths should remain available from provider_ext::gemini"
    );

    for family_module in ["chat", "embedding", "image", "video", "model_sets"] {
        let declaration = format!("pub mod {family_module}");
        assert!(
            gemini_models_rs.contains(&declaration),
            "Gemini model catalog should expose the {family_module} model group"
        );
    }
}

#[test]
fn experimental_bridge_is_owned_by_bridge_crate_and_reexported_by_facade() {
    let lib_rs = read_source("src/lib.rs");
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
        lib_rs.contains("pub use siumai_bridge::*;"),
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
fn generate_text_projection_delegates_content_part_mapping_to_spec() {
    let text_rs = read_source("src/text.rs");
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
                || is_low_priority_content_part_audit_path(&relative_path)
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
        "Track C audit must classify every high-value production source that directly constructs ContentPart or provider maps before the path is accepted: {missing:#?}"
    );
}

#[test]
fn stable_unified_prelude_excludes_compatibility_construction_aliases() {
    let lib_rs = read_source("src/lib.rs");
    let unified_source = prelude_unified_source(&lib_rs);

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
        lib_rs.contains("pub mod compat {")
            && lib_rs.contains("pub use crate::compat::{")
            && lib_rs.contains("StreamingToolCallTracker")
            && lib_rs.contains("Experimental_GenerateImageResult")
            && lib_rs.contains("step_count_is"),
        "compatibility construction and legacy helper aliases should remain explicit under prelude::compat"
    );
}

#[test]
fn stable_unified_prelude_does_not_mirror_core_streaming_internals() {
    let lib_rs = read_source("src/lib.rs");
    let unified_source = prelude_unified_source(&lib_rs);
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
        lib_rs.contains("pub mod streaming {")
            && lib_rs.contains("pub use siumai_core::streaming::*;"),
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
    let unified_source = prelude_unified_source(&lib_rs);
    let public_surface_doc =
        fs::read_to_string(crate_root().join("../docs/architecture/public-surface.md"))
            .expect("read public surface doc");

    for utility_name in [
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
    ] {
        assert!(
            !source_identifiers(unified_source).contains(utility_name),
            "prelude::unified should not export low-level utility helper `{utility_name}`"
        );
        assert!(
            source_identifiers(&lib_rs).contains(utility_name),
            "the explicit facade root should still export `{utility_name}` for opt-in utility users"
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
            && public_surface_doc.contains("use siumai::{parse_json, normalize_headers};"),
        "public-surface.md should document scoped low-level utility helper imports"
    );
}

#[test]
fn stable_unified_prelude_scopes_retry_api() {
    let lib_rs = read_source("src/lib.rs");
    let unified_source = prelude_unified_source(&lib_rs);
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
fn stable_unified_prelude_does_not_mirror_tooling_runtime_module() {
    let lib_rs = read_source("src/lib.rs");
    let unified_source = prelude_unified_source(&lib_rs);
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
fn stable_unified_prelude_does_not_export_middleware_internals() {
    let lib_rs = read_source("src/lib.rs");
    let unified_source = prelude_unified_source(&lib_rs);
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
        lib_rs.contains("pub use siumai_core::{client, defaults, execution")
            && lib_rs.contains("pub mod experimental {"),
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
fn stable_unified_prelude_keeps_only_audited_compatibility_and_runtime_aliases() {
    let lib_rs = read_source("src/lib.rs");
    let unified_source = prelude_unified_source(&lib_rs);
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
    let unified_source = prelude_unified_source(&lib_rs);
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
        lib_rs.contains("pub mod types {") && lib_rs.contains("pub use crate::compat::types::*;"),
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
    let unified_source = prelude_unified_source(&lib_rs);
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
        lib_rs.contains("pub mod registry {")
            && lib_rs.contains("ProviderFactory")
            && lib_rs.contains("BuildContext")
            && lib_rs.contains("ProviderBuildOverrides"),
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
    let unified_source = prelude_unified_source(&lib_rs);
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
    let unified_source = prelude_unified_source(&lib_rs);
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
        lib_rs.contains("pub mod extensions {")
            && lib_rs.contains("pub use crate::extensions::*;")
            && public_surface_doc.contains("use siumai::extensions::*;")
            && public_surface_doc.contains("use siumai::extensions::types::*;")
            && public_surface_doc.contains("siumai::prelude::extensions::*"),
        "facade docs and prelude should keep non-family extension imports on explicit extension paths"
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
    let public_surface_doc =
        fs::read_to_string(crate_root().join("../docs/architecture/public-surface.md"))
            .expect("read public surface doc");

    assert!(
        !lib_rs.contains("pub use siumai_core::hosted_tools"),
        "facade hosted_tools should not re-export provider-specific constructors from siumai-core"
    );

    for expected in [
        "siumai_protocol_openai::hosted_tools::openai::*",
        "siumai_protocol_anthropic::hosted_tools::anthropic::*",
        "siumai_protocol_gemini::hosted_tools::google::*",
    ] {
        assert!(
            lib_rs.contains(expected),
            "facade hosted_tools should re-export protocol-owned constructor surface `{expected}`"
        );
    }

    assert!(
        public_surface_doc.contains("protocol-owned provider-defined tool constructors")
            && public_surface_doc.contains("core only owns the passive `Tool::ProviderDefined`"),
        "public-surface.md should describe hosted tool ownership"
    );
}
