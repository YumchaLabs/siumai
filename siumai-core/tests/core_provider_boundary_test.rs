use std::fs;
use std::path::{Path, PathBuf};

fn crate_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
}

fn workspace_root() -> PathBuf {
    crate_root()
        .parent()
        .expect("siumai-core should live under workspace root")
        .to_path_buf()
}

fn collect_rust_sources(dir: &Path, sources: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(dir).expect("read source directory") {
        let entry = entry.expect("read directory entry");
        let path = entry.path();

        if path.is_dir() {
            collect_rust_sources(&path, sources);
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            sources.push(path);
        }
    }
}

fn collect_workspace_crate_sources_by_prefix(
    workspace_root: &Path,
    prefixes: &[&str],
    sources: &mut Vec<PathBuf>,
) {
    for entry in fs::read_dir(workspace_root).expect("read workspace root") {
        let entry = entry.expect("read workspace entry");
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }

        let name = entry.file_name();
        let name = name.to_string_lossy();
        if prefixes.iter().any(|prefix| name.starts_with(prefix)) {
            collect_rust_sources(&path, sources);
        }
    }
}

fn normalized_relative_path(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .expect("path should be under crate root")
        .iter()
        .map(|component| component.to_string_lossy())
        .collect::<Vec<_>>()
        .join("/")
}

fn production_non_comment_source(source: &str) -> String {
    source
        .split("\n#[cfg(test)]")
        .next()
        .unwrap_or(source)
        .lines()
        .filter(|line| {
            let trimmed = line.trim_start();
            !trimmed.starts_with("//") && !trimmed.starts_with('*')
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn core_manifest_does_not_depend_on_registry_facade_provider_or_protocol_crates() {
    let manifest = fs::read_to_string(crate_root().join("Cargo.toml")).expect("read Cargo.toml");

    for forbidden_dependency in [
        "siumai-registry",
        "siumai-bridge",
        "siumai-extras",
        "siumai-protocol-",
        "siumai =",
    ] {
        assert!(
            !manifest.contains(forbidden_dependency),
            "siumai-core must stay provider-agnostic and must not depend on `{forbidden_dependency}`"
        );
    }

    let forbidden_provider_impl_dependencies = manifest
        .lines()
        .filter(|line| line.trim_start().starts_with("siumai-provider-"))
        .filter(|line| !line.trim_start().starts_with("siumai-provider-utils"))
        .collect::<Vec<_>>();
    assert!(
        forbidden_provider_impl_dependencies.is_empty(),
        "siumai-core must not depend on provider implementation crates; only the generic `siumai-provider-utils` seam is allowed:\n{}",
        forbidden_provider_impl_dependencies.join("\n")
    );

    assert!(
        manifest.contains("siumai-spec = { workspace = true, default-features = false }"),
        "siumai-core should depend downward on siumai-spec as its data-contract crate"
    );
}

#[test]
fn core_production_source_does_not_import_registry_facade_provider_or_protocol_crates() {
    let manifest_dir = crate_root();
    let src_dir = manifest_dir.join("src");
    let mut rust_sources = Vec::new();
    collect_rust_sources(&src_dir, &mut rust_sources);

    let mut violations = Vec::new();
    for source_path in rust_sources {
        let relative_path = normalized_relative_path(&manifest_dir, &source_path);
        if relative_path.ends_with("/tests.rs") {
            continue;
        }

        let source = fs::read_to_string(&source_path)
            .unwrap_or_else(|error| panic!("read {relative_path}: {error}"));
        let production_source = production_non_comment_source(&source);

        for forbidden in [
            "siumai_registry::",
            "siumai_protocol_",
            "siumai_bridge::",
            "siumai_extras::",
            "use siumai::",
            "siumai::prelude",
            "siumai::registry",
            "siumai::compat",
            "siumai::provider_ext",
            "siumai::providers",
            "siumai::experimental",
            "siumai::Provider",
            "siumai::Siumai",
            "ProviderFactory",
            "RegistryOptions",
            "ProviderBuildOverrides",
            "create_provider_registry",
            "registry::global",
            "Siumai::builder",
            "Provider::openai",
            "Provider::anthropic",
            "Provider::gemini",
            "Provider::google",
            "Provider::azure",
            "Provider::bedrock",
            "Provider::cohere",
            "Provider::deepseek",
            "Provider::groq",
            "Provider::ollama",
            "Provider::xai",
            "Provider::minimaxi",
            "Provider::togetherai",
        ] {
            if production_source.contains(forbidden) {
                violations.push(format!("{relative_path}: `{forbidden}`"));
            }
        }

        let source_without_provider_utils =
            production_source.replace("siumai_provider_utils::", "");
        if source_without_provider_utils.contains("siumai_provider_") {
            violations.push(format!(
                "{relative_path}: `siumai_provider_` (only the generic `siumai_provider_utils` seam is allowed)"
            ));
        }
    }

    assert!(
        violations.is_empty(),
        "siumai-core production code must not import registry, facade, provider, protocol, bridge, or extras crate surfaces:\n{}",
        violations.join("\n")
    );
}

#[test]
fn core_utils_do_not_own_provider_model_aliases() {
    let manifest_dir = crate_root();
    let utils_dir = manifest_dir.join("src").join("utils");

    assert!(
        !utils_dir.join("model_alias.rs").exists(),
        "provider-specific model aliases belong to registry/provider-owned code, not siumai-core"
    );

    let utils_mod = std::fs::read_to_string(utils_dir.join("mod.rs")).expect("read utils/mod.rs");
    assert!(
        !utils_mod.contains("model_alias"),
        "siumai-core::utils must not export a provider model alias module"
    );

    assert!(
        !utils_dir.join("builder_helpers.rs").exists(),
        "provider builder helpers belong to siumai-provider-utils, not siumai-core::utils"
    );
}

#[test]
fn provider_utils_crate_owns_high_churn_provider_helpers_without_core_alias_modules() {
    let root = workspace_root();
    let core_utils_dir = crate_root().join("src").join("utils");
    let provider_utils_dir = root.join("siumai-provider-utils").join("src");

    let moved_modules = [
        "builder_helpers",
        "chat_request",
        "data",
        "download",
        "error_message",
        "headers",
        "id",
        "json_instruction",
        "json_parse",
        "mime",
        "option",
        "provider_options",
        "provider_reference",
        "reasoning",
        "runtime",
        "serial_job",
        "settings",
        "url",
        "utf8_decoder",
        "validate_types",
    ];
    for module in moved_modules {
        let provider_source = fs::read_to_string(provider_utils_dir.join(format!("{module}.rs")))
            .unwrap_or_else(|error| panic!("read provider-utils module {module}: {error}"));

        assert!(
            !provider_source.trim().is_empty(),
            "siumai-provider-utils::{module} should own the helper implementation"
        );
        assert!(
            !core_utils_dir.join(format!("{module}.rs")).exists(),
            "siumai-core::utils::{module} should not keep a shallow provider-utils alias module"
        );
    }

    let core_manifest =
        fs::read_to_string(crate_root().join("Cargo.toml")).expect("read siumai-core Cargo.toml");
    assert!(
        core_manifest
            .contains("siumai-provider-utils = { workspace = true, default-features = false }"),
        "siumai-core should depend on the generic provider-utils seam for runtime call sites that need shared helper behavior"
    );

    let provider_manifest =
        fs::read_to_string(root.join("siumai-provider-utils").join("Cargo.toml"))
            .expect("read siumai-provider-utils Cargo.toml");
    assert!(
        provider_manifest.contains("siumai-spec = { workspace = true, default-features = false }"),
        "siumai-provider-utils should depend on spec contracts, not on siumai-core"
    );
    assert!(
        !provider_manifest.contains("siumai-core"),
        "siumai-provider-utils must not depend on siumai-core; otherwise core compatibility aliases would create a cycle"
    );
}

#[test]
fn core_utils_remaining_owned_modules_are_classified() {
    let utils_dir = crate_root().join("src").join("utils");
    let mut actual_modules = fs::read_dir(&utils_dir)
        .expect("read utils dir")
        .filter_map(|entry| {
            let entry = entry.expect("read utils entry");
            let path = entry.path();
            (path.extension().is_some_and(|extension| extension == "rs")
                && path.file_stem().is_some_and(|stem| stem != "mod"))
            .then(|| {
                path.file_stem()
                    .expect("utils file stem")
                    .to_string_lossy()
                    .to_string()
            })
        })
        .collect::<Vec<_>>();
    actual_modules.sort();

    let mut classified_modules = [
        // Stable core runtime utility: depends on core CancelHandle and stream handles.
        "cancel",
        // Explicit compatibility helper: depends on core stream part types and is facade compat-only.
        "streaming_tool_call",
    ]
    .into_iter()
    .map(str::to_string)
    .collect::<Vec<_>>();
    classified_modules.sort();

    assert_eq!(
        actual_modules, classified_modules,
        "siumai-core::utils should only contain core-owned utilities and explicit compat helpers"
    );
}

#[test]
fn provider_protocol_crates_do_not_import_moved_provider_utils_from_core() {
    let root = workspace_root();
    let moved_modules = [
        "builder_helpers",
        "chat_request",
        "data",
        "download",
        "error_message",
        "headers",
        "id",
        "json_instruction",
        "json_parse",
        "mime",
        "option",
        "provider_options",
        "provider_reference",
        "reasoning",
        "runtime",
        "serial_job",
        "settings",
        "url",
        "utf8_decoder",
        "validate_types",
    ];

    let mut rust_sources = Vec::new();
    collect_workspace_crate_sources_by_prefix(
        &root,
        &["siumai-provider-", "siumai-protocol-", "siumai-registry"],
        &mut rust_sources,
    );

    let mut violations = Vec::new();
    for source_path in rust_sources {
        let relative_path = normalized_relative_path(&root, &source_path);
        let source = fs::read_to_string(&source_path)
            .unwrap_or_else(|error| panic!("read {relative_path}: {error}"));
        let production_source = production_non_comment_source(&source);

        for module in moved_modules {
            for forbidden in [
                format!("siumai_core::utils::{module}"),
                format!("crate::utils::{module}"),
            ] {
                if production_source.contains(&forbidden) {
                    violations.push(format!("{relative_path}: `{forbidden}`"));
                }
            }
        }

        for forbidden in ["crate::utils::guess_mime", "siumai_core::utils::guess_mime"] {
            if production_source.contains(forbidden) {
                violations.push(format!("{relative_path}: `{forbidden}`"));
            }
        }
    }

    assert!(
        violations.is_empty(),
        "provider, protocol, and registry crates must import moved generic helpers through `siumai-provider-utils` / internal `crate::provider_utils`, not old core utility paths:\n{}",
        violations.join("\n")
    );
}

#[test]
fn core_validator_does_not_own_provider_model_catalogs() {
    let validator =
        std::fs::read_to_string(crate_root().join("src").join("params").join("validator.rs"))
            .expect("read params/validator.rs");

    for forbidden in [
        "is_model_supported",
        "suggest_alternative_model",
        "deepseek",
        "openrouter",
        "siliconflow",
        "fireworks",
        "moonshotai",
        "llama-v3p1",
        "claude-",
        "gemini-",
        "gpt-4",
    ] {
        assert!(
            !validator.contains(forbidden),
            "siumai-core validator must not contain provider/model catalog fragment `{forbidden}`"
        );
    }
}

#[test]
fn core_does_not_own_provider_hosted_tool_factories() {
    let manifest_dir = crate_root();
    let src_dir = manifest_dir.join("src");
    let core_lib = fs::read_to_string(src_dir.join("lib.rs")).expect("read src/lib.rs");

    assert!(
        !src_dir.join("hosted_tools").exists(),
        "provider-hosted tool factories belong to protocol/provider-owned code, not siumai-core"
    );
    assert!(
        !core_lib.contains("pub mod hosted_tools"),
        "siumai-core must not expose a provider-specific hosted_tools module"
    );
    assert!(
        !src_dir.join("tools.rs").exists(),
        "provider-defined tool catalogs belong to protocol/provider-owned code, not siumai-core"
    );
    assert!(
        !core_lib.contains("pub mod tools"),
        "siumai-core must not expose a provider-specific tools module"
    );
}

#[test]
fn core_primary_client_identity_is_provider_id_first() {
    let manifest_dir = crate_root();
    let client_rs = fs::read_to_string(manifest_dir.join("src").join("compat").join("client.rs"))
        .expect("read compat/client.rs");
    let production_source = production_non_comment_source(&client_rs);

    assert!(
        production_source.contains("fn provider_id(&self) -> Cow<'static, str>"),
        "LlmClient should expose provider_id as the primary provider identity"
    );
    assert!(
        !production_source.contains("fn provider_type(&self) -> ProviderType"),
        "ProviderType classification should not remain on the primary LlmClient contract"
    );
    assert!(
        !production_source.contains("pub fn provider_type(&self) -> ProviderType"),
        "ClientWrapper should not promote closed ProviderType classification as a primary API"
    );
    assert!(
        !production_source.contains("ProviderType::from_name"),
        "siumai-core should not map provider ids through the closed ProviderType enum"
    );

    let params_common =
        fs::read_to_string(manifest_dir.join("src").join("params").join("common.rs"))
            .expect("read params/common.rs");
    let params_common = production_non_comment_source(&params_common);
    assert!(
        !params_common.contains("ProviderParamsExt"),
        "siumai-core params should not expose ProviderType-based extension traits for primary parameter flows"
    );

    let params_validator =
        fs::read_to_string(manifest_dir.join("src").join("params").join("validator.rs"))
            .expect("read params/validator.rs");
    let params_validator = production_non_comment_source(&params_validator);
    assert!(
        params_validator.contains("validate_for_provider_id"),
        "parameter validation should expose a provider-id-first entry point"
    );
    assert!(
        params_validator.contains("pub provider_id: Cow<'static, str>"),
        "validation reports should keep provider ids as their primary provider identity"
    );
    for forbidden in [
        "pub provider: ProviderType",
        "pub source_provider: ProviderType",
        "pub target_provider: ProviderType",
    ] {
        assert!(
            !params_validator.contains(forbidden),
            "parameter validation reports should not promote closed ProviderType fields: `{forbidden}`"
        );
    }
}

#[test]
fn llm_client_is_physically_scoped_under_compat_module() {
    let manifest_dir = crate_root();
    let src_dir = manifest_dir.join("src");

    let compat_mod =
        fs::read_to_string(src_dir.join("compat").join("mod.rs")).expect("read compat/mod.rs");
    let compat_client = fs::read_to_string(src_dir.join("compat").join("client.rs"))
        .expect("read compat/client.rs");
    let client_alias = fs::read_to_string(src_dir.join("client.rs")).expect("read client.rs");
    let lib_rs = fs::read_to_string(src_dir.join("lib.rs")).expect("read lib.rs");
    let core_client =
        fs::read_to_string(src_dir.join("core").join("client.rs")).expect("read core/client.rs");

    assert!(
        lib_rs.contains("pub mod compat;"),
        "siumai-core should expose an explicit compatibility namespace"
    );
    assert!(
        compat_mod.contains("pub mod client;"),
        "siumai-core::compat should own the client compatibility module"
    );
    assert!(
        compat_client.contains("pub trait LlmClient")
            && compat_client.contains("pub enum ClientWrapper")
            && compat_client.contains("fn as_chat_capability("),
        "generic LlmClient and capability downcast definitions should live in compat/client.rs"
    );

    for forbidden in [
        "pub trait LlmClient",
        "pub enum ClientWrapper",
        "fn as_chat_capability(",
        "impl dyn LlmClient",
    ] {
        assert!(
            !client_alias.contains(forbidden),
            "siumai_core::client should be a migration alias, not the physical implementation owner: `{forbidden}`"
        );
    }
    assert!(
        client_alias.contains("pub use crate::compat::client::*;"),
        "siumai_core::client should re-export the compat client surface as a migration alias"
    );
    assert!(
        core_client.contains("pub use crate::compat::client::LlmClient;"),
        "siumai_core::core::client should point at the compat-owned LlmClient definition"
    );
}

#[test]
fn core_standards_module_does_not_reintroduce_provider_protocol_islands() {
    let standards_dir = crate_root().join("src").join("standards");
    let standards_mod =
        fs::read_to_string(standards_dir.join("mod.rs")).expect("read src/standards/mod.rs");
    let provider_fragments = [
        "amazon",
        "anthropic",
        "azure",
        "bedrock",
        "cohere",
        "deepinfra",
        "deepseek",
        "fireworks",
        "gemini",
        "google",
        "groq",
        "mistral",
        "moonshot",
        "ollama",
        "openai",
        "perplexity",
        "togetherai",
        "vertex",
        "xai",
    ];

    let mut violations = Vec::new();
    for entry in fs::read_dir(&standards_dir).expect("read src/standards") {
        let entry = entry.expect("read standards entry");
        let path = entry.path();
        let file_name = entry.file_name().to_string_lossy().replace('_', "-");

        if path.is_dir() {
            violations.push(format!(
                "{}: protocol/provider standard subdirectories belong in protocol or provider crates",
                path.display()
            ));
        }

        for fragment in provider_fragments {
            if file_name.contains(fragment)
                || standards_mod.contains(&format!("pub mod {fragment}"))
                || standards_mod.contains(&format!("mod {fragment}"))
            {
                violations.push(format!(
                    "{}: provider/protocol fragment `{fragment}`",
                    path.display()
                ));
            }
        }
    }

    assert!(
        violations.is_empty(),
        "siumai-core::standards must stay provider-agnostic; protocol/provider islands belong in `siumai-protocol-*` or `siumai-provider-*` crates:\n{}",
        violations.join("\n")
    );
}

#[test]
fn core_client_wrapper_does_not_expose_provider_specific_constructors() {
    let client =
        fs::read_to_string(crate_root().join("src").join("client.rs")).expect("read src/client.rs");

    for forbidden in [
        "pub fn openai(",
        "pub fn anthropic(",
        "pub fn gemini(",
        "pub fn groq(",
        "pub fn xai(",
        "pub fn ollama(",
        "ClientWrapper::openai",
        "Provider::openai",
    ] {
        assert!(
            !client.contains(forbidden),
            "siumai-core ClientWrapper must stay provider-agnostic; use ClientWrapper::new(...) instead of `{forbidden}`"
        );
    }
}

#[test]
fn core_auth_does_not_own_provider_specific_vertex_url_helpers() {
    let manifest_dir = crate_root();
    let src_dir = manifest_dir.join("src");
    let auth_dir = src_dir.join("auth");
    let auth_mod = fs::read_to_string(auth_dir.join("mod.rs")).expect("read src/auth/mod.rs");

    assert!(
        !auth_dir.join("vertex.rs").exists(),
        "Vertex URL helpers belong to the google-vertex provider crate, not siumai-core"
    );
    assert!(
        !auth_mod.contains("pub mod vertex"),
        "siumai-core::auth must not expose a provider-specific vertex module"
    );

    let mut rust_sources = Vec::new();
    collect_rust_sources(&src_dir, &mut rust_sources);

    for source_path in rust_sources {
        let relative_path = normalized_relative_path(&manifest_dir, &source_path);
        let source = fs::read_to_string(&source_path)
            .unwrap_or_else(|error| panic!("read {relative_path}: {error}"));

        for forbidden in [
            "GOOGLE_VERTEX_EXPRESS_BASE_URL",
            "google_vertex_base_url",
            "google_vertex_anthropic_base_url",
            "google_vertex_maas_base_url",
            "vertex_base_url",
            "aiplatform.googleapis.com/v1/projects",
        ] {
            assert!(
                !source.contains(forbidden),
                "siumai-core must not own provider-specific Vertex URL helper fragment `{forbidden}`; found in `{relative_path}`"
            );
        }
    }
}

#[test]
fn core_auth_does_not_own_provider_specific_gcp_token_providers() {
    let manifest_dir = crate_root();
    let src_dir = manifest_dir.join("src");
    let auth_dir = src_dir.join("auth");
    let auth_mod = fs::read_to_string(auth_dir.join("mod.rs")).expect("read src/auth/mod.rs");

    for module_file in ["adc.rs", "service_account.rs"] {
        assert!(
            !auth_dir.join(module_file).exists(),
            "Google Cloud token provider implementations belong to provider-owned code, not siumai-core"
        );
    }

    for module_export in ["pub mod adc", "pub mod service_account"] {
        assert!(
            !auth_mod.contains(module_export),
            "siumai-core::auth must not expose provider-specific Google Cloud auth module `{module_export}`"
        );
    }

    let mut rust_sources = Vec::new();
    collect_rust_sources(&src_dir, &mut rust_sources);

    for source_path in rust_sources {
        let relative_path = normalized_relative_path(&manifest_dir, &source_path);
        let source = fs::read_to_string(&source_path)
            .unwrap_or_else(|error| panic!("read {relative_path}: {error}"));

        for forbidden in [
            "AdcTokenProvider",
            "ServiceAccountTokenProvider",
            "ServiceAccountCredentials",
            "GOOGLE_APPLICATION_CREDENTIALS",
            "GOOGLE_OAUTH_ACCESS_TOKEN",
            "oauth2.googleapis.com/token",
            "computeMetadata/v1/instance/service-accounts",
        ] {
            assert!(
                !source.contains(forbidden),
                "siumai-core must not own provider-specific Google Cloud auth fragment `{forbidden}`; found in `{relative_path}`"
            );
        }
    }
}

#[test]
fn core_defaults_do_not_own_provider_specific_defaults() {
    let defaults = fs::read_to_string(crate_root().join("src").join("defaults.rs"))
        .expect("read src/defaults.rs");

    assert!(
        !defaults.contains("pub mod providers"),
        "provider-specific defaults belong to provider or registry-owned code, not siumai-core"
    );

    for forbidden in [
        "api.openai.com/v1",
        "gpt-4o-mini",
        "api.anthropic.com",
        "claude-3-5-haiku-20241022",
        "api.siliconflow.cn/v1",
        "deepseek-ai/DeepSeek-V3.1",
        "api.groq.com/openai/v1",
        "llama-3.3-70b-versatile",
    ] {
        assert!(
            !defaults.contains(forbidden),
            "siumai-core::defaults must not contain provider-specific default fragment `{forbidden}`"
        );
    }
}

#[test]
fn runtime_http_defaults_do_not_use_passive_http_config_default() {
    let root = workspace_root();
    let mut rust_sources = Vec::new();

    collect_rust_sources(&root.join("siumai-core").join("src"), &mut rust_sources);
    collect_rust_sources(&root.join("siumai-registry").join("src"), &mut rust_sources);
    collect_rust_sources(&root.join("siumai").join("src"), &mut rust_sources);
    collect_workspace_crate_sources_by_prefix(
        &root,
        &["siumai-protocol-", "siumai-provider-", "siumai-extras"],
        &mut rust_sources,
    );

    let mut violations = Vec::new();
    for source_path in rust_sources {
        let relative_path = normalized_relative_path(&root, &source_path);
        let source = fs::read_to_string(&source_path)
            .unwrap_or_else(|error| panic!("read {relative_path}: {error}"));
        let production_source = source
            .split("\n#[cfg(test)]")
            .next()
            .unwrap_or(source.as_str());

        if production_source.contains("HttpConfig::default()")
            || production_source.contains("crate::types::HttpConfig::default()")
        {
            violations.push(relative_path);
        }
    }

    assert!(
        violations.is_empty(),
        "runtime production paths must not treat passive `HttpConfig::default()` as runtime defaults; use `defaults::http::config_default()` for client/config construction or `HttpConfig::empty()` for request-level overrides:\n{}",
        violations.join("\n")
    );
}

#[test]
fn core_hook_builder_does_not_own_provider_specific_body_presets() {
    let hook_builder = fs::read_to_string(
        crate_root()
            .join("src")
            .join("execution")
            .join("transformers")
            .join("hook_builder.rs"),
    )
    .expect("read hook_builder.rs");

    for forbidden in [
        "with_openai_base",
        "with_anthropic_base",
        "openai_base_chat_body",
        "anthropic_base_chat_body",
        "OpenAI-compatible base chat body builder",
        "Anthropic-compatible base chat body builder",
    ] {
        assert!(
            !hook_builder.contains(forbidden),
            "siumai-core::HookBuilder must not own provider-specific body preset `{forbidden}`; use a custom builder or provider/protocol-owned helper instead"
        );
    }
}

#[test]
fn core_provider_spec_does_not_own_provider_shaped_route_fallbacks() {
    let provider_spec = fs::read_to_string(
        crate_root()
            .join("src")
            .join("core")
            .join("provider_spec.rs"),
    )
    .expect("read provider_spec.rs");

    assert!(
        provider_spec.contains("fn try_chat_url")
            && provider_spec.contains("fn try_embedding_url")
            && provider_spec.contains("fn try_image_url")
            && provider_spec.contains("fn try_rerank_url"),
        "siumai-core ProviderSpec must expose fallible route resolution for core executors"
    );

    for forbidden in [
        "fn legacy_provider_route",
        "legacy_provider_route(ctx",
        "chat/completions",
        "images/generations",
        "images/edits",
        "images/variations",
        "models/{model_id}",
        "OpenAI-compatible default route",
        "default OpenAI-style",
        "default OpenAI-compatible",
        "fn provider_route_result",
        "provider_route_result(",
        "fn chat_url(",
        "fn embedding_url(",
        "fn image_url(",
        "fn image_edit_url(",
        "fn image_variation_url(",
        "fn rerank_url(",
        "fn models_url(",
        "fn model_url(",
        "self.chat_url(stream, req, ctx)",
        "self.embedding_url(req, ctx)",
        "self.image_url(req, ctx)",
        "self.image_edit_url(req, ctx)",
        "self.image_variation_url(req, ctx)",
        "self.rerank_url(req, ctx)",
        "self.models_url(ctx)",
        "self.model_url(model_id, ctx)",
    ] {
        assert!(
            !provider_spec.contains(forbidden),
            "siumai-core ProviderSpec fallible route defaults must not depend on legacy provider-shaped route hooks or fallback routes: `{forbidden}`"
        );
    }
}

#[test]
fn core_stream_factory_does_not_own_provider_sse_end_markers() {
    let manifest_dir = crate_root();
    let streaming_dir = manifest_dir.join("src").join("streaming");
    let converters =
        fs::read_to_string(streaming_dir.join("converters.rs")).expect("read converters.rs");
    let factory = fs::read_to_string(streaming_dir.join("factory.rs")).expect("read factory.rs");
    let adapters = fs::read_to_string(streaming_dir.join("adapters.rs")).expect("read adapters.rs");
    let sse_json = fs::read_to_string(streaming_dir.join("sse_json.rs")).expect("read sse_json.rs");
    let http_interceptor = fs::read_to_string(
        manifest_dir
            .join("src")
            .join("execution")
            .join("http")
            .join("interceptor.rs"),
    )
    .expect("read execution/http/interceptor.rs");
    let transformer = fs::read_to_string(
        manifest_dir
            .join("src")
            .join("execution")
            .join("transformers")
            .join("stream.rs"),
    )
    .expect("read execution/transformers/stream.rs");

    assert!(
        converters.contains("fn is_stream_end_event(&self, _event: &Event) -> bool"),
        "siumai-core::SseEventConverter must expose a protocol-owned SSE end marker predicate"
    );
    assert!(
        transformer.contains("fn is_stream_end_event(&self, _event: &Event) -> bool"),
        "siumai-core::StreamChunkTransformer must expose a protocol-owned SSE end marker predicate"
    );
    assert!(
        factory.contains("converter.is_stream_end_event(&event)"),
        "siumai-core::StreamFactory must delegate SSE end marker recognition to the converter"
    );

    for forbidden in [
        "event.data.trim() == \"[DONE]\"",
        "event.data.trim()==\"[DONE]\"",
        "data: [DONE]",
        "data: [DONE]\\n\\n",
    ] {
        assert!(
            !factory.contains(forbidden),
            "siumai-core::StreamFactory must not own provider/protocol SSE end marker `{forbidden}`"
        );
    }

    assert!(
        !adapters.contains("data: \"[DONE]\"") && !adapters.contains("\"[DONE]\".to_string()"),
        "siumai-core streaming adapters must not synthesize provider/protocol SSE end marker payloads"
    );
    assert!(
        sse_json.contains("done_markers: Vec::new()"),
        "siumai-core SSE JSON helpers must default to no protocol-specific done markers"
    );
    assert!(
        !http_interceptor.contains("event.data.trim() == \"[DONE]\""),
        "siumai-core HTTP tracing must not treat provider/protocol SSE payloads as core-owned stream completion markers"
    );
}

#[test]
fn core_stream_helpers_only_initialize_empty_provider_metadata() {
    let manifest_dir = crate_root();

    for relative_path in ["src/streaming/builder.rs", "src/streaming/factory.rs"] {
        let source = fs::read_to_string(manifest_dir.join(relative_path))
            .unwrap_or_else(|error| panic!("read {relative_path}: {error}"));
        let production_source = source
            .split("#[cfg(test)]")
            .next()
            .expect("production source section");

        let unexpected_provider_metadata_lines = production_source
            .lines()
            .enumerate()
            .filter_map(|(index, line)| {
                let trimmed = line.trim();
                if trimmed.contains("provider_metadata") && trimmed != "provider_metadata: None," {
                    Some((index + 1, trimmed.to_string()))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();

        assert!(
            unexpected_provider_metadata_lines.is_empty(),
            "{relative_path} may only initialize provider_metadata as None in core stream helpers; provider metadata maps belong to protocol/provider-owned stream converters: {unexpected_provider_metadata_lines:?}"
        );
    }
}

#[test]
fn core_json_stream_executor_does_not_handle_provider_maps() {
    let manifest_dir = crate_root();
    let relative_path = "src/execution/executors/stream_json.rs";
    let source = fs::read_to_string(manifest_dir.join(relative_path))
        .unwrap_or_else(|error| panic!("read {relative_path}: {error}"));
    let production_source = source
        .split("#[cfg(test)]")
        .next()
        .expect("production source section");

    assert!(
        production_source.contains("JsonEventConverter")
            && production_source.contains("create_json_stream_from_transport_body"),
        "{relative_path} must keep provider-specific stream parsing delegated to injected JSON converters"
    );

    for forbidden in [
        "provider_metadata",
        "provider_options",
        "ProviderMetadata",
        "ProviderOptionsMap",
        "ContentPart::",
        "SharedV4ProviderMetadata::from",
    ] {
        assert!(
            !production_source.contains(forbidden),
            "{relative_path} must not read or construct provider metadata/options maps"
        );
    }
}

#[test]
fn core_family_contract_and_tooling_sources_do_not_handle_provider_maps() {
    let manifest_dir = crate_root();
    let checked_files = [
        "src/text.rs",
        "src/completion.rs",
        "src/speech.rs",
        "src/transcription.rs",
        "src/tooling.rs",
        "src/traits.rs",
        "src/traits/audio.rs",
        "src/traits/speech.rs",
        "src/traits/transcription.rs",
    ];

    for relative_path in checked_files {
        let source = fs::read_to_string(manifest_dir.join(relative_path))
            .unwrap_or_else(|error| panic!("read {relative_path}: {error}"));
        let production_source = source
            .split("#[cfg(test)]")
            .next()
            .expect("production source section");

        for forbidden in [
            "provider_metadata",
            "provider_options",
            "ProviderMetadata",
            "ProviderOptionsMap",
            "ContentPart::",
            "SharedV4ProviderMetadata::from",
        ] {
            assert!(
                !production_source.contains(forbidden),
                "{relative_path} must remain a provider-map-neutral family/tooling contract surface"
            );
        }
    }
}

#[test]
fn core_ui_surface_is_split_across_named_modules() {
    let source =
        fs::read_to_string(crate_root().join("src/ui.rs")).expect("read siumai-core ui.rs");

    assert!(
        source.contains("mod conversion;")
            && source.contains("mod types;")
            && source.contains("mod validation;"),
        "siumai-core::ui should stay split into named modules"
    );
    assert!(
        !source.contains("pub fn validate_ui_messages(")
            && !source.contains("pub fn convert_to_model_messages(")
            && !source.contains("pub enum UiMessageError"),
        "siumai-core::ui should not inline the implementation back into the module root"
    );
}

#[test]
fn core_sample_streaming_middleware_only_initializes_empty_provider_metadata() {
    let manifest_dir = crate_root();
    let relative_path = "src/execution/middleware/samples.rs";
    let source = fs::read_to_string(manifest_dir.join(relative_path))
        .unwrap_or_else(|error| panic!("read {relative_path}: {error}"));
    let production_source = source
        .split("#[cfg(test)]")
        .next()
        .expect("production source section");

    for forbidden in [
        "provider_options",
        "ProviderOptionsMap",
        "SharedV4ProviderMetadata::from",
        "\"openai\"",
        "\"anthropic\"",
        "\"google\"",
        "\"gemini\"",
        "\"bedrock\"",
        "\"minimaxi\"",
    ] {
        assert!(
            !production_source.contains(forbidden),
            "{relative_path} must stay a provider-neutral sample middleware"
        );
    }

    let unexpected_provider_metadata_lines = production_source
        .lines()
        .enumerate()
        .filter_map(|(index, line)| {
            let trimmed = line.trim();
            if trimmed.contains("provider_metadata") && trimmed != "provider_metadata: None," {
                Some((index + 1, trimmed.to_string()))
            } else {
                None
            }
        })
        .collect::<Vec<_>>();

    assert!(
        unexpected_provider_metadata_lines.is_empty(),
        "{relative_path} may only initialize synthetic stream parts with empty provider_metadata: {unexpected_provider_metadata_lines:?}"
    );
}

#[test]
fn core_provider_options_parser_stays_request_only_and_provider_agnostic() {
    let provider_utils_path = workspace_root()
        .join("siumai-provider-utils")
        .join("src")
        .join("provider_options.rs");
    let source = fs::read_to_string(&provider_utils_path)
        .unwrap_or_else(|error| panic!("read {}: {error}", provider_utils_path.display()));
    let production_source = source
        .split("#[cfg(test)]")
        .next()
        .expect("production source section");

    assert!(
        production_source.contains("parse_provider_options")
            && production_source.contains("ProviderOptionsMap"),
        "siumai-provider-utils/src/provider_options.rs must remain the generic request-side provider options parser"
    );

    for forbidden in [
        "provider_metadata",
        "ProviderMetadata",
        "ContentPart::",
        "\"openai\"",
        "\"anthropic\"",
        "\"google\"",
        "\"gemini\"",
        "\"bedrock\"",
        "\"minimaxi\"",
        "\"xai\"",
        "\"deepseek\"",
        "\"groq\"",
        "\"cohere\"",
        "\"ollama\"",
    ] {
        assert!(
            !production_source.contains(forbidden),
            "siumai-provider-utils/src/provider_options.rs must parse request provider options without reading response metadata or concrete provider namespaces"
        );
    }
}

#[test]
fn core_streaming_tool_call_tracker_only_uses_callback_provider_metadata() {
    let manifest_dir = crate_root();
    let relative_path = "src/utils/streaming_tool_call.rs";
    let source = fs::read_to_string(manifest_dir.join(relative_path))
        .unwrap_or_else(|error| panic!("read {relative_path}: {error}"));
    let production_source = source
        .split("#[cfg(test)]")
        .next()
        .expect("production source section");

    for forbidden in [
        "\"openai\"",
        "\"anthropic\"",
        "\"google\"",
        "\"gemini\"",
        "\"bedrock\"",
        "\"minimaxi\"",
        "\"xai\"",
        "\"deepseek\"",
        "\"groq\"",
        "\"cohere\"",
        "\"ollama\"",
    ] {
        assert!(
            !production_source.contains(forbidden),
            "{relative_path} must not hard-code provider metadata namespaces"
        );
    }

    assert!(
        production_source.contains("with_extract_metadata_fn")
            && production_source.contains("with_build_tool_call_provider_metadata_fn"),
        "{relative_path} may only handle provider metadata through caller-supplied generic callbacks"
    );

    for forbidden in [
        "provider_options",
        "ProviderOptions",
        "SharedV4ProviderMetadata::from",
        "std::collections::HashMap::from",
    ] {
        assert!(
            !production_source.contains(forbidden),
            "{relative_path} must not read request provider options or construct provider metadata maps directly"
        );
    }
}

#[test]
fn core_structured_output_helpers_only_merge_generic_response_metadata() {
    let manifest_dir = crate_root();
    let relative_path = "src/structured_output.rs";
    let source = fs::read_to_string(manifest_dir.join(relative_path))
        .unwrap_or_else(|error| panic!("read {relative_path}: {error}"));
    let production_source = source
        .split("#[cfg(test)]")
        .next()
        .expect("production source section");

    for forbidden in [
        "provider_options",
        "ProviderOptions",
        "\"openai\"",
        "\"anthropic\"",
        "\"google\"",
        "\"gemini\"",
        "\"bedrock\"",
        "\"minimaxi\"",
        "\"xai\"",
        "\"deepseek\"",
        "\"groq\"",
        "\"cohere\"",
        "\"ollama\"",
    ] {
        assert!(
            !production_source.contains(forbidden),
            "{relative_path} must stay provider-agnostic and must not inspect request provider options"
        );
    }

    let unexpected_provider_metadata_lines = production_source
        .lines()
        .map(str::trim)
        .filter(|line| line.contains("provider_metadata"))
        .filter(|line| {
            !matches!(
                *line,
                "provider_metadata,"
                    | "provider_metadata: accumulated.provider_metadata.or(provider_metadata),"
            )
        })
        .collect::<Vec<_>>();

    assert!(
        unexpected_provider_metadata_lines.is_empty(),
        "{relative_path} may only generically merge response provider_metadata while consolidating stream output: {unexpected_provider_metadata_lines:?}"
    );
}

#[test]
fn core_streaming_runtime_tests_do_not_use_provider_model_fixtures() {
    let root = crate_root();
    let streaming_dir = root.join("src").join("streaming");
    let mut rust_sources = Vec::new();
    collect_rust_sources(&streaming_dir, &mut rust_sources);

    let forbidden = [
        "gpt-",
        "claude-",
        "gemini-",
        "o1-preview",
        "api.openai.com",
        "api.anthropic.com",
        "generativelanguage.googleapis.com",
    ];

    let mut violations = Vec::new();
    for source_path in rust_sources {
        let relative_path = normalized_relative_path(&root, &source_path);
        let source = fs::read_to_string(&source_path)
            .unwrap_or_else(|error| panic!("read {relative_path}: {error}"));

        for forbidden_fragment in forbidden {
            if source.contains(forbidden_fragment) {
                violations.push(format!("{relative_path}: `{forbidden_fragment}`"));
            }
        }
    }

    assert!(
        violations.is_empty(),
        "siumai-core streaming runtime tests and docs must use provider-neutral fixtures:\n{}",
        violations.join("\n")
    );
}

#[test]
fn core_source_does_not_use_provider_model_fixture_literals() {
    let root = crate_root();
    let src_dir = root.join("src");
    let mut rust_sources = Vec::new();
    collect_rust_sources(&src_dir, &mut rust_sources);

    let forbidden = [
        "openai",
        "anthropic",
        "gemini",
        "google",
        "azure",
        "bedrock",
        "groq",
        "ollama",
        "cohere",
        "deepseek",
        "minimaxi",
        "togetherai",
        "xai",
        "gpt-",
        "claude-",
        "api.openai",
        "api.anthropic",
        "generativelanguage.googleapis",
        "aiplatform.googleapis",
        "x-openai-request-id",
        "x-goog-request-id",
    ];

    let mut violations = Vec::new();
    for source_path in rust_sources {
        let relative_path = normalized_relative_path(&root, &source_path);
        let source = fs::read_to_string(&source_path)
            .unwrap_or_else(|error| panic!("read {relative_path}: {error}"));
        let source_lower = source.to_ascii_lowercase();

        for forbidden_fragment in forbidden {
            if source_lower.contains(forbidden_fragment) {
                violations.push(format!("{relative_path}: `{forbidden_fragment}`"));
            }
        }
    }

    assert!(
        violations.is_empty(),
        "siumai-core/src must stay provider-agnostic; use provider-neutral fixtures in core docs/tests and provider-owned crates for concrete provider behavior:\n{}",
        violations.join("\n")
    );
}

#[test]
fn provider_specs_do_not_reintroduce_legacy_string_route_hooks() {
    let root = workspace_root();
    let mut rust_sources = Vec::new();
    collect_workspace_crate_sources_by_prefix(
        &root,
        &["siumai-provider-", "siumai-protocol-"],
        &mut rust_sources,
    );

    let legacy_route_defs = [
        "fn chat_url(",
        "fn embedding_url(",
        "fn image_url(",
        "fn image_edit_url(",
        "fn image_variation_url(",
        "fn rerank_url(",
        "fn models_url(",
        "fn model_url(",
    ];

    let mut violations = Vec::new();
    for source_path in rust_sources {
        let relative_path = normalized_relative_path(&root, &source_path);
        let source = fs::read_to_string(&source_path)
            .unwrap_or_else(|error| panic!("read {relative_path}: {error}"));
        let mut impl_starts = Vec::new();
        let mut offset = 0;
        while let Some(pos) = source[offset..].find("impl ProviderSpec for") {
            impl_starts.push(offset + pos);
            offset += pos + "impl ProviderSpec for".len();
        }

        for (index, start) in impl_starts.iter().copied().enumerate() {
            let end = impl_starts.get(index + 1).copied().unwrap_or(source.len());
            let segment = &source[start..end];
            for legacy_route_def in legacy_route_defs {
                if segment.contains(legacy_route_def) {
                    violations.push(format!("{relative_path}: `{legacy_route_def}`"));
                }
            }
        }
    }

    assert!(
        violations.is_empty(),
        "provider/protocol ProviderSpec impls must expose only fallible `try_*_url(...)` route hooks:\n{}",
        violations.join("\n")
    );
}

#[test]
fn migrated_gemini_route_specs_do_not_reintroduce_legacy_string_hooks() {
    let root = workspace_root();
    let checked_files = [
        "siumai-protocol-gemini/src/standards/gemini/chat.rs",
        "siumai-protocol-gemini/src/standards/gemini/embedding.rs",
        "siumai-protocol-gemini/src/standards/gemini/image.rs",
        "siumai-provider-gemini/src/providers/gemini/spec.rs",
    ];
    let legacy_route_defs = [
        "fn chat_url(",
        "fn embedding_url(",
        "fn image_url(",
        "fn image_edit_url(",
        "fn image_variation_url(",
        "fn model_url(",
    ];

    let mut violations = Vec::new();
    for relative_path in checked_files {
        let path = root.join(relative_path);
        let source = fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("read {relative_path}: {error}"));
        for legacy_def in legacy_route_defs {
            if source.contains(legacy_def) {
                violations.push(format!("{relative_path}: `{legacy_def}`"));
            }
        }
    }

    assert!(
        violations.is_empty(),
        "migrated Gemini route specs must keep `try_*_url(...)` as the only implemented route path:\n{}",
        violations.join("\n")
    );
}

#[test]
fn migrated_openai_protocol_route_specs_do_not_reintroduce_legacy_string_hooks() {
    let root = workspace_root();
    let checked_files = [
        "siumai-protocol-openai/src/standards/openai/chat.rs",
        "siumai-protocol-openai/src/standards/openai/embedding.rs",
        "siumai-protocol-openai/src/standards/openai/image.rs",
        "siumai-protocol-openai/src/standards/openai/rerank.rs",
        "siumai-protocol-openai/src/standards/openai/compat/spec.rs",
    ];
    let legacy_route_defs = [
        "fn chat_url(",
        "fn embedding_url(",
        "fn image_url(",
        "fn image_edit_url(",
        "fn image_variation_url(",
        "fn rerank_url(",
        "fn models_url(",
        "fn model_url(",
    ];

    let mut violations = Vec::new();
    for relative_path in checked_files {
        let path = root.join(relative_path);
        let source = fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("read {relative_path}: {error}"));
        for legacy_def in legacy_route_defs {
            if source.contains(legacy_def) {
                violations.push(format!("{relative_path}: `{legacy_def}`"));
            }
        }
    }

    assert!(
        violations.is_empty(),
        "migrated OpenAI protocol and compat route specs must keep `try_*_url(...)` as the only implemented route path:\n{}",
        violations.join("\n")
    );
}

#[test]
fn migrated_openai_provider_route_specs_do_not_reintroduce_legacy_string_hooks() {
    let root = workspace_root();
    let checked_files = ["siumai-provider-openai/src/providers/openai/spec.rs"];
    let legacy_route_defs = [
        "fn chat_url(",
        "fn embedding_url(",
        "fn image_url(",
        "fn image_edit_url(",
        "fn image_variation_url(",
        "fn rerank_url(",
        "fn models_url(",
        "fn model_url(",
    ];

    let mut violations = Vec::new();
    for relative_path in checked_files {
        let path = root.join(relative_path);
        let source = fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("read {relative_path}: {error}"));
        for legacy_def in legacy_route_defs {
            if source.contains(legacy_def) {
                violations.push(format!("{relative_path}: `{legacy_def}`"));
            }
        }
    }

    assert!(
        violations.is_empty(),
        "migrated OpenAI provider route specs must keep `try_*_url(...)` as the only implemented route path:\n{}",
        violations.join("\n")
    );
}

#[test]
fn migrated_anthropic_route_specs_do_not_reintroduce_legacy_string_hooks() {
    let root = workspace_root();
    let checked_files = [
        "siumai-protocol-anthropic/src/standards/anthropic/chat.rs",
        "siumai-provider-anthropic/src/providers/anthropic/spec.rs",
        "siumai-provider-google-vertex/src/providers/anthropic_vertex/spec.rs",
    ];
    let legacy_route_defs = ["fn chat_url(", "fn models_url(", "fn model_url("];

    let mut violations = Vec::new();
    for relative_path in checked_files {
        let path = root.join(relative_path);
        let source = fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("read {relative_path}: {error}"));
        for legacy_def in legacy_route_defs {
            if source.contains(legacy_def) {
                violations.push(format!("{relative_path}: `{legacy_def}`"));
            }
        }
    }

    assert!(
        violations.is_empty(),
        "migrated Anthropic route specs must keep `try_*_url(...)` as the only implemented route path:\n{}",
        violations.join("\n")
    );
}

#[test]
fn migrated_openai_compatible_provider_route_specs_do_not_reintroduce_legacy_string_hooks() {
    let root = workspace_root();
    let checked_files = [
        "siumai-provider-deepseek/src/providers/deepseek/spec.rs",
        "siumai-provider-groq/src/providers/groq/spec.rs",
    ];
    let legacy_route_defs = ["fn chat_url("];

    let mut violations = Vec::new();
    for relative_path in checked_files {
        let path = root.join(relative_path);
        let source = fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("read {relative_path}: {error}"));
        for legacy_def in legacy_route_defs {
            if source.contains(legacy_def) {
                violations.push(format!("{relative_path}: `{legacy_def}`"));
            }
        }
    }

    assert!(
        violations.is_empty(),
        "migrated OpenAI-compatible provider route specs must keep `try_*_url(...)` as the only implemented route path:\n{}",
        violations.join("\n")
    );
}

#[test]
fn migrated_rerank_provider_route_specs_do_not_reintroduce_legacy_string_hooks() {
    let root = workspace_root();
    let checked_files = [
        "siumai-provider-cohere/src/standards/cohere/mod.rs",
        "siumai-provider-cohere/src/standards/cohere/rerank.rs",
        "siumai-provider-togetherai/src/standards/togetherai/rerank.rs",
    ];
    let legacy_route_defs = ["fn chat_url(", "fn embedding_url(", "fn rerank_url("];

    let mut violations = Vec::new();
    for relative_path in checked_files {
        let path = root.join(relative_path);
        let source = fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("read {relative_path}: {error}"));
        for legacy_def in legacy_route_defs {
            if source.contains(legacy_def) {
                violations.push(format!("{relative_path}: `{legacy_def}`"));
            }
        }
    }

    assert!(
        violations.is_empty(),
        "migrated Cohere/TogetherAI route specs must keep `try_*_url(...)` as the only implemented route path:\n{}",
        violations.join("\n")
    );
}

#[test]
fn migrated_local_and_multi_surface_provider_route_specs_do_not_reintroduce_legacy_string_hooks() {
    let root = workspace_root();
    let checked_files = [
        "siumai-provider-ollama/src/providers/ollama/spec.rs",
        "siumai-provider-minimaxi/src/providers/minimaxi/spec.rs",
    ];
    let legacy_route_defs = [
        "fn chat_url(",
        "fn embedding_url(",
        "fn image_url(",
        "fn models_url(",
    ];

    let mut violations = Vec::new();
    for relative_path in checked_files {
        let path = root.join(relative_path);
        let source = fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("read {relative_path}: {error}"));
        for legacy_def in legacy_route_defs {
            if source.contains(legacy_def) {
                violations.push(format!("{relative_path}: `{legacy_def}`"));
            }
        }
    }

    assert!(
        violations.is_empty(),
        "migrated Ollama/MiniMaxi route specs must keep `try_*_url(...)` as the only implemented route path:\n{}",
        violations.join("\n")
    );
}

#[test]
fn migrated_bedrock_and_azure_route_specs_do_not_reintroduce_legacy_string_hooks() {
    let root = workspace_root();
    let checked_files = [
        "siumai-provider-amazon-bedrock/src/standards/bedrock/chat.rs",
        "siumai-provider-amazon-bedrock/src/standards/bedrock/embedding.rs",
        "siumai-provider-amazon-bedrock/src/standards/bedrock/image.rs",
        "siumai-provider-amazon-bedrock/src/standards/bedrock/rerank.rs",
        "siumai-provider-azure/src/providers/azure_openai/spec.rs",
    ];
    let legacy_route_defs = [
        "fn chat_url(",
        "fn embedding_url(",
        "fn image_url(",
        "fn image_edit_url(",
        "fn image_variation_url(",
        "fn rerank_url(",
    ];

    let mut violations = Vec::new();
    for relative_path in checked_files {
        let path = root.join(relative_path);
        let source = fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("read {relative_path}: {error}"));
        for legacy_def in legacy_route_defs {
            if source.contains(legacy_def) {
                violations.push(format!("{relative_path}: `{legacy_def}`"));
            }
        }
    }

    assert!(
        violations.is_empty(),
        "migrated Bedrock/Azure route specs must keep `try_*_url(...)` as the only implemented route path:\n{}",
        violations.join("\n")
    );
}

#[test]
fn migrated_google_vertex_route_specs_do_not_reintroduce_legacy_string_hooks() {
    let root = workspace_root();
    let checked_files = [
        "siumai-provider-google-vertex/src/standards/vertex_generative_ai.rs",
        "siumai-provider-google-vertex/src/standards/vertex_embedding.rs",
        "siumai-provider-google-vertex/src/standards/vertex_gemini_image.rs",
        "siumai-provider-google-vertex/src/standards/vertex_imagen.rs",
    ];
    let legacy_route_defs = [
        "fn chat_url(",
        "fn embedding_url(",
        "fn image_url(",
        "fn image_edit_url(",
        "fn image_variation_url(",
    ];
    let legacy_route_calls = [
        ".chat_url(",
        ".embedding_url(",
        ".image_url(",
        ".image_edit_url(",
        ".image_variation_url(",
    ];

    let mut violations = Vec::new();
    for relative_path in checked_files {
        let path = root.join(relative_path);
        let source = fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("read {relative_path}: {error}"));
        for legacy_def in legacy_route_defs {
            if source.contains(legacy_def) {
                violations.push(format!("{relative_path}: `{legacy_def}`"));
            }
        }
        for legacy_call in legacy_route_calls {
            if source.contains(legacy_call) {
                violations.push(format!("{relative_path}: `{legacy_call}`"));
            }
        }
    }

    assert!(
        violations.is_empty(),
        "migrated Google Vertex route specs must keep `try_*_url(...)` as the only implemented route path:\n{}",
        violations.join("\n")
    );
}

#[test]
fn production_request_paths_use_fallible_provider_routes() {
    let root = workspace_root();
    let checked_files = [
        "siumai-provider-anthropic/src/providers/anthropic/models.rs",
        "siumai-provider-gemini/src/providers/gemini/models.rs",
        "siumai-provider-google-vertex/src/providers/anthropic_vertex/client.rs",
        "siumai-provider-ollama/src/providers/ollama/chat.rs",
        "siumai-provider-ollama/src/providers/ollama/model_listing.rs",
        "siumai-provider-openai/src/providers/openai/client.rs",
        "siumai-provider-openai/src/providers/openai/client/models.rs",
        "siumai-provider-openai/src/providers/openai/models.rs",
        "siumai-provider-openai-compatible/src/providers/openai_compatible/openai_client.rs",
    ];

    let legacy_route_calls = [
        ".chat_url(",
        ".embedding_url(",
        ".image_url(",
        ".image_edit_url(",
        ".image_variation_url(",
        ".rerank_url(",
        ".models_url(",
        ".model_url(",
    ];

    let mut violations = Vec::new();
    for relative_path in checked_files {
        let path = root.join(relative_path);
        let source = fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("read {relative_path}: {error}"));
        let production_source = source
            .split("\n#[cfg(test)]")
            .next()
            .unwrap_or(source.as_str());

        for legacy_call in legacy_route_calls {
            if production_source.contains(legacy_call) {
                violations.push(format!("{relative_path}: `{legacy_call}`"));
            }
        }
    }

    assert!(
        violations.is_empty(),
        "production request paths must use fallible `try_*_url(...)` route resolution instead of legacy string hooks:\n{}",
        violations.join("\n")
    );
}

#[test]
fn core_executor_tests_and_docs_use_fallible_route_hooks() {
    let root = crate_root();
    let execution_dir = root.join("src").join("execution");
    let mut rust_sources = Vec::new();
    collect_rust_sources(&execution_dir, &mut rust_sources);

    let legacy_route_defs = [
        "fn chat_url(",
        "fn embedding_url(",
        "fn image_url(",
        "fn image_edit_url(",
        "fn image_variation_url(",
        "fn rerank_url(",
    ];
    let legacy_route_calls = [
        ".chat_url(",
        ".embedding_url(",
        ".image_url(",
        ".image_edit_url(",
        ".image_variation_url(",
        ".rerank_url(",
    ];

    let mut violations = Vec::new();
    for source_path in rust_sources {
        let relative_path = normalized_relative_path(&root, &source_path);
        let source = fs::read_to_string(&source_path)
            .unwrap_or_else(|error| panic!("read {relative_path}: {error}"));
        for legacy_def in legacy_route_defs {
            if source.contains(legacy_def) {
                violations.push(format!("{relative_path}: `{legacy_def}`"));
            }
        }
        for legacy_call in legacy_route_calls {
            if source.contains(legacy_call) {
                violations.push(format!("{relative_path}: `{legacy_call}`"));
            }
        }
    }

    assert!(
        violations.is_empty(),
        "core executor tests and examples must use fallible `try_*_url(...)` route hooks so new code does not learn the legacy string hook contract:\n{}",
        violations.join("\n")
    );
}

#[test]
fn route_fixture_tests_use_fallible_provider_routes() {
    let root = workspace_root();
    let mut rust_sources = Vec::new();

    collect_rust_sources(&root.join("siumai").join("tests"), &mut rust_sources);
    collect_workspace_crate_sources_by_prefix(
        &root,
        &["siumai-provider-", "siumai-protocol-"],
        &mut rust_sources,
    );

    let legacy_route_calls = [
        ".chat_url(",
        ".embedding_url(",
        ".image_url(",
        ".image_edit_url(",
        ".image_variation_url(",
        ".rerank_url(",
        ".models_url(",
        ".model_url(",
    ];

    let mut violations = Vec::new();
    for source_path in rust_sources {
        let relative_path = normalized_relative_path(&root, &source_path);
        let source = fs::read_to_string(&source_path)
            .unwrap_or_else(|error| panic!("read {relative_path}: {error}"));

        let test_source_start = if relative_path.starts_with("siumai/tests/")
            || relative_path.contains("/tests/")
            || relative_path.ends_with("/tests.rs")
        {
            0
        } else if let Some(start) = source.find("#[cfg(test)]") {
            start
        } else {
            continue;
        };

        let line_offset = source[..test_source_start].lines().count();
        let test_source = &source[test_source_start..];
        for (line_index, line) in test_source.lines().enumerate() {
            for legacy_call in legacy_route_calls {
                if line.contains(legacy_call) {
                    violations.push(format!(
                        "{relative_path}:{}: `{legacy_call}`",
                        line_offset + line_index + 1
                    ));
                }
            }
        }
    }

    assert!(
        violations.is_empty(),
        "route fixture and provider/protocol tests must assert fallible `try_*_url(...)` routes instead of legacy string hooks:\n{}",
        violations.join("\n")
    );
}

#[test]
fn core_streaming_tool_call_helper_is_not_documented_as_openai_specific() {
    let streaming_tool_call = fs::read_to_string(
        crate_root()
            .join("src")
            .join("utils")
            .join("streaming_tool_call.rs"),
    )
    .expect("read streaming_tool_call.rs");

    for forbidden in ["OpenAI-compatible", "OpenAI-style", "OpenAI-like"] {
        assert!(
            !streaming_tool_call.contains(forbidden),
            "siumai-core streaming tool-call compatibility helper must not present itself as provider-specific core runtime: `{forbidden}`"
        );
    }
}

#[test]
fn core_provider_agnostic_docs_do_not_describe_core_as_openai_compatible() {
    let root = crate_root();
    let checked_files = [
        "src/completion.rs",
        "src/execution/transformers/stream.rs",
        "src/standards/mod.rs",
        "src/utils/builder_helpers.rs",
        "src/custom_provider/guide.rs",
        "src/observability/tracing/README.md",
        "src/utils/url.rs",
    ];

    for relative_path in checked_files {
        let source = fs::read_to_string(root.join(relative_path)).expect("read checked source");
        for forbidden in [
            "OpenAI-compatible",
            "OpenAI compatible",
            "OpenAI-style",
            "OpenAI style",
            "OpenAI-like",
            "https://api.openai.com",
        ] {
            assert!(
                !source.contains(forbidden),
                "siumai-core provider-agnostic docs/examples must not describe core as provider compatibility code: {relative_path} contains `{forbidden}`"
            );
        }
    }
}

#[test]
fn core_does_not_own_provider_specific_bridge_contracts() {
    let manifest_dir = crate_root();
    let src_dir = manifest_dir.join("src");
    let mut rust_sources = Vec::new();
    collect_rust_sources(&src_dir, &mut rust_sources);

    let migration_allowlist: [&str; 0] = [];
    let provider_bridge_patterns = [
        "pub enum BridgeTarget",
        "OpenAiResponses",
        "OpenAiChatCompletions",
        "AnthropicMessages",
        "GeminiGenerateContent",
        "pub enum StreamPartNamespace",
        "to_openai_custom_event_payload",
        "to_anthropic_custom_event_payload",
        "to_gemini_custom_event_payload",
        "to_protocol_custom_event",
    ];

    for source_path in rust_sources {
        let relative_path = normalized_relative_path(&manifest_dir, &source_path);
        if migration_allowlist.contains(&relative_path.as_str()) {
            continue;
        }

        let source = fs::read_to_string(&source_path)
            .unwrap_or_else(|error| panic!("read {relative_path}: {error}"));
        for pattern in provider_bridge_patterns {
            assert!(
                !source.contains(pattern),
                "siumai-core must not own provider-specific bridge contract or stream-event serialization residue `{pattern}`; found in `{relative_path}`"
            );
        }
    }
}
