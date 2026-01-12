#![cfg(feature = "azure")]

use serde::Deserialize;
use serde::de::DeserializeOwned;
use siumai::experimental::core::{ProviderContext, ProviderSpec};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("azure")
        .join("openai-provider")
        .join("url")
}

fn case_dirs(root: &Path) -> Vec<PathBuf> {
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(root)
        .expect("read fixture root dir")
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false))
        .map(|e| e.path())
        .collect();

    dirs.sort_by(|a, b| {
        a.file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .cmp(&b.file_name().unwrap_or_default().to_string_lossy())
    });

    dirs
}

fn read_text(path: impl AsRef<Path>) -> String {
    std::fs::read_to_string(path).expect("read fixture text")
}

fn read_json<T: DeserializeOwned>(path: impl AsRef<Path>) -> T {
    let text = read_text(path);
    serde_json::from_str(&text).expect("parse fixture json")
}

#[derive(Debug, Clone, Deserialize)]
struct AzureContextFixture {
    base_url: String,
    api_key: String,
    #[serde(default)]
    http_extra_headers: HashMap<String, String>,
    #[serde(default)]
    api_version: Option<String>,
    #[serde(default)]
    use_deployment_based_urls: Option<bool>,
    #[serde(default)]
    extras: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Deserialize)]
struct OpFixture {
    operation: String,
    #[serde(default)]
    chat_mode: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct EmbeddingRequestFixture {
    input: Vec<String>,
    model: String,
}

#[derive(Debug, Clone, Deserialize)]
struct ImageGenerationRequestFixture {
    prompt: String,
    model: String,
    #[serde(default)]
    count: Option<u32>,
}

fn spec_from_fixture(
    ctx: &AzureContextFixture,
    op: &OpFixture,
) -> siumai::provider_ext::azure::AzureOpenAiSpec {
    let mut spec = siumai::provider_ext::azure::AzureOpenAiSpec::new(
        siumai::provider_ext::azure::AzureUrlConfig {
            api_version: ctx.api_version.clone().unwrap_or_else(|| "v1".to_string()),
            use_deployment_based_urls: ctx.use_deployment_based_urls.unwrap_or(false),
        },
    );

    if op.operation == "chat" && matches!(op.chat_mode.as_deref(), Some("chat_completions")) {
        spec = spec.with_chat_mode(siumai::provider_ext::azure::AzureChatMode::ChatCompletions);
    }

    spec
}

fn provider_context_from_fixture(ctx: AzureContextFixture) -> ProviderContext {
    ProviderContext::new(
        "azure",
        ctx.base_url,
        Some(ctx.api_key),
        ctx.http_extra_headers,
    )
    .with_extras(ctx.extras)
}

fn run_case(root: &Path) {
    let ctx_fx: AzureContextFixture = read_json(root.join("context.json"));
    let op: OpFixture = read_json(root.join("op.json"));
    let expected_url = read_text(root.join("expected_url.txt")).trim().to_string();

    let spec = spec_from_fixture(&ctx_fx, &op);
    let ctx = provider_context_from_fixture(ctx_fx);

    let got_url = match op.operation.as_str() {
        "chat" => {
            let req: siumai::prelude::unified::ChatRequest = read_json(root.join("request.json"));
            spec.chat_url(req.stream, &req, &ctx)
        }
        "embedding" => {
            let fx: EmbeddingRequestFixture = read_json(root.join("request.json"));
            let req = siumai::prelude::unified::EmbeddingRequest {
                input: fx.input,
                model: Some(fx.model),
                ..Default::default()
            };
            spec.embedding_url(&req, &ctx)
        }
        "image_generation" => {
            let fx: ImageGenerationRequestFixture = read_json(root.join("request.json"));
            let req = siumai::prelude::unified::ImageGenerationRequest {
                prompt: fx.prompt,
                model: Some(fx.model),
                count: fx.count.unwrap_or(1),
                ..Default::default()
            };
            spec.image_url(&req, &ctx)
        }
        "audio_tts" => {
            let tx = spec.choose_audio_transformer(&ctx).transformer;
            format!("{}{}", spec.audio_base_url(&ctx), tx.tts_endpoint())
        }
        "audio_stt" => {
            let tx = spec.choose_audio_transformer(&ctx).transformer;
            format!("{}{}", spec.audio_base_url(&ctx), tx.stt_endpoint())
        }
        other => panic!("unknown operation: {other}"),
    };

    assert_eq!(got_url, expected_url, "fixture case: {}", root.display());
}

#[test]
fn azure_openai_provider_url_fixtures_match() {
    let roots = case_dirs(&fixtures_dir());
    assert!(!roots.is_empty(), "no fixture cases found");
    for root in roots {
        run_case(&root);
    }
}
