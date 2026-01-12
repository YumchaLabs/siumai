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
        .join("request")
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
}

fn read_optional_headers(path: impl AsRef<Path>) -> HashMap<String, String> {
    let path = path.as_ref();
    if !path.exists() {
        return HashMap::new();
    }
    read_json(path)
}

fn run_case(root: &Path) {
    let req: siumai::prelude::unified::ChatRequest = read_json(root.join("request.json"));
    let ctx_fx: AzureContextFixture = read_json(root.join("context.json"));
    let request_headers = read_optional_headers(root.join("request_headers.json"));
    let expected_url = read_text(root.join("expected_url.txt")).trim().to_string();
    let expected_headers: HashMap<String, String> = read_json(root.join("expected_headers.json"));

    let spec =
        siumai::experimental::providers::azure::providers::azure_openai::AzureOpenAiSpec::new(
            siumai::experimental::providers::azure::providers::azure_openai::AzureUrlConfig {
                api_version: ctx_fx.api_version.unwrap_or_else(|| "v1".to_string()),
                use_deployment_based_urls: ctx_fx.use_deployment_based_urls.unwrap_or(false),
            },
        );
    let ctx = ProviderContext::new(
        "azure",
        ctx_fx.base_url,
        Some(ctx_fx.api_key),
        ctx_fx.http_extra_headers,
    );

    let got_url = spec.chat_url(req.stream, &req, &ctx);
    assert_eq!(got_url, expected_url, "fixture case: {}", root.display());

    let base_headers = spec.build_headers(&ctx).expect("build headers");
    let merged = spec.merge_request_headers(base_headers, &request_headers);
    let got_headers = siumai::experimental::execution::http::headers::headermap_to_hashmap(&merged);

    assert_eq!(
        got_headers,
        expected_headers,
        "fixture case: {}",
        root.display()
    );
}

#[test]
fn azure_openai_provider_request_fixtures_match() {
    let roots = case_dirs(&fixtures_dir());
    assert!(!roots.is_empty(), "no fixture cases found");
    for root in roots {
        run_case(&root);
    }
}
