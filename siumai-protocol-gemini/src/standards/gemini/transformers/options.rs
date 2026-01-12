use super::*;

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub(super) struct GeminiProviderOptions {
    pub(super) response_mime_type: Option<String>,
    pub(super) cached_content: Option<String>,
    pub(super) response_modalities: Option<Vec<String>>,
    pub(super) thinking_config: Option<serde_json::Value>,
    pub(super) safety_settings: Option<Vec<serde_json::Value>>,
    pub(super) labels: Option<std::collections::HashMap<String, String>>,
    pub(super) audio_timestamp: Option<bool>,

    // Legacy (deprecated) compatibility fields.
    pub(super) code_execution: Option<LegacyToggleConfig>,
    pub(super) search_grounding: Option<LegacySearchGroundingConfig>,
    pub(super) file_search: Option<LegacyFileSearchConfig>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub(super) struct LegacyToggleConfig {
    pub(super) enabled: bool,
}

impl Default for LegacyToggleConfig {
    fn default() -> Self {
        Self { enabled: true }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub(super) struct LegacySearchGroundingConfig {
    pub(super) enabled: bool,
    pub(super) dynamic_retrieval_config: Option<LegacyDynamicRetrievalConfig>,
}

impl Default for LegacySearchGroundingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            dynamic_retrieval_config: None,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub(super) struct LegacyDynamicRetrievalConfig {
    pub(super) mode: LegacyDynamicRetrievalMode,
    pub(super) dynamic_threshold: Option<serde_json::Number>,
}

impl Default for LegacyDynamicRetrievalConfig {
    fn default() -> Self {
        Self {
            mode: LegacyDynamicRetrievalMode::ModeUnspecified,
            dynamic_threshold: None,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub(super) enum LegacyDynamicRetrievalMode {
    ModeUnspecified,
    ModeDynamic,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub(super) struct LegacyFileSearchConfig {
    pub(super) file_search_store_names: Vec<String>,
}

fn normalize_gemini_provider_options_json(value: &serde_json::Value) -> serde_json::Value {
    fn normalize_key(k: &str) -> Option<&'static str> {
        Some(match k {
            // GeminiOptions (top-level)
            "responseMimeType" => "response_mime_type",
            "cachedContent" => "cached_content",
            "responseModalities" => "response_modalities",
            "thinkingConfig" => "thinking_config",
            "safetySettings" => "safety_settings",
            "audioTimestamp" => "audio_timestamp",
            "codeExecution" => "code_execution",
            "searchGrounding" => "search_grounding",
            "fileSearch" => "file_search",
            // SearchGroundingConfig
            "dynamicRetrievalConfig" => "dynamic_retrieval_config",
            "dynamicThreshold" => "dynamic_threshold",
            // FileSearchConfig
            "fileSearchStoreNames" => "file_search_store_names",
            _ => return None,
        })
    }

    fn inner(value: &serde_json::Value) -> serde_json::Value {
        match value {
            serde_json::Value::Object(map) => {
                let mut out = serde_json::Map::new();
                for (k, v) in map {
                    let nk = normalize_key(k).unwrap_or(k);
                    out.insert(nk.to_string(), inner(v));
                }
                serde_json::Value::Object(out)
            }
            serde_json::Value::Array(arr) => {
                serde_json::Value::Array(arr.iter().map(inner).collect())
            }
            other => other.clone(),
        }
    }

    inner(value)
}

pub(super) fn gemini_options_from_request(req: &ChatRequest) -> Option<GeminiProviderOptions> {
    if let Some(value) = req.provider_options_map.get("gemini") {
        let normalized = normalize_gemini_provider_options_json(value);
        if let Ok(opts) = serde_json::from_value::<GeminiProviderOptions>(normalized) {
            return Some(opts);
        }
    }

    None
}
