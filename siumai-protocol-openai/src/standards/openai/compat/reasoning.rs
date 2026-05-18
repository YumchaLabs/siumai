//! OpenAI-compatible reasoning request policy.
//!
//! Fluent reasoning helpers are public ergonomics. Provider-specific wire shapes live here so the
//! builder and config surfaces do not each maintain their own provider-id switch.

use std::collections::HashMap;
use std::sync::Arc;

use serde_json::{Map, Number, Value};

use super::adapter::{ProviderAdapter, RequestBodyTransformer, RequestTransformingAdapter};
use super::types::RequestType;
use crate::error::LlmError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReasoningFamily {
    ThinkingFields,
    DeepSeekThinking,
    MoonshotThinking,
    XaiReasoningEffort,
    ReasoningFields,
}

#[derive(Debug, Clone, PartialEq)]
enum ReasoningPatchOp {
    Upsert {
        key: &'static str,
        value: Value,
    },
    Remove {
        key: &'static str,
    },
    SetThinking {
        enable: bool,
        budget_tokens: Option<u32>,
        preserve_existing: bool,
    },
}

/// Provider-native request parameter patch produced by the reasoning policy.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct OpenAiCompatibleReasoningPatch {
    operations: Vec<ReasoningPatchOp>,
}

impl OpenAiCompatibleReasoningPatch {
    fn new(operations: Vec<ReasoningPatchOp>) -> Self {
        Self { operations }
    }

    fn upsert(key: &'static str, value: Value) -> Self {
        Self::new(vec![ReasoningPatchOp::Upsert { key, value }])
    }

    fn upserts<const N: usize>(items: [(&'static str, Value); N]) -> Self {
        Self::new(
            items
                .into_iter()
                .map(|(key, value)| ReasoningPatchOp::Upsert { key, value })
                .collect(),
        )
    }

    fn remove<const N: usize>(keys: [&'static str; N]) -> Self {
        Self::new(
            keys.into_iter()
                .map(|key| ReasoningPatchOp::Remove { key })
                .collect(),
        )
    }

    fn set_thinking(enable: bool, budget_tokens: Option<u32>, preserve_existing: bool) -> Self {
        Self::new(vec![ReasoningPatchOp::SetThinking {
            enable,
            budget_tokens,
            preserve_existing,
        }])
    }

    /// Return whether this patch has no effect.
    pub fn is_empty(&self) -> bool {
        self.operations.is_empty()
    }

    /// Apply this patch to a builder-owned provider-default parameter map.
    pub fn apply_to_hash_map(&self, params: &mut HashMap<String, Value>) {
        for operation in &self.operations {
            match operation {
                ReasoningPatchOp::Upsert { key, value } => {
                    params.insert((*key).to_string(), value.clone());
                }
                ReasoningPatchOp::Remove { key } => {
                    params.remove(*key);
                }
                ReasoningPatchOp::SetThinking {
                    enable,
                    budget_tokens,
                    preserve_existing,
                } => {
                    let thinking = build_thinking_value(
                        params.get("thinking"),
                        *enable,
                        *budget_tokens,
                        *preserve_existing,
                    );
                    params.insert("thinking".to_string(), thinking);
                }
            }
        }
    }

    /// Apply this patch to a serialized OpenAI-compatible request body.
    pub fn apply_to_value(&self, params: &mut Value) {
        let Some(obj) = params.as_object_mut() else {
            return;
        };
        self.apply_to_json_map(obj);
    }

    /// Apply this patch to a JSON object map.
    pub fn apply_to_json_map(&self, params: &mut Map<String, Value>) {
        for operation in &self.operations {
            match operation {
                ReasoningPatchOp::Upsert { key, value } => {
                    params.insert((*key).to_string(), value.clone());
                }
                ReasoningPatchOp::Remove { key } => {
                    params.remove(*key);
                }
                ReasoningPatchOp::SetThinking {
                    enable,
                    budget_tokens,
                    preserve_existing,
                } => {
                    let thinking = build_thinking_value(
                        params.get("thinking"),
                        *enable,
                        *budget_tokens,
                        *preserve_existing,
                    );
                    params.insert("thinking".to_string(), thinking);
                }
            }
        }
    }

    /// Wrap an adapter so config-level reasoning defaults can be applied after older defaults.
    pub fn wrap_adapter(self, inner: Box<dyn ProviderAdapter>) -> Box<dyn ProviderAdapter> {
        if self.is_empty() {
            return inner;
        }

        let patch = self;
        let transformer: Arc<dyn RequestBodyTransformer> = Arc::new(
            move |body: &mut Value, _model: &str, request_type: RequestType| {
                if matches!(request_type, RequestType::Chat) {
                    patch.apply_to_value(body);
                }
                Ok::<(), LlmError>(())
            },
        );

        Box::new(RequestTransformingAdapter::new(inner, transformer))
    }
}

/// Provider runtime policy for OpenAI-compatible fluent reasoning helpers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OpenAiCompatibleReasoningPolicy {
    family: ReasoningFamily,
}

impl OpenAiCompatibleReasoningPolicy {
    /// Resolve the reasoning policy for a built-in or custom OpenAI-compatible provider id.
    pub fn for_provider(provider_id: &str) -> Self {
        let normalized = provider_id.trim().to_ascii_lowercase();
        let family = match normalized.as_str() {
            "alibaba" | "qwen" | "siliconflow" => ReasoningFamily::ThinkingFields,
            "deepseek" => ReasoningFamily::DeepSeekThinking,
            "moonshot" | "moonshotai" => ReasoningFamily::MoonshotThinking,
            "xai" => ReasoningFamily::XaiReasoningEffort,
            _ => ReasoningFamily::ReasoningFields,
        };

        Self { family }
    }

    /// Return the provider-native parameters for a reasoning/thinking toggle.
    pub fn reasoning(&self, enable: bool) -> OpenAiCompatibleReasoningPatch {
        self.toggle(enable)
    }

    /// Return the provider-native parameters for a reasoning token budget.
    pub fn reasoning_budget(&self, budget: i32) -> OpenAiCompatibleReasoningPatch {
        self.budget(clamp_i32_budget(budget))
    }

    /// Return the provider-native parameters for the legacy thinking toggle alias.
    pub fn thinking(&self, enable: bool) -> OpenAiCompatibleReasoningPatch {
        self.toggle(enable)
    }

    /// Return the provider-native parameters for the legacy thinking budget alias.
    pub fn thinking_budget(&self, budget: u32) -> OpenAiCompatibleReasoningPatch {
        self.budget(clamp_u32_budget(budget))
    }

    fn toggle(&self, enable: bool) -> OpenAiCompatibleReasoningPatch {
        match self.family {
            ReasoningFamily::ThinkingFields => {
                OpenAiCompatibleReasoningPatch::upsert("enable_thinking", Value::Bool(enable))
            }
            ReasoningFamily::DeepSeekThinking => {
                OpenAiCompatibleReasoningPatch::set_thinking(enable, None, false)
            }
            ReasoningFamily::MoonshotThinking => {
                OpenAiCompatibleReasoningPatch::set_thinking(enable, None, true)
            }
            ReasoningFamily::XaiReasoningEffort => {
                if enable {
                    OpenAiCompatibleReasoningPatch::upsert(
                        "reasoning_effort",
                        Value::String("low".to_string()),
                    )
                } else {
                    OpenAiCompatibleReasoningPatch::remove(["reasoning_effort", "reasoningEffort"])
                }
            }
            ReasoningFamily::ReasoningFields => {
                OpenAiCompatibleReasoningPatch::upsert("enable_reasoning", Value::Bool(enable))
            }
        }
    }

    fn budget(&self, budget: u32) -> OpenAiCompatibleReasoningPatch {
        match self.family {
            ReasoningFamily::ThinkingFields => OpenAiCompatibleReasoningPatch::upserts([
                ("enable_thinking", Value::Bool(true)),
                (
                    "thinking_budget",
                    Value::Number(Number::from(u64::from(budget))),
                ),
            ]),
            ReasoningFamily::DeepSeekThinking => {
                OpenAiCompatibleReasoningPatch::set_thinking(true, None, false)
            }
            ReasoningFamily::MoonshotThinking => {
                OpenAiCompatibleReasoningPatch::set_thinking(true, Some(budget), true)
            }
            ReasoningFamily::XaiReasoningEffort => OpenAiCompatibleReasoningPatch::upsert(
                "reasoning_effort",
                Value::String("high".to_string()),
            ),
            ReasoningFamily::ReasoningFields => OpenAiCompatibleReasoningPatch::upserts([
                ("enable_reasoning", Value::Bool(true)),
                (
                    "reasoning_budget",
                    Value::Number(Number::from(u64::from(budget))),
                ),
            ]),
        }
    }
}

fn clamp_i32_budget(budget: i32) -> u32 {
    budget.clamp(128, 32768) as u32
}

fn clamp_u32_budget(budget: u32) -> u32 {
    budget.clamp(128, 32768)
}

fn build_thinking_value(
    existing: Option<&Value>,
    enable: bool,
    budget_tokens: Option<u32>,
    preserve_existing: bool,
) -> Value {
    let mut thinking = if preserve_existing {
        existing
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default()
    } else {
        Map::new()
    };

    thinking.insert(
        "type".to_string(),
        Value::String(if enable { "enabled" } else { "disabled" }.to_string()),
    );

    if let Some(budget_tokens) = budget_tokens {
        thinking.remove("budgetTokens");
        thinking.insert(
            "budget_tokens".to_string(),
            Value::Number(Number::from(u64::from(budget_tokens))),
        );
    } else if !preserve_existing {
        thinking.remove("budgetTokens");
        thinking.remove("budget_tokens");
    }

    Value::Object(thinking)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn apply(patch: OpenAiCompatibleReasoningPatch) -> HashMap<String, Value> {
        let mut params = HashMap::new();
        patch.apply_to_hash_map(&mut params);
        params
    }

    #[test]
    fn alibaba_family_maps_reasoning_to_thinking_fields() {
        let policy = OpenAiCompatibleReasoningPolicy::for_provider("qwen");
        let mut params = apply(policy.reasoning(true));
        policy.reasoning_budget(2048).apply_to_hash_map(&mut params);

        assert_eq!(
            params.get("enable_thinking"),
            Some(&serde_json::json!(true))
        );
        assert_eq!(
            params.get("thinking_budget"),
            Some(&serde_json::json!(2048))
        );
        assert!(!params.contains_key("enable_reasoning"));
        assert!(!params.contains_key("reasoning_budget"));
    }

    #[test]
    fn deepseek_maps_budget_to_thinking_without_token_budget() {
        let params =
            apply(OpenAiCompatibleReasoningPolicy::for_provider("deepseek").reasoning_budget(2048));

        assert_eq!(
            params.get("thinking"),
            Some(&serde_json::json!({ "type": "enabled" }))
        );
    }

    #[test]
    fn moonshot_toggle_preserves_existing_budget() {
        let policy = OpenAiCompatibleReasoningPolicy::for_provider("moonshotai");
        let mut params = HashMap::new();

        policy.reasoning_budget(4096).apply_to_hash_map(&mut params);
        policy.reasoning(false).apply_to_hash_map(&mut params);

        assert_eq!(
            params.get("thinking"),
            Some(&serde_json::json!({
                "type": "disabled",
                "budget_tokens": 4096
            }))
        );
    }

    #[test]
    fn xai_maps_toggle_and_budget_to_reasoning_effort() {
        let policy = OpenAiCompatibleReasoningPolicy::for_provider("xai");
        let mut params = HashMap::new();

        policy.reasoning(true).apply_to_hash_map(&mut params);
        assert_eq!(
            params.get("reasoning_effort"),
            Some(&serde_json::json!("low"))
        );

        policy.reasoning_budget(2048).apply_to_hash_map(&mut params);
        assert_eq!(
            params.get("reasoning_effort"),
            Some(&serde_json::json!("high"))
        );

        policy.reasoning(false).apply_to_hash_map(&mut params);
        assert!(!params.contains_key("reasoning_effort"));
        assert!(!params.contains_key("reasoningEffort"));
    }

    #[test]
    fn openrouter_and_generic_keep_legacy_reasoning_fields() {
        for provider_id in ["openrouter", "custom-gateway"] {
            let policy = OpenAiCompatibleReasoningPolicy::for_provider(provider_id);
            let mut params = HashMap::new();

            policy.reasoning(true).apply_to_hash_map(&mut params);
            policy.reasoning_budget(2048).apply_to_hash_map(&mut params);

            assert_eq!(
                params.get("enable_reasoning"),
                Some(&serde_json::json!(true))
            );
            assert_eq!(
                params.get("reasoning_budget"),
                Some(&serde_json::json!(2048))
            );
            assert!(!params.contains_key("enable_thinking"), "{provider_id}");
            assert!(!params.contains_key("thinking_budget"), "{provider_id}");
        }
    }

    #[test]
    fn budgets_are_clamped() {
        let policy = OpenAiCompatibleReasoningPolicy::for_provider("openrouter");
        let mut low = HashMap::new();
        let mut high = HashMap::new();

        policy.reasoning_budget(-1).apply_to_hash_map(&mut low);
        policy.reasoning_budget(50000).apply_to_hash_map(&mut high);

        assert_eq!(low.get("reasoning_budget"), Some(&serde_json::json!(128)));
        assert_eq!(
            high.get("reasoning_budget"),
            Some(&serde_json::json!(32768))
        );
    }
}
