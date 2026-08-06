use std::collections::BTreeMap;

use serde::Deserialize;
use serde_json::{Map, Value};

#[derive(Debug, Deserialize)]
pub(crate) struct MessageResponseWire {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub role: String,
    #[serde(default)]
    pub content: Vec<Value>,
    pub model: String,
    pub stop_reason: Option<String>,
    pub stop_sequence: Option<String>,
    pub stop_details: Option<Value>,
    #[serde(default)]
    pub usage: UsageWire,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct UsageWire {
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub output_tokens_details: Option<OutputTokensDetailsWire>,
    pub cache_creation_input_tokens: Option<u64>,
    pub cache_read_input_tokens: Option<u64>,
    pub service_tier: Option<String>,
    pub server_tool_use: Option<Value>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

impl UsageWire {
    pub fn merge(&mut self, update: Self) {
        merge_optional(&mut self.input_tokens, update.input_tokens);
        merge_optional(&mut self.output_tokens, update.output_tokens);
        merge_optional(
            &mut self.output_tokens_details,
            update.output_tokens_details,
        );
        merge_optional(
            &mut self.cache_creation_input_tokens,
            update.cache_creation_input_tokens,
        );
        merge_optional(
            &mut self.cache_read_input_tokens,
            update.cache_read_input_tokens,
        );
        merge_optional(&mut self.service_tier, update.service_tier);
        merge_optional(&mut self.server_tool_use, update.server_tool_use);
        self.extra.extend(update.extra);
    }

    pub fn provider_details(&self) -> Map<String, Value> {
        let mut details = self.extra.clone().into_iter().collect::<Map<_, _>>();
        insert_optional_number(&mut details, "input_tokens", self.input_tokens);
        insert_optional_number(&mut details, "output_tokens", self.output_tokens);
        if let Some(output_tokens_details) = &self.output_tokens_details {
            details.insert(
                "output_tokens_details".to_string(),
                output_tokens_details.to_value(),
            );
        }
        insert_optional_number(
            &mut details,
            "cache_creation_input_tokens",
            self.cache_creation_input_tokens,
        );
        insert_optional_number(
            &mut details,
            "cache_read_input_tokens",
            self.cache_read_input_tokens,
        );
        if let Some(service_tier) = &self.service_tier {
            details.insert(
                "service_tier".to_string(),
                Value::String(service_tier.clone()),
            );
        }
        if let Some(server_tool_use) = &self.server_tool_use {
            details.insert("server_tool_use".to_string(), server_tool_use.clone());
        }
        details
    }

    pub fn iterations(&self) -> Option<&Value> {
        self.extra.get("iterations")
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct OutputTokensDetailsWire {
    pub thinking_tokens: Option<u64>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

impl OutputTokensDetailsWire {
    fn to_value(&self) -> Value {
        let mut details = self.extra.clone().into_iter().collect::<Map<_, _>>();
        insert_optional_number(&mut details, "thinking_tokens", self.thinking_tokens);
        Value::Object(details)
    }
}

fn merge_optional<T>(target: &mut Option<T>, update: Option<T>) {
    if update.is_some() {
        *target = update;
    }
}

fn insert_optional_number(target: &mut Map<String, Value>, field: &str, value: Option<u64>) {
    if let Some(value) = value {
        target.insert(field.to_string(), Value::from(value));
    }
}

#[derive(Debug, Deserialize)]
pub(crate) struct StreamEventWire {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(flatten)]
    pub fields: Map<String, Value>,
}

impl StreamEventWire {
    pub fn field(&self, name: &str) -> Option<&Value> {
        self.fields.get(name)
    }
}

#[derive(Debug, Deserialize)]
pub(crate) struct StreamMessageStartWire {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub role: String,
    pub model: Option<String>,
    pub stop_reason: Option<String>,
    pub stop_sequence: Option<String>,
    pub stop_details: Option<Value>,
    #[serde(default)]
    pub usage: UsageWire,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

#[derive(Debug, Default, Deserialize)]
pub(crate) struct StreamMessageDeltaWire {
    pub stop_reason: Option<String>,
    pub stop_sequence: Option<String>,
    pub stop_details: Option<Value>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}
