//! Shared types for OpenAI-compatible providers (extracted)

use std::{borrow::Cow, collections::HashMap};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestType {
    Chat,
    Embedding,
    Rerank,
    ImageGeneration,
}

#[derive(Debug, Clone)]
pub struct FieldMappings {
    pub thinking_fields: Vec<Cow<'static, str>>,
    pub content_field: Cow<'static, str>,
    pub tool_calls_field: Cow<'static, str>,
    pub role_field: Cow<'static, str>,
}

impl Default for FieldMappings {
    fn default() -> Self {
        Self {
            thinking_fields: vec![Cow::Borrowed("thinking")],
            content_field: Cow::Borrowed("content"),
            tool_calls_field: Cow::Borrowed("tool_calls"),
            role_field: Cow::Borrowed("role"),
        }
    }
}

pub trait FieldAccessor {
    fn get_field_value(&self, json: &serde_json::Value, field_path: &str) -> Option<String>;
    fn extract_thinking_content(
        &self,
        json: &serde_json::Value,
        mappings: &FieldMappings,
    ) -> Option<String>;
    fn extract_content(&self, json: &serde_json::Value, mappings: &FieldMappings)
    -> Option<String>;
}

#[derive(Debug, Clone, Default)]
pub struct JsonFieldAccessor;

impl JsonFieldAccessor {
    fn get_nested_value<'a>(
        &self,
        json: &'a serde_json::Value,
        path: &str,
    ) -> Option<&'a serde_json::Value> {
        let parts: Vec<&str> = path.split('.').collect();
        let mut current = json;
        for part in parts {
            if part.is_empty() {
                continue;
            }
            if let Ok(index) = part.parse::<usize>() {
                current = current.get(index)?;
            } else if part.starts_with('[') && part.ends_with(']') {
                let index_str = &part[1..part.len() - 1];
                let index = index_str.parse::<usize>().ok()?;
                current = current.get(index)?;
            } else {
                current = current.get(part)?;
            }
        }
        Some(current)
    }
    fn extract_string_value(&self, value: &serde_json::Value) -> Option<String> {
        match value {
            serde_json::Value::String(s) => Some(s.clone()),
            serde_json::Value::Number(n) => Some(n.to_string()),
            serde_json::Value::Bool(b) => Some(b.to_string()),
            _ => None,
        }
    }
}

impl FieldAccessor for JsonFieldAccessor {
    fn get_field_value(&self, json: &serde_json::Value, field_path: &str) -> Option<String> {
        self.get_nested_value(json, field_path)
            .and_then(|v| self.extract_string_value(v))
    }
    fn extract_thinking_content(
        &self,
        json: &serde_json::Value,
        mappings: &FieldMappings,
    ) -> Option<String> {
        for field_name in &mappings.thinking_fields {
            let paths = vec![
                format!("choices.0.delta.{}", field_name),
                format!("choices.0.message.{}", field_name),
                format!("delta.{}", field_name),
                format!("message.{}", field_name),
                field_name.to_string(),
            ];
            for path in paths {
                if let Some(value) = self.get_field_value(json, &path) {
                    if !value.trim().is_empty() {
                        return Some(value);
                    }
                }
            }
        }
        None
    }
    fn extract_content(
        &self,
        json: &serde_json::Value,
        mappings: &FieldMappings,
    ) -> Option<String> {
        let f = &mappings.content_field;
        let paths = vec![
            format!("choices.0.delta.{}", f),
            format!("choices.0.message.{}", f),
            format!("delta.{}", f),
            format!("message.{}", f),
            f.to_string(),
        ];
        for path in paths {
            if let Some(value) = self.get_field_value(json, &path)
                && !value.trim().is_empty()
            {
                return Some(value);
            }
        }
        None
    }
}

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub force_streaming: bool,
    pub supports_thinking: bool,
    pub supports_tools: bool,
    pub supports_vision: bool,
    pub parameter_mappings: HashMap<String, String>,
    pub max_tokens: Option<u32>,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            force_streaming: false,
            supports_thinking: false,
            supports_tools: true,
            supports_vision: false,
            parameter_mappings: HashMap::new(),
            max_tokens: None,
        }
    }
}

impl ModelConfig {
    pub fn deepseek() -> Self {
        let mut parameter_mappings = HashMap::new();
        parameter_mappings.insert(
            "thinking_budget".to_string(),
            "reasoning_effort".to_string(),
        );
        Self {
            supports_thinking: true,
            max_tokens: Some(8192),
            parameter_mappings,
            ..Default::default()
        }
    }
    pub fn qwen_reasoning() -> Self {
        Self {
            force_streaming: true,
            supports_thinking: true,
            ..Default::default()
        }
    }
    pub fn standard_chat() -> Self {
        Self::default()
    }
}
