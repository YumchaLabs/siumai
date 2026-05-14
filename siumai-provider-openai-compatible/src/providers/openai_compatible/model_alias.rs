//! Provider-owned model alias normalization for OpenAI-compatible vendors.
//!
//! These aliases depend on concrete provider ids and vendor model namespaces, so they belong in
//! the OpenAI-compatible provider package rather than `siumai-core`.

/// Normalize a model id for OpenAI-compatible provider presets.
pub fn normalize_model_id(provider_id: &str, model: &str) -> String {
    if model.is_empty() {
        return model.to_string();
    }

    let pid = provider_id.to_ascii_lowercase();
    let mut m = model.trim().to_string();
    let ml = m.to_ascii_lowercase();

    if let Some(rest) = ml.strip_prefix("models/") {
        m = rest.to_string();
    }

    match pid.as_str() {
        "deepseek" => match ml.as_str() {
            "deepseek-r1" | "r1" | "reasoner" => "deepseek-reasoner".to_string(),
            "deepseek-v3" | "v3" | "chat" => "deepseek-chat".to_string(),
            _ => m,
        },
        "siliconflow" => {
            if ml.starts_with("deepseek-v3.1") {
                return "deepseek-ai/DeepSeek-V3.1".to_string();
            }
            if ml.starts_with("deepseek-v3") {
                return "deepseek-ai/DeepSeek-V3".to_string();
            }
            if ml.starts_with("deepseek-r1") {
                return "deepseek-ai/DeepSeek-R1".to_string();
            }
            if ml.starts_with("qwen3-235b-a22b") {
                return "Qwen/Qwen3-235B-A22B".to_string();
            }
            if ml.starts_with("qwen3-32b") {
                return "Qwen/Qwen3-32B".to_string();
            }
            if ml.starts_with("qwen3-14b") {
                return "Qwen/Qwen3-14B".to_string();
            }
            if ml.starts_with("qwen3-8b") {
                return "Qwen/Qwen3-8B".to_string();
            }
            if ml == "qwen-2.5-72b-instruct" || ml == "qwen2.5-72b-instruct" {
                return "Qwen/Qwen2.5-72B-Instruct".to_string();
            }
            if ml == "qwen-2.5-32b-instruct" || ml == "qwen2.5-32b-instruct" {
                return "Qwen/Qwen2.5-32B-Instruct".to_string();
            }
            if ml == "qwen-2.5-14b-instruct" || ml == "qwen2.5-14b-instruct" {
                return "Qwen/Qwen2.5-14B-Instruct".to_string();
            }
            if ml == "qwen-2.5-7b-instruct" || ml == "qwen2.5-7b-instruct" {
                return "Qwen/Qwen2.5-7B-Instruct".to_string();
            }
            if ml.starts_with("kimi-k2-instruct") {
                return "moonshotai/Kimi-K2-Instruct".to_string();
            }
            if ml == "glm-4.5" {
                return "zai-org/GLM-4.5".to_string();
            }
            if ml == "glm-4.5-air" {
                return "zai-org/GLM-4.5-Air".to_string();
            }
            if ml == "glm-4.5v" {
                return "zai-org/GLM-4.5V".to_string();
            }
            if ml.starts_with("llama-3.1-")
                || ml.starts_with("llama-3.2-")
                || ml.starts_with("llama-3.3-")
            {
                return format!("meta-llama/{m}");
            }
            if ml.starts_with("mistral-") || ml.starts_with("mixtral-") {
                return format!("mistralai/{m}");
            }
            m
        }
        "together" => {
            if ml.contains('/') {
                return m;
            }
            if ml.starts_with("llama-3.1-")
                || ml.starts_with("llama-3.2-")
                || ml.starts_with("llama-3.3-")
            {
                return format!("meta-llama/{m}");
            }
            if ml.starts_with("mistral-") || ml.starts_with("mixtral-") {
                return format!("mistralai/{m}");
            }
            m
        }
        "fireworks" => {
            if ml == "llama-v3p1-8b-instruct" {
                return "accounts/fireworks/models/llama-v3p1-8b-instruct".to_string();
            }
            m
        }
        "openrouter" => {
            if ml.contains('/') {
                return m;
            }
            if ml.starts_with("gpt-5")
                || ml.starts_with("gpt-4o")
                || ml.starts_with("gpt-4.1")
                || ml == "gpt-4"
                || ml == "o1"
                || ml == "o1-mini"
                || ml == "o3-mini"
                || ml == "o4-mini"
                || ml.starts_with("gpt-3.5")
            {
                return format!("openai/{m}");
            }
            if ml.starts_with("claude-3.5-sonnet")
                || ml.starts_with("claude-3-5-sonnet")
                || ml.starts_with("claude-3.5-haiku")
                || ml.starts_with("claude-3-5-haiku")
                || ml.starts_with("claude-sonnet-4")
                || ml.starts_with("claude-opus-4")
                || ml.starts_with("claude-opus-4.1")
                || ml.starts_with("claude-2")
            {
                let norm = ml.replace("claude-3-5-", "claude-3.5-");
                if norm != ml {
                    m = norm;
                }
                return format!("anthropic/{m}");
            }
            if ml.starts_with("gemini-1.5-")
                || ml.starts_with("gemini-2.0-")
                || ml.starts_with("gemini-2.5-")
            {
                return format!("google/{m}");
            }
            if ml.starts_with("deepseek-") || ml == "deepseek" {
                if ml == "deepseek-v3" {
                    return "deepseek/deepseek-v3".to_string();
                }
                if ml == "deepseek-r1" {
                    return "deepseek/deepseek-r1".to_string();
                }
                return format!("deepseek/{m}");
            }
            if ml.starts_with("grok-") {
                return format!("xai/{m}");
            }
            if ml.contains("sonar") {
                return format!("perplexity/{m}");
            }
            if ml.starts_with("llama-3.1-")
                || ml.starts_with("llama-3.2-")
                || ml.starts_with("llama-3.3-")
            {
                return format!("meta-llama/{m}");
            }
            if ml.starts_with("mistral-") || ml.starts_with("mixtral-") {
                return format!("mistralai/{m}");
            }
            if ml.starts_with("command-r") || ml.starts_with("command-") {
                return format!("cohere/{m}");
            }
            if ml.starts_with("qwen") {
                return format!("qwen/{m}");
            }
            m
        }
        _ => m,
    }
}

#[cfg(test)]
mod tests {
    use super::normalize_model_id as norm;

    #[test]
    fn deepseek_aliases() {
        assert_eq!(norm("deepseek", "deepseek-v3"), "deepseek-chat");
        assert_eq!(norm("deepseek", "deepseek-r1"), "deepseek-reasoner");
        assert_eq!(norm("deepseek", "chat"), "deepseek-chat");
        assert_eq!(norm("deepseek", "reasoner"), "deepseek-reasoner");
        assert_eq!(norm("deepseek", "deepseek-chat"), "deepseek-chat");
    }

    #[test]
    fn openrouter_aliases() {
        assert_eq!(norm("openrouter", "gpt-4o-mini"), "openai/gpt-4o-mini");
        assert_eq!(
            norm("openrouter", "claude-3-5-sonnet"),
            "anthropic/claude-3.5-sonnet"
        );
        assert_eq!(
            norm("openrouter", "gemini-2.5-pro"),
            "google/gemini-2.5-pro"
        );
        assert_eq!(
            norm("openrouter", "llama-3.1-70b-instruct"),
            "meta-llama/llama-3.1-70b-instruct"
        );
        assert_eq!(
            norm("openrouter", "llama-3.1-sonar-small-128k-online"),
            "perplexity/llama-3.1-sonar-small-128k-online"
        );
        assert_eq!(
            norm("openrouter", "command-r-plus"),
            "cohere/command-r-plus"
        );
        assert_eq!(
            norm("openrouter", "openai/gpt-4o-mini"),
            "openai/gpt-4o-mini"
        );
    }

    #[test]
    fn openai_compatible_vendor_aliases() {
        assert_eq!(
            norm("siliconflow", "deepseek-v3.1"),
            "deepseek-ai/DeepSeek-V3.1"
        );
        assert_eq!(
            norm("siliconflow", "qwen-2.5-72b-instruct"),
            "Qwen/Qwen2.5-72B-Instruct"
        );
        assert_eq!(
            norm("siliconflow", "kimi-k2-instruct"),
            "moonshotai/Kimi-K2-Instruct"
        );
        assert_eq!(
            norm("together", "llama-3.1-8b-instruct"),
            "meta-llama/llama-3.1-8b-instruct"
        );
        assert_eq!(
            norm("fireworks", "llama-v3p1-8b-instruct"),
            "accounts/fireworks/models/llama-v3p1-8b-instruct"
        );
    }
}
