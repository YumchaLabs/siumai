//! 自定义新增提供商参数 (Custom Add Provider Parameters)
//!
//! 本示例演示如何为提供商添加新的参数，而无需等待库更新。
//! 当提供商发布新功能时，您可以立即使用它们。
//!
//! This example demonstrates how to add new parameters to providers without waiting
//! for library updates. When providers release new features, you can use them immediately.
//!
//! # 使用场景 (Use Cases)
//!
//! - ✅ 提供商发布了新功能，但库还未更新
//! - ✅ 测试实验性/Beta功能
//! - ✅ 使用私有/定制的提供商扩展
//!
//! # 运行示例 (Run)
//!
//! ```bash
//! cargo run --example 自定义新增提供商参数 --features xai
//! ```

use siumai::prelude::*;
use siumai::types::{CustomProviderOptions, ProviderOptions};

/// 示例：为xAI添加新功能参数
///
/// 假设xAI刚发布了"deferred"模式和"parallel_function_calling"功能，
/// 但库还没有内置支持。我们可以立即使用它们！
#[derive(Debug, Clone)]
pub struct XaiNewFeatures {
    /// 延迟模式 - 假设这是xAI的新功能
    pub deferred: Option<bool>,
    /// 并行函数调用 - 假设这是xAI的新功能
    pub parallel_function_calling: Option<bool>,
}

impl CustomProviderOptions for XaiNewFeatures {
    fn provider_id(&self) -> &str {
        "xai"
    }

    fn to_json(&self) -> Result<serde_json::Value, LlmError> {
        let mut map = serde_json::Map::new();

        if let Some(deferred) = self.deferred {
            map.insert("deferred".to_string(), serde_json::Value::Bool(deferred));
        }

        if let Some(parallel) = self.parallel_function_calling {
            map.insert(
                "parallel_function_calling".to_string(),
                serde_json::Value::Bool(parallel),
            );
        }

        Ok(serde_json::Value::Object(map))
    }
}

/// 示例：为OpenAI添加新功能参数
#[derive(Debug, Clone)]
pub struct OpenAiNewFeatures {
    /// 自定义参数 - 假设这是OpenAI的新功能
    pub reasoning_effort: Option<String>,
    /// 实验性模式 - 假设这是OpenAI的新功能
    pub experimental_mode: Option<bool>,
}

impl CustomProviderOptions for OpenAiNewFeatures {
    fn provider_id(&self) -> &str {
        "openai"
    }

    fn to_json(&self) -> Result<serde_json::Value, LlmError> {
        let mut map = serde_json::Map::new();

        if let Some(effort) = &self.reasoning_effort {
            map.insert(
                "reasoning_effort".to_string(),
                serde_json::Value::String(effort.clone()),
            );
        }

        if let Some(experimental) = self.experimental_mode {
            map.insert(
                "experimental_mode".to_string(),
                serde_json::Value::Bool(experimental),
            );
        }

        Ok(serde_json::Value::Object(map))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔧 自定义新增提供商参数示例");
    println!("🔧 Custom Add Provider Parameters Example\n");

    // ============================================================
    // 示例 1: 为xAI添加新功能
    // Example 1: Add new features to xAI
    // ============================================================
    println!("📝 示例 1: 为xAI添加新功能");
    println!("📝 Example 1: Add new features to xAI");
    println!("================================================\n");

    let xai_features = XaiNewFeatures {
        deferred: Some(true),
        parallel_function_calling: Some(true),
    };

    // 转换为ProviderOptions
    let options = ProviderOptions::from_custom(xai_features)?;

    println!("✅ 创建了自定义xAI参数:");
    println!("✅ Created custom xAI parameters:");
    println!("   {:?}\n", options);

    // ============================================================
    // 示例 2: 在ChatRequest中使用
    // Example 2: Use with ChatRequest
    // ============================================================
    println!("📝 示例 2: 在ChatRequest中使用");
    println!("📝 Example 2: Use with ChatRequest");
    println!("=====================================\n");

    let openai_features = OpenAiNewFeatures {
        reasoning_effort: Some("high".to_string()),
        experimental_mode: Some(true),
    };

    let request = ChatRequest::new(vec![
        ChatMessage::user("分析这段数据 / Analyze this data").build(),
    ])
    .with_provider_options(ProviderOptions::from_custom(openai_features)?);

    println!("✅ 创建了带自定义参数的ChatRequest");
    println!("✅ Created ChatRequest with custom parameters");
    println!(
        "   提供商 / Provider: {:?}",
        request.provider_options.provider_id()
    );
    println!();

    // ============================================================
    // 示例 3: 直接使用HashMap（更灵活）
    // Example 3: Direct HashMap usage (more flexible)
    // ============================================================
    println!("📝 示例 3: 直接使用HashMap（更灵活）");
    println!("📝 Example 3: Direct HashMap usage (more flexible)");
    println!("====================================================\n");

    use std::collections::HashMap;

    let mut custom_params = HashMap::new();
    custom_params.insert("new_feature".to_string(), serde_json::json!("enabled"));
    custom_params.insert("priority_level".to_string(), serde_json::json!(5));

    let options = ProviderOptions::Custom {
        provider_id: "xai".to_string(),
        options: custom_params,
    };

    println!("✅ 创建了灵活的自定义参数:");
    println!("✅ Created flexible custom parameters:");
    println!("   {:?}\n", options);

    // ============================================================
    // 最佳实践 / Best Practices
    // ============================================================
    println!("💡 最佳实践 / Best Practices");
    println!("==============================\n");

    println!("✅ 推荐 / DO:");
    println!("   - 只为库中尚未支持的新功能使用自定义参数");
    println!("   - Only use custom parameters for new features not yet in the library");
    println!("   - 为提供商的Beta/实验性功能使用");
    println!("   - Use for provider's Beta/experimental features");
    println!();

    println!("❌ 不推荐 / DON'T:");
    println!("   - 不要为已有内置支持的功能使用自定义参数");
    println!("   - Don't use custom parameters for features with built-in support");
    println!("   - 例如：temperature, max_tokens 应该使用 CommonParams");
    println!("   - Example: temperature, max_tokens should use CommonParams");
    println!();

    // ============================================================
    // 工作原理 / How It Works
    // ============================================================
    println!("🔍 工作原理 / How It Works");
    println!("===========================\n");

    println!("1. 实现 CustomProviderOptions trait");
    println!("   Implement CustomProviderOptions trait");
    println!();

    println!("2. 使用 ProviderOptions::from_custom() 转换");
    println!("   Convert using ProviderOptions::from_custom()");
    println!();

    println!("3. 库会自动将参数注入到API请求中");
    println!("   Library automatically injects parameters into API request");
    println!("   (通过 ProviderSpec::chat_before_send() hook)");
    println!();

    println!("4. 提供商收到完整的请求参数");
    println!("   Provider receives complete request parameters");
    println!();

    println!("🎉 示例完成！/ Example Complete!");
    println!();
    println!("📚 关键要点 / Key Takeaways:");
    println!("   1. 可以立即使用提供商的新功能，无需等待库更新");
    println!(
        "      Can use provider's new features immediately without waiting for library updates"
    );
    println!("   2. 类型安全的方式扩展功能");
    println!("      Type-safe way to extend functionality");
    println!("   3. 所有6个提供商都支持此功能");
    println!("      All 6 providers support this feature");
    println!("   4. 当库添加内置支持后，可以平滑迁移");
    println!("      Smooth migration when library adds built-in support");

    Ok(())
}
