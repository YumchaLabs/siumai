//! 自定义完整提供商 (Custom Complete Provider)
//!
//! 本示例演示如何实现一个完全自定义的AI提供商。
//! 适用于私有API、非OpenAI兼容的API等场景。
//!
//! This example demonstrates how to implement a completely custom AI provider.
//! Suitable for private APIs, non-OpenAI-compatible APIs, etc.
//!
//! # 使用场景 (Use Cases)
//!
//! - ✅ 私有/内部AI服务
//! - ✅ 非OpenAI兼容的API
//! - ✅ 自定义协议的AI服务
//! - ✅ 本地模型服务
//!
//! # 运行示例 (Run)
//!
//! ```bash
//! cargo run --example 自定义完整提供商
//! ```

use async_trait::async_trait;
use futures::stream;
use siumai::custom_provider::{CustomChatRequest, CustomChatResponse, CustomProvider};
use siumai::prelude::*;

/// 自定义提供商示例 - 模拟一个简单的AI服务
///
/// Custom Provider Example - Simulates a simple AI service
#[derive(Clone)]
pub struct MyCustomProvider {
    /// API密钥 / API Key
    api_key: String,
    /// 基础URL / Base URL
    base_url: String,
}

impl MyCustomProvider {
    pub fn new(api_key: String, base_url: String) -> Self {
        Self { api_key, base_url }
    }
}

#[async_trait]
impl CustomProvider for MyCustomProvider {
    fn name(&self) -> &str {
        "my-custom-provider"
    }

    fn supported_models(&self) -> Vec<String> {
        vec![
            "custom-model-v1".to_string(),
            "custom-model-v2".to_string(),
            "custom-model-pro".to_string(),
        ]
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new()
            .with_chat()
            .with_streaming()
            .with_tools()
    }

    async fn chat(&self, request: CustomChatRequest) -> Result<CustomChatResponse, LlmError> {
        println!("📤 发送请求到自定义提供商 / Sending request to custom provider");
        println!("   API Key: {}...", &self.api_key[..8]);
        println!("   Base URL: {}", self.base_url);
        println!("   Model: {}", request.model);
        println!("   Messages: {}", request.messages.len());

        // 在实际实现中，这里会调用真实的API
        // In real implementation, this would call the actual API

        // 模拟API响应 / Simulate API response
        let response_text = format!(
            "这是来自 {} 的响应。我收到了您的消息。\n\
             This is a response from {}. I received your message.",
            self.name(),
            self.name()
        );

        use std::collections::HashMap;
        let mut metadata = HashMap::new();
        metadata.insert("provider".to_string(), serde_json::json!(self.name()));
        metadata.insert("model".to_string(), serde_json::json!(request.model));

        Ok(CustomChatResponse {
            content: response_text,
            finish_reason: Some("stop".to_string()),
            usage: Some(Usage::new(10, 20)),
            tool_calls: None,
            metadata,
        })
    }

    async fn chat_stream(&self, request: CustomChatRequest) -> Result<ChatStream, LlmError> {
        println!("📤 发送流式请求到自定义提供商 / Sending streaming request to custom provider");
        println!("   Model: {}", request.model);

        // 模拟流式响应 / Simulate streaming response
        let chunks = vec![
            "这是",
            "来自",
            "自定义",
            "提供商",
            "的",
            "流式",
            "响应。",
            "\n",
            "This ",
            "is ",
            "a ",
            "streaming ",
            "response ",
            "from ",
            "custom ",
            "provider.",
        ];

        let events: Vec<Result<ChatStreamEvent, LlmError>> = chunks
            .into_iter()
            .map(|chunk| {
                Ok(ChatStreamEvent::ContentDelta {
                    delta: chunk.to_string(),
                    index: None,
                })
            })
            .collect();

        Ok(Box::pin(stream::iter(events)))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔧 自定义完整提供商示例");
    println!("🔧 Custom Complete Provider Example\n");

    // ============================================================
    // 示例 1: 创建自定义提供商
    // Example 1: Create custom provider
    // ============================================================
    println!("📝 示例 1: 创建自定义提供商");
    println!("📝 Example 1: Create custom provider");
    println!("======================================\n");

    let provider = MyCustomProvider::new(
        "sk-custom-api-key-12345".to_string(),
        "https://api.my-custom-ai.com".to_string(),
    );

    println!("✅ 创建了自定义提供商:");
    println!("✅ Created custom provider:");
    println!("   名称 / Name: {}", provider.name());
    println!(
        "   支持的模型 / Supported models: {:?}",
        provider.supported_models()
    );
    println!("   能力 / Capabilities: {:?}", provider.capabilities());
    println!();

    // ============================================================
    // 示例 2: 使用自定义提供商进行聊天
    // Example 2: Use custom provider for chat
    // ============================================================
    println!("📝 示例 2: 使用自定义提供商进行聊天");
    println!("📝 Example 2: Use custom provider for chat");
    println!("============================================\n");

    let request = CustomChatRequest::new(
        vec![
            ChatMessage::user("你好！请介绍一下你自己。\nHello! Please introduce yourself.")
                .build(),
        ],
        "custom-model-v1".to_string(),
    );

    let response = provider.chat(request).await?;

    println!("\n✅ 收到响应 / Received response:");
    println!("   {}", response.content);
    println!("   Token使用 / Token usage: {:?}", response.usage);
    println!();

    // ============================================================
    // 示例 3: 使用流式响应
    // Example 3: Use streaming response
    // ============================================================
    println!("📝 示例 3: 使用流式响应");
    println!("📝 Example 3: Use streaming response");
    println!("======================================\n");

    let request = CustomChatRequest::new(
        vec![ChatMessage::user("给我讲个故事 / Tell me a story").build()],
        "custom-model-pro".to_string(),
    );

    let mut stream = provider.chat_stream(request).await?;

    print!("📥 流式响应 / Streaming response: ");
    use futures::StreamExt;
    while let Some(event) = stream.next().await {
        match event? {
            ChatStreamEvent::ContentDelta { delta, .. } => {
                print!("{}", delta);
                std::io::Write::flush(&mut std::io::stdout())?;
            }
            _ => {}
        }
    }
    println!("\n   ✅ 完成 / Done");
    println!();

    // ============================================================
    // 实现要点 / Implementation Points
    // ============================================================
    println!("💡 实现要点 / Implementation Points");
    println!("====================================\n");

    println!("1. 实现 CustomProvider trait");
    println!("   Implement CustomProvider trait");
    println!("   - name(): 提供商名称 / Provider name");
    println!("   - supported_models(): 支持的模型列表 / Supported models");
    println!("   - capabilities(): 提供商能力 / Provider capabilities");
    println!("   - chat(): 同步聊天 / Synchronous chat");
    println!("   - chat_stream(): 流式聊天 / Streaming chat");
    println!();

    println!("2. 处理请求和响应");
    println!("   Handle requests and responses");
    println!("   - 解析 ChatRequest");
    println!("   - Parse ChatRequest");
    println!("   - 调用您的API");
    println!("   - Call your API");
    println!("   - 转换为 ChatResponse");
    println!("   - Convert to ChatResponse");
    println!();

    println!("3. 实现流式响应（可选）");
    println!("   Implement streaming (optional)");
    println!("   - 返回 Stream<Item = ChatStreamEvent>");
    println!("   - Return Stream<Item = ChatStreamEvent>");
    println!("   - 逐块发送内容");
    println!("   - Send content chunk by chunk");
    println!();

    println!("🎉 示例完成！/ Example Complete!");
    println!();
    println!("📚 关键要点 / Key Takeaways:");
    println!("   1. CustomProvider 允许完全自定义AI提供商");
    println!("      CustomProvider allows completely custom AI providers");
    println!("   2. 适用于任何API协议");
    println!("      Works with any API protocol");
    println!("   3. 支持同步和流式响应");
    println!("      Supports both sync and streaming responses");
    println!("   4. 完全控制请求和响应格式");
    println!("      Full control over request and response format");

    Ok(())
}
