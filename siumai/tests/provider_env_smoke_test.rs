#![cfg(any(
    feature = "openai",
    feature = "anthropic",
    feature = "google",
    feature = "deepseek",
    feature = "groq"
))]

use futures::StreamExt;
use siumai::Provider;
use siumai::prelude::unified::*;
use std::env;
use std::time::Duration;

#[cfg(feature = "openai")]
const OPENAI_DEFAULT_MODEL: &str = "gpt-4o-mini";

#[cfg(feature = "anthropic")]
const ANTHROPIC_DEFAULT_MODEL: &str = "claude-3-5-haiku-20241022";

#[cfg(feature = "google")]
const GEMINI_DEFAULT_MODEL: &str = "gemini-2.5-flash";

#[cfg(feature = "deepseek")]
const DEEPSEEK_DEFAULT_MODEL: &str = "deepseek-chat";

#[cfg(feature = "groq")]
const GROQ_DEFAULT_MODEL: &str = "llama-3.1-8b-instant";

#[cfg(feature = "anthropic")]
const ANTHROPIC_FALLBACK_MODELS: &[&str] = &[
    ANTHROPIC_DEFAULT_MODEL,
    "claude-3-5-sonnet-20241022",
    "claude-3-7-sonnet-20250219",
    "claude-sonnet-4-20250514",
    "claude-3-5-haiku-latest",
];

fn compact(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn prompt_messages() -> Vec<ChatMessage> {
    vec![
        system!("You are terse. Reply in under six words."),
        user!("Reply with SIUMAI_OK."),
    ]
}

fn stream_prompt_messages() -> Vec<ChatMessage> {
    vec![
        system!("You are terse. Reply in under six words."),
        user!("Reply with STREAM_OK."),
    ]
}

fn is_present(env_name: &str) -> bool {
    matches!(
        env::var(env_name),
        Ok(value) if !value.trim().is_empty() && value.trim() != "demo-key"
    )
}

fn is_truthy_env(env_name: &str) -> bool {
    matches!(
        env::var(env_name),
        Ok(value)
            if matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
    )
}

fn is_known_live_access_restriction(provider: &str, err: &str) -> bool {
    if is_truthy_env("SIUMAI_ENV_SMOKE_STRICT") {
        return false;
    }

    match provider {
        "gemini" => err.contains("User location is not supported for the API use."),
        "groq" => {
            let lower = err.to_ascii_lowercase();
            lower.contains("api error: 403") && lower.contains("groq api error: forbidden")
        }
        _ => false,
    }
}

fn unwrap_live_step(provider: &str, step: &str, result: Result<String, String>) -> Option<String> {
    match result {
        Ok(value) => Some(value),
        Err(err) if is_known_live_access_restriction(provider, &err) => {
            eprintln!("[skip] {provider} live smoke unavailable at step '{step}': {err}");
            None
        }
        Err(err) => panic!("{provider} {step}: {err}"),
    }
}

async fn generate_text<M>(model: &M, request: ChatRequest) -> Result<String, String>
where
    M: ChatCapability + ?Sized,
{
    let response = tokio::time::timeout(
        Duration::from_secs(90),
        text::generate(model, request, text::GenerateOptions::default()),
    )
    .await
    .map_err(|_| "request timed out after 90s".to_string())?
    .map_err(|err| err.to_string())?;

    let content = response.content_text().unwrap_or_default();
    if content.trim().is_empty() {
        Err("response content was empty".to_string())
    } else {
        Ok(compact(&content))
    }
}

async fn collect_stream_text<M>(model: &M, request: ChatRequest) -> Result<String, String>
where
    M: ChatCapability + ?Sized,
{
    let mut stream = tokio::time::timeout(
        Duration::from_secs(90),
        text::stream(model, request, text::StreamOptions::default()),
    )
    .await
    .map_err(|_| "stream setup timed out after 90s".to_string())?
    .map_err(|err| err.to_string())?;

    let mut collected = String::new();
    let mut saw_stream_end = false;

    while let Some(event) = tokio::time::timeout(Duration::from_secs(90), stream.next())
        .await
        .map_err(|_| "stream iteration timed out after 90s".to_string())?
    {
        match event.map_err(|err| err.to_string())? {
            ChatStreamEvent::ContentDelta { delta, .. } => collected.push_str(&delta),
            ChatStreamEvent::StreamEnd { .. } => {
                saw_stream_end = true;
                break;
            }
            _ => {}
        }
    }

    if !saw_stream_end {
        Err("stream ended without StreamEnd".to_string())
    } else if collected.trim().is_empty() {
        Err("stream content was empty".to_string())
    } else {
        Ok(compact(&collected))
    }
}

#[cfg(feature = "openai")]
fn openai_model() -> String {
    env::var("OPENAI_MODEL").unwrap_or_else(|_| OPENAI_DEFAULT_MODEL.to_string())
}

#[cfg(feature = "anthropic")]
fn anthropic_model_candidates() -> Vec<String> {
    let mut models = Vec::new();

    if let Ok(model) = env::var("ANTHROPIC_MODEL") {
        let trimmed = model.trim();
        if !trimmed.is_empty() {
            models.push(trimmed.to_string());
        }
    }

    for model in ANTHROPIC_FALLBACK_MODELS {
        if !models.iter().any(|existing| existing == model) {
            models.push((*model).to_string());
        }
    }

    models
}

#[cfg(feature = "google")]
fn gemini_model() -> String {
    env::var("GEMINI_MODEL").unwrap_or_else(|_| GEMINI_DEFAULT_MODEL.to_string())
}

#[cfg(feature = "deepseek")]
fn deepseek_model() -> String {
    env::var("DEEPSEEK_MODEL").unwrap_or_else(|_| DEEPSEEK_DEFAULT_MODEL.to_string())
}

#[cfg(feature = "groq")]
fn groq_model() -> String {
    env::var("GROQ_MODEL").unwrap_or_else(|_| GROQ_DEFAULT_MODEL.to_string())
}

#[cfg(feature = "openai")]
async fn openai_builder_generate(explicit_base_url: bool) -> Result<String, String> {
    let model = openai_model();
    let mut builder = Provider::openai()
        .model(&model)
        .max_tokens(24)
        .timeout(Duration::from_secs(90))
        .connect_timeout(Duration::from_secs(20));

    if explicit_base_url {
        let base_url = env::var("OPENAI_BASE_URL")
            .map_err(|_| "OPENAI_BASE_URL not present for explicit test".to_string())?;
        builder = builder.base_url(base_url);
    }

    let client = builder.build().await.map_err(|err| err.to_string())?;
    generate_text(&client, ChatRequest::new(prompt_messages())).await
}

#[cfg(feature = "openai")]
async fn openai_builder_stream(explicit_base_url: bool) -> Result<String, String> {
    let model = openai_model();
    let mut builder = Provider::openai()
        .model(&model)
        .max_tokens(24)
        .timeout(Duration::from_secs(90))
        .connect_timeout(Duration::from_secs(20));

    if explicit_base_url {
        let base_url = env::var("OPENAI_BASE_URL")
            .map_err(|_| "OPENAI_BASE_URL not present for explicit test".to_string())?;
        builder = builder.base_url(base_url);
    }

    let client = builder.build().await.map_err(|err| err.to_string())?;
    collect_stream_text(&client, ChatRequest::new(stream_prompt_messages())).await
}

#[cfg(feature = "openai")]
async fn openai_registry_generate() -> Result<String, String> {
    let model = registry::global()
        .language_model(&format!("openai:{}", openai_model()))
        .map_err(|err| format!("registry build failed: {err}"))?;
    generate_text(&model, ChatRequest::new(prompt_messages())).await
}

#[cfg(feature = "openai")]
async fn openai_registry_stream() -> Result<String, String> {
    let model = registry::global()
        .language_model(&format!("openai:{}", openai_model()))
        .map_err(|err| format!("registry build failed: {err}"))?;
    collect_stream_text(&model, ChatRequest::new(stream_prompt_messages())).await
}

#[cfg(feature = "anthropic")]
async fn anthropic_builder_generate(explicit_base_url: bool) -> Result<String, String> {
    let mut errors = Vec::new();

    for model in anthropic_model_candidates() {
        let mut builder = Provider::anthropic()
            .model(&model)
            .max_tokens(24)
            .timeout(Duration::from_secs(90))
            .connect_timeout(Duration::from_secs(20));

        if explicit_base_url {
            let base_url = env::var("ANTHROPIC_BASE_URL")
                .map_err(|_| "ANTHROPIC_BASE_URL not present for explicit test".to_string())?;
            builder = builder.base_url(base_url);
        }

        match builder.build().await {
            Ok(client) => match generate_text(&client, ChatRequest::new(prompt_messages())).await {
                Ok(content) => return Ok(format!("model={model} content={content}")),
                Err(err) => errors.push(format!("{model}: {err}")),
            },
            Err(err) => errors.push(format!("{model}: {err}")),
        }
    }

    Err(errors.join(" | "))
}

#[cfg(feature = "anthropic")]
async fn anthropic_builder_stream(explicit_base_url: bool) -> Result<String, String> {
    let mut errors = Vec::new();

    for model in anthropic_model_candidates() {
        let mut builder = Provider::anthropic()
            .model(&model)
            .max_tokens(24)
            .timeout(Duration::from_secs(90))
            .connect_timeout(Duration::from_secs(20));

        if explicit_base_url {
            let base_url = env::var("ANTHROPIC_BASE_URL")
                .map_err(|_| "ANTHROPIC_BASE_URL not present for explicit test".to_string())?;
            builder = builder.base_url(base_url);
        }

        match builder.build().await {
            Ok(client) => {
                match collect_stream_text(&client, ChatRequest::new(stream_prompt_messages())).await
                {
                    Ok(content) => return Ok(format!("model={model} content={content}")),
                    Err(err) => errors.push(format!("{model}: {err}")),
                }
            }
            Err(err) => errors.push(format!("{model}: {err}")),
        }
    }

    Err(errors.join(" | "))
}

#[cfg(feature = "anthropic")]
async fn anthropic_registry_generate() -> Result<String, String> {
    let mut errors = Vec::new();

    for model in anthropic_model_candidates() {
        let handle = match registry::global().language_model(&format!("anthropic:{model}")) {
            Ok(handle) => handle,
            Err(err) => {
                errors.push(format!("{model}: registry build failed: {err}"));
                continue;
            }
        };

        match generate_text(&handle, ChatRequest::new(prompt_messages())).await {
            Ok(content) => return Ok(format!("model={model} content={content}")),
            Err(err) => errors.push(format!("{model}: {err}")),
        }
    }

    Err(errors.join(" | "))
}

#[cfg(feature = "anthropic")]
async fn anthropic_registry_stream() -> Result<String, String> {
    let mut errors = Vec::new();

    for model in anthropic_model_candidates() {
        let handle = match registry::global().language_model(&format!("anthropic:{model}")) {
            Ok(handle) => handle,
            Err(err) => {
                errors.push(format!("{model}: registry build failed: {err}"));
                continue;
            }
        };

        match collect_stream_text(&handle, ChatRequest::new(stream_prompt_messages())).await {
            Ok(content) => return Ok(format!("model={model} content={content}")),
            Err(err) => errors.push(format!("{model}: {err}")),
        }
    }

    Err(errors.join(" | "))
}

#[cfg(feature = "google")]
async fn gemini_builder_generate() -> Result<String, String> {
    let model = gemini_model();
    let client = Provider::gemini()
        .model(&model)
        .max_tokens(24)
        .timeout(Duration::from_secs(90))
        .connect_timeout(Duration::from_secs(20))
        .build()
        .await
        .map_err(|err| err.to_string())?;

    generate_text(&client, ChatRequest::new(prompt_messages())).await
}

#[cfg(feature = "google")]
async fn gemini_builder_stream() -> Result<String, String> {
    let model = gemini_model();
    let client = Provider::gemini()
        .model(&model)
        .max_tokens(24)
        .timeout(Duration::from_secs(90))
        .connect_timeout(Duration::from_secs(20))
        .build()
        .await
        .map_err(|err| err.to_string())?;

    collect_stream_text(&client, ChatRequest::new(stream_prompt_messages())).await
}

#[cfg(feature = "google")]
async fn gemini_registry_generate() -> Result<String, String> {
    let model = registry::global()
        .language_model(&format!("gemini:{}", gemini_model()))
        .map_err(|err| format!("registry build failed: {err}"))?;
    generate_text(&model, ChatRequest::new(prompt_messages())).await
}

#[cfg(feature = "google")]
async fn gemini_registry_stream() -> Result<String, String> {
    let model = registry::global()
        .language_model(&format!("gemini:{}", gemini_model()))
        .map_err(|err| format!("registry build failed: {err}"))?;
    collect_stream_text(&model, ChatRequest::new(stream_prompt_messages())).await
}

#[cfg(feature = "deepseek")]
async fn deepseek_builder_generate() -> Result<String, String> {
    let model = deepseek_model();
    let client = Provider::deepseek()
        .model(&model)
        .max_tokens(24)
        .timeout(Duration::from_secs(90))
        .connect_timeout(Duration::from_secs(20))
        .build()
        .await
        .map_err(|err| err.to_string())?;

    generate_text(&client, ChatRequest::new(prompt_messages())).await
}

#[cfg(feature = "deepseek")]
async fn deepseek_builder_stream() -> Result<String, String> {
    let model = deepseek_model();
    let client = Provider::deepseek()
        .model(&model)
        .max_tokens(24)
        .timeout(Duration::from_secs(90))
        .connect_timeout(Duration::from_secs(20))
        .build()
        .await
        .map_err(|err| err.to_string())?;

    collect_stream_text(&client, ChatRequest::new(stream_prompt_messages())).await
}

#[cfg(feature = "deepseek")]
async fn deepseek_registry_generate() -> Result<String, String> {
    let model = registry::global()
        .language_model(&format!("deepseek:{}", deepseek_model()))
        .map_err(|err| format!("registry build failed: {err}"))?;
    generate_text(&model, ChatRequest::new(prompt_messages())).await
}

#[cfg(feature = "deepseek")]
async fn deepseek_registry_stream() -> Result<String, String> {
    let model = registry::global()
        .language_model(&format!("deepseek:{}", deepseek_model()))
        .map_err(|err| format!("registry build failed: {err}"))?;
    collect_stream_text(&model, ChatRequest::new(stream_prompt_messages())).await
}

#[cfg(feature = "groq")]
async fn groq_builder_generate() -> Result<String, String> {
    let model = groq_model();
    let client = Provider::groq()
        .model(&model)
        .max_tokens(24)
        .timeout(Duration::from_secs(90))
        .connect_timeout(Duration::from_secs(20))
        .build()
        .await
        .map_err(|err| err.to_string())?;

    generate_text(&client, ChatRequest::new(prompt_messages())).await
}

#[cfg(feature = "groq")]
async fn groq_builder_stream() -> Result<String, String> {
    let model = groq_model();
    let client = Provider::groq()
        .model(&model)
        .max_tokens(24)
        .timeout(Duration::from_secs(90))
        .connect_timeout(Duration::from_secs(20))
        .build()
        .await
        .map_err(|err| err.to_string())?;

    collect_stream_text(&client, ChatRequest::new(stream_prompt_messages())).await
}

#[cfg(feature = "groq")]
async fn groq_registry_generate() -> Result<String, String> {
    let model = registry::global()
        .language_model(&format!("groq:{}", groq_model()))
        .map_err(|err| format!("registry build failed: {err}"))?;
    generate_text(&model, ChatRequest::new(prompt_messages())).await
}

#[cfg(feature = "groq")]
async fn groq_registry_stream() -> Result<String, String> {
    let model = registry::global()
        .language_model(&format!("groq:{}", groq_model()))
        .map_err(|err| format!("registry build failed: {err}"))?;
    collect_stream_text(&model, ChatRequest::new(stream_prompt_messages())).await
}

#[cfg(feature = "openai")]
#[tokio::test]
#[ignore = "requires OPENAI_API_KEY and live network access"]
async fn openai_env_builder_and_registry_smoke() {
    if !is_present("OPENAI_API_KEY") {
        eprintln!("[skip] OPENAI_API_KEY not set, skipping live OpenAI env smoke");
        return;
    }

    let non_stream = openai_builder_generate(false)
        .await
        .expect("openai builder non-stream");
    assert!(
        non_stream.contains("SIUMAI_OK"),
        "expected SIUMAI_OK in builder response, got: {non_stream}"
    );

    let stream = openai_builder_stream(false)
        .await
        .expect("openai builder stream");
    assert!(
        stream.contains("STREAM_OK"),
        "expected STREAM_OK in builder stream, got: {stream}"
    );

    let registry_non_stream = openai_registry_generate()
        .await
        .expect("openai registry non-stream");
    assert!(
        registry_non_stream.contains("SIUMAI_OK"),
        "expected SIUMAI_OK in registry response, got: {registry_non_stream}"
    );

    let registry_stream = openai_registry_stream()
        .await
        .expect("openai registry stream");
    assert!(
        registry_stream.contains("STREAM_OK"),
        "expected STREAM_OK in registry stream, got: {registry_stream}"
    );

    if is_present("OPENAI_BASE_URL") {
        let explicit_non_stream = openai_builder_generate(true)
            .await
            .expect("openai explicit base non-stream");
        assert!(
            explicit_non_stream.contains("SIUMAI_OK"),
            "expected SIUMAI_OK via explicit OPENAI_BASE_URL, got: {explicit_non_stream}"
        );

        let explicit_stream = openai_builder_stream(true)
            .await
            .expect("openai explicit base stream");
        assert!(
            explicit_stream.contains("STREAM_OK"),
            "expected STREAM_OK via explicit OPENAI_BASE_URL, got: {explicit_stream}"
        );
    }
}

#[cfg(feature = "anthropic")]
#[tokio::test]
#[ignore = "requires ANTHROPIC_API_KEY and live network access"]
async fn anthropic_env_builder_and_registry_smoke() {
    if !is_present("ANTHROPIC_API_KEY") {
        eprintln!("[skip] ANTHROPIC_API_KEY not set, skipping live Anthropic env smoke");
        return;
    }

    let non_stream = anthropic_builder_generate(false)
        .await
        .expect("anthropic builder non-stream");
    assert!(
        non_stream.contains("SIUMAI_OK"),
        "expected SIUMAI_OK in builder response, got: {non_stream}"
    );

    let stream = anthropic_builder_stream(false)
        .await
        .expect("anthropic builder stream");
    assert!(
        stream.contains("STREAM_OK"),
        "expected STREAM_OK in builder stream, got: {stream}"
    );

    let registry_non_stream = anthropic_registry_generate()
        .await
        .expect("anthropic registry non-stream");
    assert!(
        registry_non_stream.contains("SIUMAI_OK"),
        "expected SIUMAI_OK in registry response, got: {registry_non_stream}"
    );

    let registry_stream = anthropic_registry_stream()
        .await
        .expect("anthropic registry stream");
    assert!(
        registry_stream.contains("STREAM_OK"),
        "expected STREAM_OK in registry stream, got: {registry_stream}"
    );

    if is_present("ANTHROPIC_BASE_URL") {
        let explicit_non_stream = anthropic_builder_generate(true)
            .await
            .expect("anthropic explicit base non-stream");
        assert!(
            explicit_non_stream.contains("SIUMAI_OK"),
            "expected SIUMAI_OK via explicit ANTHROPIC_BASE_URL, got: {explicit_non_stream}"
        );

        let explicit_stream = anthropic_builder_stream(true)
            .await
            .expect("anthropic explicit base stream");
        assert!(
            explicit_stream.contains("STREAM_OK"),
            "expected STREAM_OK via explicit ANTHROPIC_BASE_URL, got: {explicit_stream}"
        );
    }
}

#[cfg(feature = "google")]
#[tokio::test]
#[ignore = "requires GEMINI_API_KEY and live network access"]
async fn gemini_env_builder_and_registry_smoke() {
    if !is_present("GEMINI_API_KEY") {
        eprintln!("[skip] GEMINI_API_KEY not set, skipping live Gemini env smoke");
        return;
    }

    let Some(non_stream) = unwrap_live_step(
        "gemini",
        "builder non-stream",
        gemini_builder_generate().await,
    ) else {
        return;
    };
    assert!(
        non_stream.contains("SIUMAI_OK"),
        "expected SIUMAI_OK in Gemini builder response, got: {non_stream}"
    );

    let Some(stream) = unwrap_live_step("gemini", "builder stream", gemini_builder_stream().await)
    else {
        return;
    };
    assert!(
        stream.contains("STREAM_OK"),
        "expected STREAM_OK in Gemini builder stream, got: {stream}"
    );

    let Some(registry_non_stream) = unwrap_live_step(
        "gemini",
        "registry non-stream",
        gemini_registry_generate().await,
    ) else {
        return;
    };
    assert!(
        registry_non_stream.contains("SIUMAI_OK"),
        "expected SIUMAI_OK in Gemini registry response, got: {registry_non_stream}"
    );

    let Some(registry_stream) =
        unwrap_live_step("gemini", "registry stream", gemini_registry_stream().await)
    else {
        return;
    };
    assert!(
        registry_stream.contains("STREAM_OK"),
        "expected STREAM_OK in Gemini registry stream, got: {registry_stream}"
    );
}

#[cfg(feature = "deepseek")]
#[tokio::test]
#[ignore = "requires DEEPSEEK_API_KEY and live network access"]
async fn deepseek_env_builder_and_registry_smoke() {
    if !is_present("DEEPSEEK_API_KEY") {
        eprintln!("[skip] DEEPSEEK_API_KEY not set, skipping live DeepSeek env smoke");
        return;
    }

    let non_stream = deepseek_builder_generate()
        .await
        .expect("deepseek builder non-stream");
    assert!(
        non_stream.contains("SIUMAI_OK"),
        "expected SIUMAI_OK in DeepSeek builder response, got: {non_stream}"
    );

    let stream = deepseek_builder_stream()
        .await
        .expect("deepseek builder stream");
    assert!(
        stream.contains("STREAM_OK"),
        "expected STREAM_OK in DeepSeek builder stream, got: {stream}"
    );

    let registry_non_stream = deepseek_registry_generate()
        .await
        .expect("deepseek registry non-stream");
    assert!(
        registry_non_stream.contains("SIUMAI_OK"),
        "expected SIUMAI_OK in DeepSeek registry response, got: {registry_non_stream}"
    );

    let registry_stream = deepseek_registry_stream()
        .await
        .expect("deepseek registry stream");
    assert!(
        registry_stream.contains("STREAM_OK"),
        "expected STREAM_OK in DeepSeek registry stream, got: {registry_stream}"
    );
}

#[cfg(feature = "groq")]
#[tokio::test]
#[ignore = "requires GROQ_API_KEY and live network access"]
async fn groq_env_builder_and_registry_smoke() {
    if !is_present("GROQ_API_KEY") {
        eprintln!("[skip] GROQ_API_KEY not set, skipping live Groq env smoke");
        return;
    }

    let Some(non_stream) =
        unwrap_live_step("groq", "builder non-stream", groq_builder_generate().await)
    else {
        return;
    };
    assert!(
        non_stream.contains("SIUMAI_OK"),
        "expected SIUMAI_OK in Groq builder response, got: {non_stream}"
    );

    let Some(stream) = unwrap_live_step("groq", "builder stream", groq_builder_stream().await)
    else {
        return;
    };
    assert!(
        stream.contains("STREAM_OK"),
        "expected STREAM_OK in Groq builder stream, got: {stream}"
    );

    let Some(registry_non_stream) = unwrap_live_step(
        "groq",
        "registry non-stream",
        groq_registry_generate().await,
    ) else {
        return;
    };
    assert!(
        registry_non_stream.contains("SIUMAI_OK"),
        "expected SIUMAI_OK in Groq registry response, got: {registry_non_stream}"
    );

    let Some(registry_stream) =
        unwrap_live_step("groq", "registry stream", groq_registry_stream().await)
    else {
        return;
    };
    assert!(
        registry_stream.contains("STREAM_OK"),
        "expected STREAM_OK in Groq registry stream, got: {registry_stream}"
    );
}
