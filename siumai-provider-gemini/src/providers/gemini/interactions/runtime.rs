use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::LlmError;
use crate::execution::executors::common::{
    HttpBody, HttpExecutionConfig, HttpExecutionResult, execute_get_request, execute_json_request,
};
use crate::execution::http::headers::headermap_to_hashmap;
use crate::execution::wiring::HttpExecutionWiring;
use crate::provider_options::gemini::GoogleLanguageModelInteractionsOptions;
use crate::types::{
    ChatRequest, ChatResponse, HttpConfig, HttpRequestInfo, HttpResponseInfo, Warning,
};

use super::GoogleInteractionsLanguageModel;
use super::response::parse_interactions_response;

const INTERACTIONS_API_REVISION: &str = "2026-05-20";
const DEFAULT_POLLING_TIMEOUT_MS: u64 = 30 * 60 * 1000;
const DEFAULT_INITIAL_POLL_DELAY_MS: u64 = 1000;
const DEFAULT_MAX_POLL_DELAY_MS: u64 = 10_000;

pub(super) async fn execute_interactions_non_stream(
    model: &GoogleInteractionsLanguageModel,
    request: ChatRequest,
    http_client: reqwest::Client,
    retry_options: Option<crate::retry_api::RetryOptions>,
) -> Result<ChatResponse, LlmError> {
    let options = parse_interactions_runtime_options(&request)?;
    let prepared = model.prepare_request_body(&request, false)?;
    let request_body = serde_json::to_value(prepared.body).map_err(|error| {
        LlmError::ParseError(format!(
            "Failed to serialize google.interactions request body: {error}"
        ))
    })?;
    let request_info = serde_json::to_string(&request_body)
        .ok()
        .map(|body| HttpRequestInfo { body: Some(body) });

    let http_config =
        interactions_http_config(&model.config().http_config, request.http_config.as_ref());
    let execution_config = build_execution_config(model, http_client, retry_options).await;
    let post_url = interactions_url(model.base_url());
    let post_result = execute_json_request(
        &execution_config,
        &post_url,
        HttpBody::Json(request_body.clone()),
        Some(&http_config),
        false,
    )
    .await?;

    let terminal_result = if model.model_input().is_agent()
        && !is_terminal_response(&post_result.json)?
    {
        let interaction_id = interaction_id(&post_result.json).ok_or_else(|| {
            LlmError::ParseError(
                "google.interactions: cannot poll a background interaction without an id. The POST response did not include an interaction id."
                    .to_string(),
            )
        })?;
        poll_interaction_until_terminal(
            &execution_config,
            model.base_url(),
            interaction_id,
            &http_config,
            Duration::from_millis(
                options
                    .polling_timeout_ms
                    .unwrap_or(DEFAULT_POLLING_TIMEOUT_MS),
            ),
        )
        .await?
    } else {
        post_result
    };

    let mut generate_id = || model.generate_id();
    let mut response = parse_interactions_response(terminal_result.json.clone(), &mut generate_id)?;
    merge_warnings(&mut response, prepared.warnings);
    response.request = request_info;
    response.response = Some(HttpResponseInfo {
        timestamp: chrono::Utc::now(),
        model_id: response
            .model
            .clone()
            .or_else(|| Some(model.model_id().to_string())),
        headers: headermap_to_hashmap(&terminal_result.headers),
        body: Some(terminal_result.json),
    });
    Ok(response)
}

async fn build_execution_config(
    model: &GoogleInteractionsLanguageModel,
    http_client: reqwest::Client,
    retry_options: Option<crate::retry_api::RetryOptions>,
) -> HttpExecutionConfig {
    let ctx = super::super::context::build_context(model.config()).await;
    let mut wiring = HttpExecutionWiring::new("gemini", http_client, ctx)
        .with_interceptors(model.config().http_interceptors.clone())
        .with_retry_options(retry_options);

    if let Some(transport) = model.config().http_transport.clone() {
        wiring = wiring.with_transport(transport);
    }

    wiring.config(Arc::new(super::super::spec::GeminiSpec))
}

async fn poll_interaction_until_terminal(
    execution_config: &HttpExecutionConfig,
    base_url: &str,
    interaction_id: &str,
    http_config: &HttpConfig,
    timeout: Duration,
) -> Result<HttpExecutionResult, LlmError> {
    let started_at = Instant::now();
    let mut next_delay = Duration::from_millis(DEFAULT_INITIAL_POLL_DELAY_MS);
    let max_delay = Duration::from_millis(DEFAULT_MAX_POLL_DELAY_MS);
    let url = interaction_get_url(base_url, interaction_id);

    loop {
        if started_at.elapsed() >= timeout {
            return Err(LlmError::TimeoutError(format!(
                "google.interactions: timed out polling interaction {interaction_id} after {}ms.",
                timeout.as_millis()
            )));
        }

        let result = execute_get_request(execution_config, &url, Some(http_config)).await?;
        if is_terminal_response(&result.json)? {
            return Ok(result);
        }

        let remaining = timeout
            .checked_sub(started_at.elapsed())
            .unwrap_or(Duration::ZERO);
        if remaining.is_zero() {
            return Err(LlmError::TimeoutError(format!(
                "google.interactions: timed out polling interaction {interaction_id} after {}ms.",
                timeout.as_millis()
            )));
        }

        tokio::time::sleep(next_delay.min(remaining)).await;
        next_delay = (next_delay * 2).min(max_delay);
    }
}

fn interactions_url(base_url: &str) -> String {
    crate::utils::url::join_url(base_url.trim_end_matches('/'), "interactions")
}

fn interaction_get_url(base_url: &str, interaction_id: &str) -> String {
    crate::utils::url::join_url(
        base_url.trim_end_matches('/'),
        &format!("interactions/{}", urlencoding::encode(interaction_id)),
    )
}

fn parse_interactions_runtime_options(
    request: &ChatRequest,
) -> Result<GoogleLanguageModelInteractionsOptions, LlmError> {
    let Some(value) = request.provider_option("google") else {
        return Ok(GoogleLanguageModelInteractionsOptions::default());
    };
    if value.is_null() {
        return Ok(GoogleLanguageModelInteractionsOptions::default());
    }
    serde_json::from_value(value.clone()).map_err(|error| {
        LlmError::InvalidParameter(format!(
            "invalid google.interactions provider options: {error}"
        ))
    })
}

fn interactions_http_config(
    config_http: &HttpConfig,
    request_http: Option<&HttpConfig>,
) -> HttpConfig {
    let mut out = request_http.cloned().unwrap_or_else(HttpConfig::empty);
    if !has_header(&config_http.headers, "api-revision")
        && !has_header(&out.headers, "api-revision")
    {
        out.headers.insert(
            "Api-Revision".to_string(),
            INTERACTIONS_API_REVISION.to_string(),
        );
    }
    out
}

fn has_header(headers: &std::collections::HashMap<String, String>, name: &str) -> bool {
    headers.keys().any(|key| key.eq_ignore_ascii_case(name))
}

fn interaction_id(raw: &serde_json::Value) -> Option<&str> {
    raw.get("id")
        .and_then(serde_json::Value::as_str)
        .filter(|value| !value.is_empty())
}

fn is_terminal_response(raw: &serde_json::Value) -> Result<bool, LlmError> {
    let status = raw
        .get("status")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| {
            LlmError::ParseError("google.interactions response is missing status".to_string())
        })?;
    Ok(matches!(
        status,
        "completed" | "failed" | "cancelled" | "incomplete"
    ))
}

fn merge_warnings(response: &mut ChatResponse, warnings: Vec<Warning>) {
    if warnings.is_empty() {
        return;
    }
    response
        .warnings
        .get_or_insert_with(Vec::new)
        .extend(warnings);
}
