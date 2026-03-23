use super::OpenAiClient;
use crate::error::LlmError;
use crate::streaming::{ChatStream, ChatStreamHandle};
use crate::traits::ChatCapability;
use crate::types::{ChatMessage, ChatRequest, ChatResponse, Tool};
use async_trait::async_trait;
use std::sync::Arc;

fn wrap_handle_with_responses_remote_cancel(
    http_client: OpenAiClient,
    handle: ChatStreamHandle,
) -> ChatStreamHandle {
    let ChatStreamHandle { stream, cancel } = handle;

    let response_id: Arc<tokio::sync::Mutex<Option<String>>> =
        Arc::new(tokio::sync::Mutex::new(None));
    let response_id_for_stream = Arc::clone(&response_id);

    let (done_tx, done_rx) = tokio::sync::oneshot::channel::<()>();
    struct DoneOnDrop(Option<tokio::sync::oneshot::Sender<()>>);
    impl Drop for DoneOnDrop {
        fn drop(&mut self) {
            if let Some(tx) = self.0.take() {
                let _ = tx.send(());
            }
        }
    }

    let mut inner = stream;
    let wrapped = async_stream::stream! {
        use futures_util::StreamExt;
        let _done = DoneOnDrop(Some(done_tx));
        while let Some(item) = inner.next().await {
            if let Ok(crate::types::ChatStreamEvent::Custom { event_type, data }) = item.as_ref()
                && event_type == "openai:response-metadata"
                && let Some(id) = data.get("id").and_then(|v| v.as_str())
            {
                let mut g = response_id_for_stream.lock().await;
                if g.is_none() {
                    *g = Some(id.to_string());
                }
            }
            yield item;
        }
    };

    let cancel_for_task = cancel.clone();
    let response_id_for_task = Arc::clone(&response_id);
    tokio::spawn(async move {
        tokio::select! {
            _ = cancel_for_task.cancelled() => {},
            _ = done_rx => {
                // If the stream ended because the caller cancelled, still attempt remote cancel.
                if !cancel_for_task.is_cancelled() {
                    return;
                }
            }
        }

        let id = { response_id_for_task.lock().await.clone() };
        let Some(id) = id else { return };
        if let Err(e) = http_client.responses_cancel(&id).await {
            tracing::warn!(
                target: "siumai::openai::client",
                error = %e,
                response_id = %id,
                "Remote cancel request failed"
            );
        }
    });

    ChatStreamHandle {
        stream: Box::pin(wrapped),
        cancel,
    }
}

impl OpenAiClient {
    fn prepare_chat_request(
        &self,
        request: ChatRequest,
        stream: bool,
    ) -> Result<ChatRequest, LlmError> {
        let request = crate::utils::chat_request::normalize_chat_request(
            request,
            crate::utils::chat_request::ChatRequestDefaults::new(&self.common_params)
                .with_provider_options_map(&self.default_provider_options_map),
            stream,
        );
        if request.common_params.model.trim().is_empty() {
            return Err(LlmError::InvalidParameter(
                "OpenAI request requires a model".to_string(),
            ));
        }
        Ok(request)
    }

    async fn chat_with_tools_inner(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatResponse, LlmError> {
        let mut builder = ChatRequest::builder()
            .messages(messages)
            .common_params(self.common_params.clone());
        if let Some(ts) = tools {
            builder = builder.tools(ts);
        }
        self.chat_request_via_spec(builder.build()).await
    }

    // Stream chat via ProviderSpec (unified path)
    pub(super) async fn chat_stream_via_spec(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatStream, LlmError> {
        let mut builder = ChatRequest::builder()
            .messages(messages)
            .common_params(self.common_params.clone())
            .stream(true);
        if let Some(ts) = tools {
            builder = builder.tools(ts);
        }
        self.chat_stream_request_via_spec(builder.build()).await
    }

    // Execute chat (non-stream) via ProviderSpec with a fully-formed ChatRequest
    pub(super) async fn chat_request_via_spec(
        &self,
        request: ChatRequest,
    ) -> Result<ChatResponse, LlmError> {
        use crate::execution::executors::chat::ChatExecutor;
        let request = self.prepare_chat_request(request, false)?;
        let exec = self.build_chat_executor(&request);
        ChatExecutor::execute(&*exec, request).await
    }

    // Execute chat (stream) via ProviderSpec with a fully-formed ChatRequest
    pub(super) async fn chat_stream_request_via_spec(
        &self,
        request: ChatRequest,
    ) -> Result<ChatStream, LlmError> {
        use crate::execution::executors::chat::ChatExecutor;
        let request = self.prepare_chat_request(request, true)?;
        let exec = self.build_chat_executor(&request);
        ChatExecutor::execute_stream(&*exec, request).await
    }
}

#[async_trait]
impl ChatCapability for OpenAiClient {
    /// Chat with tools implementation
    async fn chat_with_tools(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatResponse, LlmError> {
        self.chat_with_tools_inner(messages, tools).await
    }

    /// Streaming chat with tools
    async fn chat_stream(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatStream, LlmError> {
        self.chat_stream_via_spec(messages, tools).await
    }

    async fn chat_stream_with_cancel(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatStreamHandle, LlmError> {
        let this = self.clone();
        let handle = crate::utils::cancel::make_cancellable_stream_handle_from_future(async move {
            this.chat_stream_via_spec(messages, tools).await
        });

        Ok(wrap_handle_with_responses_remote_cancel(
            self.clone_without_http_transport(),
            handle,
        ))
    }

    async fn chat_request(&self, request: ChatRequest) -> Result<ChatResponse, LlmError> {
        self.chat_request_via_spec(request).await
    }

    async fn chat_stream_request(&self, request: ChatRequest) -> Result<ChatStream, LlmError> {
        self.chat_stream_request_via_spec(request).await
    }

    async fn chat_stream_request_with_cancel(
        &self,
        request: ChatRequest,
    ) -> Result<ChatStreamHandle, LlmError> {
        let this = self.clone();
        let handle = crate::utils::cancel::make_cancellable_stream_handle_from_future(async move {
            this.chat_stream_request_via_spec(request).await
        });

        Ok(wrap_handle_with_responses_remote_cancel(
            self.clone_without_http_transport(),
            handle,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::http::interceptor::{HttpInterceptor, HttpRequestContext};
    use futures_util::StreamExt;
    use std::sync::{Arc, Mutex};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    #[test]
    fn prepare_chat_request_for_stream_fills_missing_common_params_defaults() {
        let cfg = crate::providers::openai::OpenAiConfig::new("test-key")
            .with_model("gpt-default")
            .with_temperature(0.7)
            .with_max_tokens(256)
            .with_top_p(0.9);
        let client =
            crate::providers::openai::OpenAiClient::from_config(cfg).expect("from_config ok");

        let request = ChatRequest::builder()
            .messages(vec![ChatMessage::user("hi").build()])
            .build();

        let prepared = client
            .prepare_chat_request(request, true)
            .expect("prepare stream request");

        assert!(prepared.stream);
        assert_eq!(prepared.common_params.model, "gpt-default");
        assert_eq!(prepared.common_params.temperature, Some(0.7));
        assert_eq!(prepared.common_params.max_tokens, Some(256));
        assert_eq!(prepared.common_params.top_p, Some(0.9));
    }

    #[test]
    fn prepare_chat_request_for_non_stream_preserves_explicit_common_params() {
        let cfg = crate::providers::openai::OpenAiConfig::new("test-key")
            .with_model("gpt-default")
            .with_temperature(0.7)
            .with_max_tokens(256)
            .with_top_p(0.9);
        let client =
            crate::providers::openai::OpenAiClient::from_config(cfg).expect("from_config ok");

        let request = ChatRequest::builder()
            .model("gpt-explicit")
            .temperature(0.2)
            .messages(vec![ChatMessage::user("hi").build()])
            .stream(true)
            .build();

        let prepared = client
            .prepare_chat_request(request, false)
            .expect("prepare non-stream request");

        assert!(!prepared.stream);
        assert_eq!(prepared.common_params.model, "gpt-explicit");
        assert_eq!(prepared.common_params.temperature, Some(0.2));
        assert_eq!(prepared.common_params.max_tokens, Some(256));
        assert_eq!(prepared.common_params.top_p, Some(0.9));
    }

    #[tokio::test]
    async fn chat_request_uses_client_common_params_defaults_without_request_overrides() {
        struct Capture(Arc<Mutex<Option<serde_json::Value>>>);

        impl HttpInterceptor for Capture {
            fn on_before_send(
                &self,
                _ctx: &HttpRequestContext,
                _builder: reqwest::RequestBuilder,
                body: &serde_json::Value,
                _headers: &reqwest::header::HeaderMap,
            ) -> Result<reqwest::RequestBuilder, LlmError> {
                *self.0.lock().expect("capture lock") = Some(body.clone());
                Err(LlmError::InvalidParameter("stop".into()))
            }
        }

        let cfg = crate::providers::openai::OpenAiConfig::new("test-key")
            .with_model("gpt-4-test")
            .with_temperature(0.7)
            .with_max_tokens(256);
        let captured = Arc::new(Mutex::new(None));
        let client = crate::providers::openai::OpenAiClient::from_config(cfg)
            .expect("from_config ok")
            .with_http_interceptors(vec![Arc::new(Capture(Arc::clone(&captured)))]);

        let request = ChatRequest::new(vec![ChatMessage::user("hi").build()]);
        let err = client
            .chat_request(request)
            .await
            .expect_err("interceptor should abort before http send");
        match err {
            LlmError::InvalidParameter(message) => assert_eq!(message, "stop"),
            other => panic!("unexpected error: {other:?}"),
        }

        let body = captured
            .lock()
            .expect("capture lock")
            .clone()
            .expect("captured body");
        assert_eq!(
            body.get("model").and_then(|v| v.as_str()),
            Some("gpt-4-test")
        );
        assert_eq!(body.get("temperature").and_then(|v| v.as_f64()), Some(0.7));
        assert_eq!(
            body.get("max_output_tokens").and_then(|v| v.as_u64()),
            Some(256)
        );
    }

    async fn write_chunk(tcp: &mut tokio::net::TcpStream, bytes: &[u8]) -> tokio::io::Result<()> {
        let header = format!("{:x}\r\n", bytes.len());
        tcp.write_all(header.as_bytes()).await?;
        tcp.write_all(bytes).await?;
        tcp.write_all(b"\r\n").await?;
        tcp.flush().await
    }

    #[tokio::test]
    async fn openai_client_remote_cancel_calls_http_cancel_endpoint() {
        let (cancel_seen_tx, cancel_seen_rx) = tokio::sync::oneshot::channel::<()>();

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let server = tokio::spawn(async move {
            // 1) Streaming `/responses` request (SSE over HTTP).
            let (mut tcp_stream, _) = listener.accept().await.unwrap();

            // Read request headers (and ignore body).
            let mut buf = Vec::<u8>::new();
            let mut tmp = [0u8; 1024];
            loop {
                let n = tcp_stream.read(&mut tmp).await.unwrap();
                if n == 0 {
                    break;
                }
                buf.extend_from_slice(&tmp[..n]);
                if buf.windows(4).any(|w| w == b"\r\n\r\n") {
                    break;
                }
            }

            let sse_created = concat!(
                "event: response.created\n",
                "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_1\",\"model\":\"gpt-test\",\"created_at\":0}}\n\n",
            );

            let headers = concat!(
                "HTTP/1.1 200 OK\r\n",
                "content-type: text/event-stream\r\n",
                "transfer-encoding: chunked\r\n",
                "\r\n"
            );
            tcp_stream.write_all(headers.as_bytes()).await.unwrap();
            write_chunk(&mut tcp_stream, sse_created.as_bytes())
                .await
                .unwrap();

            // 2) HTTP cancel request: `POST /v1/responses/resp_1/cancel`.
            let (mut tcp_cancel, _) = listener.accept().await.unwrap();
            let mut buf = Vec::<u8>::new();
            loop {
                let n = tcp_cancel.read(&mut tmp).await.unwrap();
                if n == 0 {
                    break;
                }
                buf.extend_from_slice(&tmp[..n]);
                if buf.windows(4).any(|w| w == b"\r\n\r\n") {
                    break;
                }
            }
            let header_end = buf
                .windows(4)
                .position(|w| w == b"\r\n\r\n")
                .map(|p| p + 4)
                .unwrap_or(buf.len());
            let header_str = String::from_utf8_lossy(&buf[..header_end]);
            let first_line = header_str.lines().next().unwrap_or("");
            assert!(
                first_line.starts_with("POST /v1/responses/resp_1/cancel "),
                "unexpected http request line: {first_line}"
            );

            let body = b"{}";
            let resp = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n",
                body.len()
            );
            tcp_cancel.write_all(resp.as_bytes()).await.unwrap();
            tcp_cancel.write_all(body).await.unwrap();
            let _ = tcp_cancel.shutdown().await;

            let _ = cancel_seen_tx.send(());
            let _ = tcp_stream.shutdown().await;
        });

        let client =
            crate::providers::openai::OpenAiBuilder::new(crate::builder::BuilderBase::default())
                .api_key("test")
                .base_url(format!("http://{addr}/v1"))
                .model("gpt-test")
                .build()
                .await
                .unwrap();

        let ChatStreamHandle { mut stream, cancel } = client
            .chat_stream_with_cancel(vec![ChatMessage::user("hi").build()], None)
            .await
            .unwrap();

        // Wait until we see response metadata (so the response id is known), then cancel.
        while let Some(item) = stream.next().await {
            let ev = item.unwrap();
            if let crate::types::ChatStreamEvent::Custom { event_type, .. } = ev {
                if event_type == "openai:response-metadata" {
                    break;
                }
            }
        }

        cancel.cancel();
        drop(stream);

        tokio::time::timeout(std::time::Duration::from_secs(2), cancel_seen_rx)
            .await
            .expect("expected HTTP cancel request")
            .expect("cancel channel closed");

        server.await.unwrap();
    }
}
