//! Telemetry Wrapper for Streaming
//!
//! Wraps ChatStream to automatically emit telemetry events when streams complete.

use futures::Stream;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use crate::error::LlmError;
use crate::streaming::{ChatStream, ChatStreamEvent, StreamProcessor};
use crate::types::{ChatMessage, FinishReason};

/// Telemetry wrapper for ChatStream
struct TelemetryStreamWrapper {
    inner: ChatStream,
    processor: StreamProcessor,
    telemetry_config: Arc<crate::telemetry::TelemetryConfig>,
    trace_id: String,
    provider_id: String,
    model: String,
    input_messages: Vec<ChatMessage>,
    start_time: std::time::SystemTime,
    last_finish_reason: Option<FinishReason>,
    telemetry_sent: bool,
}

impl Stream for TelemetryStreamWrapper {
    type Item = Result<ChatStreamEvent, LlmError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let poll_result = Pin::new(&mut self.inner).poll_next(cx);

        match &poll_result {
            Poll::Ready(Some(Ok(event))) => {
                // Process the event to accumulate data
                self.processor.process_event(event.clone());

                // Capture finish_reason from StreamEnd event
                if let ChatStreamEvent::StreamEnd { response } = event {
                    self.last_finish_reason = response.finish_reason.clone();
                }
            }
            Poll::Ready(None) | Poll::Ready(Some(Err(_))) => {
                // Stream ended - send telemetry event if not already sent
                if self.telemetry_config.enabled && !self.telemetry_sent {
                    self.telemetry_sent = true;

                    let final_response = self
                        .processor
                        .build_final_response_with_finish_reason(self.last_finish_reason.clone());
                    let duration = std::time::SystemTime::now()
                        .duration_since(self.start_time)
                        .ok();

                    let mut gen_event = crate::telemetry::events::GenerationEvent::new(
                        uuid::Uuid::new_v4().to_string(),
                        self.trace_id.clone(),
                        self.provider_id.clone(),
                        self.model.clone(),
                    );

                    if self.telemetry_config.record_inputs {
                        gen_event = gen_event.with_input(self.input_messages.clone());
                    }

                    if self.telemetry_config.record_outputs {
                        gen_event = gen_event.with_output(final_response.clone());
                    }

                    if self.telemetry_config.record_usage {
                        if let Some(usage) = &final_response.usage {
                            gen_event = gen_event.with_usage(usage.clone());
                        }
                    }

                    if let Some(reason) = &final_response.finish_reason {
                        gen_event = gen_event.with_finish_reason(reason.clone());
                    }

                    if let Some(dur) = duration {
                        gen_event = gen_event.with_duration(dur);
                    }

                    // Spawn a task to send telemetry event asynchronously
                    tokio::spawn(async move {
                        crate::telemetry::emit(
                            crate::telemetry::events::TelemetryEvent::Generation(gen_event),
                        )
                        .await;
                    });
                }
            }
            _ => {}
        }

        poll_result
    }
}

/// Wrap a ChatStream with telemetry tracking
///
/// This function wraps a ChatStream and automatically sends a GenerationEvent
/// when the stream completes, including all collected data (messages, usage, etc.)
///
/// # Arguments
/// * `stream` - The ChatStream to wrap
/// * `telemetry_config` - Telemetry configuration
/// * `trace_id` - Trace ID for correlation
/// * `provider_id` - Provider identifier
/// * `model` - Model name
/// * `input_messages` - Input messages for telemetry
///
/// # Returns
/// A wrapped ChatStream that emits telemetry events
pub fn wrap_stream_with_telemetry(
    stream: ChatStream,
    telemetry_config: Arc<crate::telemetry::TelemetryConfig>,
    trace_id: String,
    provider_id: String,
    model: String,
    input_messages: Vec<ChatMessage>,
) -> ChatStream {
    let wrapper = TelemetryStreamWrapper {
        inner: stream,
        processor: StreamProcessor::new(),
        telemetry_config,
        trace_id,
        provider_id,
        model,
        input_messages,
        start_time: std::time::SystemTime::now(),
        last_finish_reason: None,
        telemetry_sent: false,
    };

    Box::pin(wrapper)
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;
    use std::sync::{Arc, Mutex};

    struct CaptureExporter(Arc<Mutex<Vec<crate::telemetry::TelemetryEvent>>>);
    #[async_trait::async_trait]
    impl crate::telemetry::TelemetryExporter for CaptureExporter {
        async fn export(
            &self,
            event: &crate::telemetry::TelemetryEvent,
        ) -> Result<(), crate::error::LlmError> {
            self.0.lock().unwrap().push(event.clone());
            Ok(())
        }
    }

    #[tokio::test]
    async fn emits_generation_event_on_stream_end() {
        // Prepare a short stream with a final StreamEnd event
        let events = vec![
            Ok(ChatStreamEvent::ContentDelta {
                delta: "hi".into(),
                index: None,
            }),
            Ok(ChatStreamEvent::StreamEnd {
                response: crate::types::ChatResponse::empty_with_finish_reason(
                    crate::types::FinishReason::Stop,
                ),
            }),
        ];
        let stream: ChatStream = Box::pin(futures::stream::iter(events));

        let sink = Arc::new(Mutex::new(Vec::new()));
        // Ensure clean state
        crate::telemetry::clear_exporters().await;
        crate::telemetry::add_exporter(Box::new(CaptureExporter(sink.clone()))).await;

        let cfg = Arc::new(
            crate::telemetry::TelemetryConfig::builder()
                .enabled(true)
                .record_inputs(true)
                .record_outputs(true)
                .record_usage(true)
                .build(),
        );

        let wrapped = wrap_stream_with_telemetry(
            stream,
            cfg,
            uuid::Uuid::new_v4().to_string(),
            "test-provider".into(),
            "test-model".into(),
            vec![crate::types::ChatMessage::user("hello").build()],
        );

        // Drain the stream to completion
        futures::pin_mut!(wrapped);
        while let Some(_ev) = wrapped.next().await {}

        // Allow spawned task to run
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        let captured = sink.lock().unwrap();
        assert!(
            captured
                .iter()
                .any(|e| matches!(e, crate::telemetry::TelemetryEvent::Generation(_)))
        );
    }
}
