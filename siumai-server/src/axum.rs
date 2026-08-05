use std::convert::Infallible;

use axum::response::Sse;
use axum::response::sse::{Event, KeepAlive};
use futures::{Stream, StreamExt};
use siumai_core::LanguageStream;
use siumai_runtime::RunStream;

use crate::GatewayEvent;

/// Project a shared runtime stream to an Axum SSE response.
pub fn run_sse(
    stream: RunStream,
) -> Sse<impl Stream<Item = Result<Event, Infallible>> + Send + 'static> {
    event_stream(stream.map(GatewayEvent::from_run))
}

/// Project a plain one-model stream to an Axum SSE response.
pub fn language_sse(
    stream: LanguageStream,
) -> Sse<impl Stream<Item = Result<Event, Infallible>> + Send + 'static> {
    event_stream(stream.map(GatewayEvent::from_language))
}

fn event_stream<S>(stream: S) -> Sse<impl Stream<Item = Result<Event, Infallible>> + Send + 'static>
where
    S: Stream<Item = GatewayEvent> + Send + 'static,
{
    let stream = stream.map(|event| {
        let event_name = event.kind();
        let data = serde_json::to_string(&event).unwrap_or_else(|_| {
            "{\"type\":\"server.serialization-failed\",\"data\":null}".to_string()
        });
        Ok(Event::default().event(event_name).data(data))
    });
    Sse::new(stream).keep_alive(KeepAlive::default())
}
