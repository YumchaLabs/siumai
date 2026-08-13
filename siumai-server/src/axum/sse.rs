use std::convert::Infallible;

use axum::body::Body;
use axum::response::sse::{Event, KeepAlive};
use axum::response::{IntoResponse, Response, Sse};
use futures::{Stream, StreamExt};
use siumai_core::LanguageStream;
use siumai_runtime::{RunEvent, RunStream};

use crate::{GatewayEvent, GatewayPolicy, GatewayProjectionError, ServerTrustContext};

use super::response::apply_gateway_headers;

/// Project a canonical one-model stream with the default safe server policy.
pub fn language_sse(stream: LanguageStream) -> Response<Body> {
    language_sse_with_policy(stream, GatewayPolicy::default())
}

/// Project a canonical one-model stream with an explicit server policy.
pub fn language_sse_with_policy(stream: LanguageStream, policy: GatewayPolicy) -> Response<Body> {
    sse_response(stream, None, policy, GatewayEvent::from_language)
}

/// Project a canonical trusted tool-run stream with the default server policy.
pub fn run_sse(stream: RunStream, trust: ServerTrustContext) -> Response<Body> {
    run_sse_with_policy(stream, trust, GatewayPolicy::default())
}

/// Project a canonical trusted tool-run stream with an explicit server policy.
pub fn run_sse_with_policy(
    stream: RunStream,
    trust: ServerTrustContext,
    policy: GatewayPolicy,
) -> Response<Body> {
    let route = trust.route().clone();
    sse_response(stream, Some(trust), policy, move |event, policy| {
        if let RunEvent::Started { target } = &event
            && target
                .route()
                .is_some_and(|target_route| target_route != &route)
        {
            return Err(GatewayProjectionError::TrustRouteMismatch);
        }
        GatewayEvent::from_run(event, policy)
    })
}

fn sse_response<S, I, F>(
    stream: S,
    trust: Option<ServerTrustContext>,
    policy: GatewayPolicy,
    project: F,
) -> Response<Body>
where
    S: Stream<Item = I> + Send + 'static,
    I: Send + 'static,
    F: Fn(I, &GatewayPolicy) -> Result<GatewayEvent, GatewayProjectionError>
        + Send
        + Sync
        + 'static,
{
    let limit = policy.limits().sse_event_bytes();
    let idle_timeout = policy.stream().idle_timeout();
    let projection_policy = policy.clone();
    let downstream = async_stream::stream! {
        let mut upstream = Box::pin(stream);
        let mut terminal_seen = false;

        while !terminal_seen {
            let next = match idle_timeout {
                Some(timeout) => match tokio::time::timeout(timeout, upstream.next()).await {
                    Ok(next) => next,
                    Err(_) => {
                        yield Ok::<_, Infallible>(fallback_event(
                            GatewayEvent::stream_failed(
                                "stream_idle_timeout",
                                "gateway stream idle timeout",
                            ),
                            limit,
                        ));
                        break;
                    }
                },
                None => upstream.next().await,
            };

            let Some(item) = next else {
                yield Ok::<_, Infallible>(fallback_event(
                    GatewayEvent::stream_failed(
                        "unexpected_eof",
                        "gateway stream ended without a terminal event",
                    ),
                    limit,
                ));
                break;
            };

            let event = match project(item, &projection_policy) {
                Ok(event) => event,
                Err(error) => {
                    yield Ok::<_, Infallible>(fallback_event(
                        GatewayEvent::projection_failed(&error),
                        limit,
                    ));
                    break;
                }
            };
            terminal_seen = event.is_terminal();
            match encode_event(event, limit) {
                Ok(event) => yield Ok::<_, Infallible>(event),
                Err(code) => {
                    yield Ok::<_, Infallible>(fallback_event(
                        GatewayEvent::stream_failed(code, "server projection failed"),
                        limit,
                    ));
                    break;
                }
            }
        }
    };

    let sse = Sse::new(downstream);
    let mut response = match policy.stream().keep_alive_interval() {
        Some(interval) => sse
            .keep_alive(KeepAlive::new().interval(interval).text("keep-alive"))
            .into_response(),
        None => sse.into_response(),
    };
    apply_gateway_headers(&mut response, trust.as_ref(), &policy);
    response
}

fn encode_event(event: GatewayEvent, limit: usize) -> Result<Event, &'static str> {
    let kind = event.kind();
    match serde_json::to_string(&event) {
        Ok(data) if data.len() <= limit => Ok(Event::default().event(kind).data(data)),
        Ok(_) => Err("sse_event_too_large"),
        Err(_) => Err("projection_serialization_failed"),
    }
}

fn fallback_event(event: GatewayEvent, limit: usize) -> Event {
    match encode_event(event, limit) {
        Ok(event) => event,
        Err(_) => Event::default().event("server.projection.failed").data(
            r#"{"type":"server.projection.failed","data":{"error":{"code":"projection_failed","message":"server projection failed"}}}"#,
        ),
    }
}
