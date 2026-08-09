//! Provider-owned native result and stream carriers for OpenAI Responses.

use std::fmt;
use std::pin::Pin;
use std::task::{Context, Poll};

use futures_util::{Stream, StreamExt};
use siumai_core::stream::established_stream;
use siumai_core::{
    Cancellation, Error, ErrorKind, LanguageResponse, LanguageStream, LanguageStreamEvent,
    StreamTerminal,
};
use siumai_protocol_openai::responses::{ResponseWire, ResponsesStreamEvent};

/// A native OpenAI Responses result paired with its portable projection.
#[derive(Clone)]
pub struct OpenAiResponsesResponse {
    native: ResponseWire,
    portable: LanguageResponse,
}

impl fmt::Debug for OpenAiResponsesResponse {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OpenAiResponsesResponse")
            .field("id", &self.native.id)
            .field("model", &self.native.model)
            .field("status", &self.native.status)
            .field("native_output_items", &self.native.output.len())
            .field("portable_status", &self.portable.status())
            .finish_non_exhaustive()
    }
}

impl OpenAiResponsesResponse {
    pub(crate) fn new(native: ResponseWire, portable: LanguageResponse) -> Self {
        Self { native, portable }
    }

    pub fn native(&self) -> &ResponseWire {
        &self.native
    }

    pub fn portable(&self) -> &LanguageResponse {
        &self.portable
    }

    pub fn into_portable(self) -> LanguageResponse {
        self.portable
    }

    pub fn into_parts(self) -> (ResponseWire, LanguageResponse) {
        (self.native, self.portable)
    }
}

/// One native Responses event and the immutable portable events derived from it.
pub struct OpenAiResponsesStreamFrame {
    native: ResponsesStreamEvent,
    portable_events: Vec<LanguageStreamEvent>,
    canonical_terminal_response: Option<ResponseWire>,
}

impl fmt::Debug for OpenAiResponsesStreamFrame {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OpenAiResponsesStreamFrame")
            .field("native", &self.native)
            .field("portable_event_count", &self.portable_events.len())
            .field("terminal", &self.is_terminal())
            .field(
                "has_canonical_terminal_response",
                &self.canonical_terminal_response.is_some(),
            )
            .finish()
    }
}

impl OpenAiResponsesStreamFrame {
    pub(crate) fn new(
        native: ResponsesStreamEvent,
        portable_events: Vec<LanguageStreamEvent>,
        canonical_terminal_response: Option<ResponseWire>,
    ) -> Self {
        Self {
            native,
            portable_events,
            canonical_terminal_response,
        }
    }

    pub fn native(&self) -> &ResponsesStreamEvent {
        &self.native
    }

    pub fn portable_events(&self) -> &[LanguageStreamEvent] {
        &self.portable_events
    }

    /// Return the reconstructed canonical terminal response, when this frame settles the turn.
    ///
    /// This may contain output items reconstructed from earlier events when the
    /// raw terminal event is abbreviated. [`Self::native`] always remains the
    /// exact provider event and is never rewritten.
    pub fn canonical_terminal_response(&self) -> Option<&ResponseWire> {
        self.canonical_terminal_response.as_ref()
    }

    pub fn terminal(&self) -> Option<&StreamTerminal> {
        self.portable_events
            .iter()
            .find_map(LanguageStreamEvent::terminal)
    }

    pub fn is_terminal(&self) -> bool {
        self.terminal().is_some()
    }

    pub fn into_portable_events(self) -> Vec<LanguageStreamEvent> {
        self.portable_events
    }

    pub fn into_parts(
        self,
    ) -> (
        ResponsesStreamEvent,
        Vec<LanguageStreamEvent>,
        Option<ResponseWire>,
    ) {
        (
            self.native,
            self.portable_events,
            self.canonical_terminal_response,
        )
    }
}

/// Established native OpenAI Responses stream.
///
/// Once established, the stream settles exactly once with either a terminal frame or an error.
/// Dropping it cancels only the child operation owned by this stream.
pub struct OpenAiResponsesStream {
    inner: Pin<Box<dyn Stream<Item = Result<OpenAiResponsesStreamFrame, Error>> + Send + 'static>>,
    cancellation: Cancellation,
}

impl fmt::Debug for OpenAiResponsesStream {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OpenAiResponsesStream")
            .field("is_cancelled", &self.cancellation.is_cancelled())
            .finish_non_exhaustive()
    }
}

impl Stream for OpenAiResponsesStream {
    type Item = Result<OpenAiResponsesStreamFrame, Error>;

    fn poll_next(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.inner.as_mut().poll_next(context)
    }
}

impl Drop for OpenAiResponsesStream {
    fn drop(&mut self) {
        self.cancellation.cancel();
    }
}

impl OpenAiResponsesStream {
    pub(crate) fn established<S>(
        cancellation: Cancellation,
        cancellation_error: Error,
        source: S,
    ) -> Self
    where
        S: Stream<Item = Result<OpenAiResponsesStreamFrame, Error>> + Send + 'static,
    {
        let stream_cancellation = cancellation.child();
        let lifecycle_cancellation = stream_cancellation.clone();
        let inner = Box::pin(async_stream::stream! {
            let mut source = Box::pin(source);
            loop {
                tokio::select! {
                    biased;
                    _ = lifecycle_cancellation.cancelled() => {
                        yield Err(cancellation_error);
                        break;
                    }
                    item = source.next() => {
                        match item {
                            Some(Ok(frame)) => {
                                let terminal = frame.is_terminal();
                                yield Ok(frame);
                                if terminal {
                                    break;
                                }
                            }
                            Some(Err(error)) => {
                                yield Err(error);
                                break;
                            }
                            None => {
                                yield Err(Error::unexpected_eof());
                                break;
                            }
                        }
                    }
                }
            }
        });
        Self {
            inner,
            cancellation: stream_cancellation,
        }
    }

    /// Consume the native stream into the standard portable language stream.
    ///
    /// This is a projection of the same HTTP response and decoder state. It never issues another
    /// provider request.
    pub fn into_portable(self) -> LanguageStream {
        established_stream(Cancellation::new(), move |_| {
            async_stream::try_stream! {
                let mut stream = self;
                while let Some(item) = stream.next().await {
                    match item {
                        Ok(frame) => {
                            for event in frame.into_portable_events() {
                                yield event;
                            }
                        }
                        Err(error) if error.kind() == ErrorKind::Cancelled => {
                            yield LanguageStreamEvent::Terminal(StreamTerminal::Cancelled {
                                reason: "call cancelled".to_string(),
                                response: None,
                            });
                            return;
                        }
                        Err(error) => Err(error)?,
                    }
                }
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use futures_util::stream;

    use super::*;

    #[tokio::test]
    async fn native_stream_settles_once_on_eof_and_cancellation() {
        let mut eof = OpenAiResponsesStream::established(
            Cancellation::new(),
            Error::cancelled("cancelled"),
            stream::empty(),
        );
        assert_eq!(
            eof.next().await.unwrap().unwrap_err().kind(),
            ErrorKind::UnexpectedEof
        );
        assert!(eof.next().await.is_none());

        let parent = Cancellation::new();
        let native = OpenAiResponsesStream::established(
            parent.clone(),
            Error::cancelled("cancelled"),
            stream::pending(),
        );
        let mut portable = native.into_portable();
        parent.cancel();
        assert!(matches!(
            portable.next().await,
            Some(LanguageStreamEvent::Terminal(
                StreamTerminal::Cancelled { .. }
            ))
        ));
        assert!(portable.next().await.is_none());
    }
}
