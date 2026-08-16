//! Provider-owned native result and stream carriers for OpenAI Responses.

use std::fmt;
use std::pin::Pin;
use std::task::{Context, Poll};

use futures_util::{Stream, StreamExt};
use siumai_core::stream::established_stream;
use siumai_core::{
    Cancellation, Error, ErrorKind, LanguageCallError, LanguageResponse, LanguageStream,
    LanguageStreamEvent, LanguageTermination, StreamTerminal,
};
use siumai_openai_compatible::extension::v2::SseStream;
use siumai_protocol_openai::responses::{
    ResponseWire, ResponsesReplayStatus, ResponsesStreamEvent,
};

/// A native OpenAI Responses result paired with its portable projection.
pub struct OpenAiResponsesResponse {
    native: ResponseWire,
    portable: Result<LanguageResponse, LanguageCallError>,
}

impl fmt::Debug for OpenAiResponsesResponse {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OpenAiResponsesResponse")
            .field("id", &self.native.id)
            .field("model", &self.native.model)
            .field("status", &self.native.status)
            .field("native_output_items", &self.native.output.len())
            .field(
                "portable_outcome",
                &match &self.portable {
                    Ok(response) => match response.termination() {
                        LanguageTermination::Completed(_) => "completed",
                        LanguageTermination::Incomplete(_) => "incomplete",
                        _ => "other",
                    },
                    Err(error) if error.kind() == ErrorKind::Cancelled => "cancelled",
                    Err(_) => "failed",
                },
            )
            .field(
                "has_partial_output",
                &self
                    .portable
                    .as_ref()
                    .err()
                    .is_some_and(|error| error.partial().is_some()),
            )
            .finish_non_exhaustive()
    }
}

impl OpenAiResponsesResponse {
    pub(crate) fn new(
        native: ResponseWire,
        portable: Result<LanguageResponse, LanguageCallError>,
    ) -> Self {
        Self { native, portable }
    }

    pub fn native(&self) -> &ResponseWire {
        &self.native
    }

    pub fn portable(&self) -> Result<&LanguageResponse, &LanguageCallError> {
        self.portable.as_ref()
    }

    pub fn into_portable(self) -> Result<LanguageResponse, LanguageCallError> {
        self.portable
    }

    pub fn into_parts(self) -> (ResponseWire, Result<LanguageResponse, LanguageCallError>) {
        (self.native, self.portable)
    }
}

/// One native Responses event and the immutable portable events derived from it.
pub struct OpenAiResponsesStreamFrame {
    native: ResponsesStreamEvent,
    portable_events: Vec<LanguageStreamEvent>,
    canonical_terminal_response: Option<ResponseWire>,
    replay_status: ResponsesReplayStatus,
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
            .field("replay_status", &self.replay_status)
            .finish()
    }
}

impl OpenAiResponsesStreamFrame {
    pub(crate) fn new(
        native: ResponsesStreamEvent,
        portable_events: Vec<LanguageStreamEvent>,
        canonical_terminal_response: Option<ResponseWire>,
        replay_status: ResponsesReplayStatus,
    ) -> Self {
        Self {
            native,
            portable_events,
            canonical_terminal_response,
            replay_status,
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

    /// Return whether the native turn remains safe to replay after stream reconciliation.
    pub const fn replay_status(&self) -> ResponsesReplayStatus {
        self.replay_status
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
        ResponsesReplayStatus,
    ) {
        (
            self.native,
            self.portable_events,
            self.canonical_terminal_response,
            self.replay_status,
        )
    }
}

/// Established native OpenAI Responses stream.
///
/// Once established, the stream settles exactly once with either a terminal frame or an error.
/// Dropping it cancels only the child operation owned by this stream.
pub struct OpenAiResponsesStream {
    inner: SseStream<OpenAiResponsesStreamFrame>,
}

impl fmt::Debug for OpenAiResponsesStream {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OpenAiResponsesStream")
            .field("inner", &self.inner)
            .finish_non_exhaustive()
    }
}

impl Stream for OpenAiResponsesStream {
    type Item = Result<OpenAiResponsesStreamFrame, Error>;

    fn poll_next(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Pin::new(&mut self.inner).poll_next(context)
    }
}

impl OpenAiResponsesStream {
    pub(crate) fn new(inner: SseStream<OpenAiResponsesStreamFrame>) -> Self {
        Self { inner }
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
                    let frame = item?;
                    for event in frame.into_portable_events() {
                        yield event;
                    }
                }
            }
        })
    }
}
