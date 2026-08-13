//! Explicitly unstable bidirectional provider-session contracts.
//!
//! These APIs are intentionally separate from the six stable one-shot families.

mod session;

pub use session::{
    ProviderSession, SessionCloseMetadata, SessionCloseOrigin, SessionCloseRequest, SessionFailure,
    SessionIncoming, SessionLineageId, SessionLineageIdError, SessionTerminal,
    SessionTransportKind,
};
