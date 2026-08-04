//! Error handling (re-export).
//!
//! Canonical errors coexist temporarily with legacy `siumai-spec` re-exports
//! while provider paths migrate to the next contract.

mod contract;
pub mod helpers;
pub mod policy;
pub use contract::{
    DiagnosticHeaderError, DiagnosticTextError, Error, ErrorContext, ErrorDetail, ErrorKind,
    PublicDiagnosticText, ResourceKind, ResponseDiagnostics, SafeResponseHeaders,
    SensitiveErrorSource, SensitiveResponse,
};
pub use helpers::*;
pub use policy::*;
pub use siumai_spec::error::*;
