//! Canonical, sanitized error contracts.

mod contract;
pub use contract::{
    DiagnosticTextError, Error, ErrorContext, ErrorDetail, ErrorKind, MAX_RETRY_AFTER_HINT,
    PublicDiagnosticText, ResourceKind, ResponseDiagnostics, SensitiveErrorSource,
    SensitiveResponse,
};
