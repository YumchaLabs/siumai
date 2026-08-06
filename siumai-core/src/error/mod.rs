//! Canonical, sanitized error contracts.

mod contract;
pub use contract::{
    DiagnosticHeaderError, DiagnosticTextError, Error, ErrorContext, ErrorDetail, ErrorKind,
    PublicDiagnosticText, ResourceKind, ResponseDiagnostics, SafeResponseHeaders,
    SensitiveErrorSource, SensitiveResponse,
};
