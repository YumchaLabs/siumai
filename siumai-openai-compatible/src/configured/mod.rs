mod codec_policy;
mod credentials;
pub(crate) mod execution;
mod mode;
mod model;
mod profile;
mod provider;

#[doc(hidden)]
pub use codec_policy::{
    ChatCodecPolicy, CompatibleStreamDecoder, PreparedChatCall, PreparedResponsesCall,
    ResponsesCodecPolicy,
};
pub use credentials::{
    BearerCredential, CredentialRequest, CredentialSourceError, DynamicCredentialSource,
    OpenAiCompatibleCredential,
};
pub use mode::OpenAiCompatibleApiMode;
pub use model::OpenAiCompatibleLanguageModel;
pub use profile::OpenAiCompatibleProfile;
pub use provider::{
    OpenAiCompatibleConfigError, OpenAiCompatibleProvider, OpenAiCompatibleProviderBuilder,
};
pub use siumai_protocol_openai::responses::ResponsesWireDialect;
