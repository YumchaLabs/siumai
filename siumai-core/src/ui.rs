//! AI SDK-style UI message helpers.

mod conversion;
mod types;
mod validation;

pub use conversion::{
    convert_to_chat_request, convert_to_chat_request_with, convert_to_chat_request_with_tooling,
    convert_to_model_messages, convert_to_model_messages_with,
    convert_to_model_messages_with_tooling,
};
pub use types::{
    ConvertUiMessagesOptions, SafeValidateUIMessagesResult, SafeValidateUiMessagesResult,
    UiMessageError, UiSchemaValidator, ValidateUiMessagesSchemaOptions,
};
pub use validation::{
    safe_validate_ui_messages, safe_validate_ui_messages_with_schemas, validate_ui_messages,
    validate_ui_messages_with_schemas,
};

#[cfg(test)]
mod tests;
