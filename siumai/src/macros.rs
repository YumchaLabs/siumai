//! Public macros for building chat messages and requests
//!
//! These macros are exported at crate root via `#[macro_export]` and
//! remain source-compatible with previous versions.

/// Creates a user message
///
/// Always returns `ChatMessage` for consistent type handling.
#[macro_export]
macro_rules! user {
    // Simple text message - returns ChatMessage directly
    ($content:expr) => {
        $crate::types::ChatMessage {
            role: $crate::types::MessageRole::User,
            content: $crate::types::MessageContent::Text($content.into()),
            metadata: $crate::types::MessageMetadata::default(),
        }
    };
    // Message with cache control - returns ChatMessage via builder
    ($content:expr, cache: $cache:expr) => {
        $crate::types::ChatMessage::user($content)
            .cache_control($cache)
            .build()
    };
}

/// Creates a user message builder for complex messages
///
/// Use this when you need to add images, cache control, or other complex features.
#[macro_export]
macro_rules! user_builder {
    ($content:expr) => {
        $crate::types::ChatMessage::user($content)
    };
}

/// Creates a system message
///
/// Always returns `ChatMessage` for consistent type handling.
#[macro_export]
macro_rules! system {
    // Simple text message - returns ChatMessage directly
    ($content:expr) => {
        $crate::types::ChatMessage {
            role: $crate::types::MessageRole::System,
            content: $crate::types::MessageContent::Text($content.into()),
            metadata: $crate::types::MessageMetadata::default(),
        }
    };
    // Message with cache control - returns ChatMessage via builder
    ($content:expr, cache: $cache:expr) => {
        $crate::types::ChatMessage::system($content)
            .cache_control($cache)
            .build()
    };
}

/// Creates an assistant message
///
/// Always returns `ChatMessage` for consistent type handling.
#[macro_export]
macro_rules! assistant {
    // Simple text message - returns ChatMessage directly
    ($content:expr) => {
        $crate::types::ChatMessage {
            role: $crate::types::MessageRole::Assistant,
            content: $crate::types::MessageContent::Text($content.into()),
            metadata: $crate::types::MessageMetadata::default(),
        }
    }; // Message with tool calls arm removed; use assistant_with_content instead
}

/// Creates a tool result message
///
/// Returns `ChatMessage` with tool result content.
#[macro_export]
macro_rules! tool {
    ($content:expr, id: $id:expr, name: $name:expr) => {
        $crate::types::ChatMessage {
            role: $crate::types::MessageRole::Tool,
            content: $crate::types::MessageContent::MultiModal(vec![
                $crate::types::ContentPart::tool_result_text($id, $name, $content),
            ]),
            metadata: $crate::types::MessageMetadata::default(),
        }
    };
}

/// Multimodal user message macro
///
/// Always returns `ChatMessage` for consistent type handling.
#[macro_export]
macro_rules! user_with_image {
    ($text:expr, $image_url:expr) => {
        $crate::types::ChatMessage::user($text)
            .with_image($image_url.to_string(), None)
            .build()
    };
    ($text:expr, $image_url:expr, detail: $detail:expr) => {
        $crate::types::ChatMessage::user($text)
            .with_image($image_url.to_string(), Some($detail.to_string()))
            .build()
    };
}

/// Creates a collection of messages with convenient syntax
#[macro_export]
macro_rules! messages {
    ($($msg:expr),* $(,)?) => {
        vec![$($msg),*]
    };
}

/// Creates a conversation with alternating user and assistant messages
#[macro_export]
macro_rules! conversation {
    ($($user:expr => $assistant:expr),* $(,)?) => {
        {
            let mut msgs = Vec::new();
            $(
                msgs.push($crate::user!($user));
                msgs.push($crate::assistant!($assistant));
            )*
            msgs
        }
    };
}

/// Creates a conversation with a system prompt
#[macro_export]
macro_rules! conversation_with_system {
    ($system:expr, $($user:expr => $assistant:expr),* $(,)?) => {
        {
            let mut msgs = vec![$crate::system!($system)];
            $(
                msgs.push($crate::user!($user));
                msgs.push($crate::assistant!($assistant));
            )*
            msgs
        }
    };
}

/// Creates a quick chat request with a single user message
#[macro_export]
macro_rules! quick_chat {
    ($msg:expr) => {
        vec![$crate::user!($msg)]
    };
    (system: $system:expr, $msg:expr) => {
        vec![$crate::system!($system), $crate::user!($msg)]
    };
}
