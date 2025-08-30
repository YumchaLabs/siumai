//! Default Configuration Values
//!
//! This module centralizes all default values used throughout the Siumai SDK.
//! Having defaults in one place makes them easier to maintain, document, and adjust.

use std::time::Duration;

/// HTTP client default configurations
pub mod http {
    use super::*;

    /// Default request timeout for HTTP requests
    ///
    /// Set to 60 seconds to accommodate large language models that may take
    /// 10-20 seconds to respond, plus network latency and proxy delays.
    pub const REQUEST_TIMEOUT: Duration = Duration::from_secs(60);

    /// Default connection timeout for establishing HTTP connections
    ///
    /// Set to 10 seconds which is sufficient for most network conditions
    /// while not being too aggressive.
    pub const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

    /// Default User-Agent string for HTTP requests
    pub const USER_AGENT: &str = "siumai/0.1.0";

    /// Default maximum number of idle connections per host
    pub const MAX_IDLE_PER_HOST: usize = 10;

    /// Default maximum total idle connections
    pub const MAX_IDLE_TOTAL: usize = 100;

    /// Default keep-alive timeout for HTTP connections
    pub const KEEP_ALIVE_TIMEOUT: Duration = Duration::from_secs(90);

    /// Default TCP keep-alive interval
    pub const TCP_KEEP_ALIVE: Duration = Duration::from_secs(60);
}

/// Timeout configurations for different use cases
pub mod timeouts {
    use super::*;

    /// Fast response timeout for interactive applications
    ///
    /// Suitable for small to medium models (7B-32B parameters)
    /// that typically respond within 1-5 seconds.
    pub const FAST: Duration = Duration::from_secs(30);

    /// Standard timeout for production applications
    ///
    /// Suitable for most models including large ones (72B-235B parameters)
    /// that may take 5-15 seconds to respond.
    pub const STANDARD: Duration = Duration::from_secs(60);

    /// Extended timeout for complex operations
    ///
    /// Suitable for very large models, reasoning models, or batch processing
    /// that may take 15-60 seconds to complete.
    pub const EXTENDED: Duration = Duration::from_secs(120);

    /// Long-running timeout for batch processing
    ///
    /// Suitable for complex reasoning tasks, long document processing,
    /// or operations that may take several minutes.
    pub const LONG_RUNNING: Duration = Duration::from_secs(300);

    /// Maximum reasonable timeout
    ///
    /// Upper bound for any operation to prevent indefinite hanging.
    pub const MAXIMUM: Duration = Duration::from_secs(600);
}

/// Model-specific timeout recommendations
pub mod model_timeouts {
    use super::*;

    /// Timeout for small models (7B-14B parameters)
    ///
    /// These models typically respond very quickly (1-3 seconds)
    pub const SMALL_MODELS: Duration = timeouts::FAST;

    /// Timeout for medium models (32B-72B parameters)
    ///
    /// These models typically respond within 3-8 seconds
    pub const MEDIUM_MODELS: Duration = timeouts::STANDARD;

    /// Timeout for large models (235B+ parameters)
    ///
    /// These models may take 8-20 seconds to respond
    pub const LARGE_MODELS: Duration = timeouts::EXTENDED;

    /// Timeout for reasoning models (DeepSeek R1, QwQ, etc.)
    ///
    /// These models perform complex reasoning and may take 10-30 seconds
    pub const REASONING_MODELS: Duration = timeouts::EXTENDED;

    /// Timeout for code generation models
    ///
    /// Code generation may involve complex analysis and take 5-20 seconds
    pub const CODE_MODELS: Duration = timeouts::EXTENDED;

    /// Timeout for multimodal models (vision + text)
    ///
    /// Processing images along with text may take additional time
    pub const MULTIMODAL_MODELS: Duration = timeouts::EXTENDED;

    /// Timeout for embedding models
    ///
    /// Embedding generation is typically fast (1-5 seconds)
    pub const EMBEDDING_MODELS: Duration = timeouts::FAST;

    /// Timeout for reranking models
    ///
    /// Reranking is typically fast (1-5 seconds)
    pub const RERANK_MODELS: Duration = timeouts::FAST;
}

/// Rate limiting and retry defaults
pub mod rate_limiting {
    use super::*;

    /// Default maximum number of retry attempts
    pub const MAX_RETRIES: u32 = 3;

    /// Default base delay for exponential backoff (in milliseconds)
    pub const BASE_RETRY_DELAY_MS: u64 = 1000;

    /// Default maximum delay for exponential backoff
    pub const MAX_RETRY_DELAY: Duration = Duration::from_secs(30);

    /// Default jitter factor for retry delays (0.0 to 1.0)
    pub const RETRY_JITTER: f64 = 0.1;

    /// Default requests per minute for rate limiting
    pub const REQUESTS_PER_MINUTE: u32 = 60;

    /// Default burst size for rate limiting
    pub const BURST_SIZE: u32 = 10;
}

/// Streaming and real-time defaults
pub mod streaming {
    use super::*;

    /// Default buffer size for streaming responses
    pub const BUFFER_SIZE: usize = 8192;

    /// Default timeout for streaming chunk reception
    pub const CHUNK_TIMEOUT: Duration = Duration::from_secs(30);

    /// Default keep-alive interval for Server-Sent Events
    pub const SSE_KEEP_ALIVE: Duration = Duration::from_secs(15);

    /// Default maximum time to wait for stream start
    pub const STREAM_START_TIMEOUT: Duration = Duration::from_secs(10);
}

/// Performance and optimization defaults
pub mod performance {
    /// Default batch size for batch processing
    pub const BATCH_SIZE: usize = 10;

    /// Default concurrency limit for parallel requests
    pub const CONCURRENCY_LIMIT: usize = 5;

    /// Default cache size for response caching
    pub const CACHE_SIZE: usize = 1000;

    /// Default cache TTL (time to live)
    pub const CACHE_TTL_SECONDS: u64 = 3600; // 1 hour
}

/// Logging and tracing defaults
pub mod logging {
    /// Default log level
    pub const LOG_LEVEL: &str = "info";

    /// Default maximum log file size (in bytes)
    pub const MAX_LOG_FILE_SIZE: u64 = 100 * 1024 * 1024; // 100MB

    /// Default number of log files to keep in rotation
    pub const LOG_FILE_ROTATION_COUNT: u32 = 5;

    /// Default maximum size for logged request/response bodies
    pub const MAX_BODY_LOG_SIZE: usize = 1024; // 1KB

    /// Default sampling rate for tracing (0.0 to 1.0)
    pub const TRACING_SAMPLING_RATE: f64 = 1.0;
}

/// Provider-specific defaults
pub mod providers {
    use super::*;

    /// OpenAI-specific defaults
    pub mod openai {
        use super::*;

        /// Default base URL for OpenAI API
        pub const BASE_URL: &str = "https://api.openai.com/v1";

        /// Default model for OpenAI
        pub const DEFAULT_MODEL: &str = "gpt-4o-mini";

        /// Default timeout for OpenAI requests
        pub const TIMEOUT: Duration = timeouts::STANDARD;
    }

    /// Anthropic-specific defaults
    pub mod anthropic {
        use super::*;

        /// Default base URL for Anthropic API
        pub const BASE_URL: &str = "https://api.anthropic.com";

        /// Default model for Anthropic
        pub const DEFAULT_MODEL: &str = "claude-3-5-haiku-20241022";

        /// Default timeout for Anthropic requests
        pub const TIMEOUT: Duration = timeouts::STANDARD;
    }

    /// SiliconFlow-specific defaults
    pub mod siliconflow {
        use super::*;

        /// Default base URL for SiliconFlow API
        pub const BASE_URL: &str = "https://api.siliconflow.cn/v1";

        /// Default model for SiliconFlow
        pub const DEFAULT_MODEL: &str = "deepseek-ai/DeepSeek-V3.1";

        /// Default timeout for SiliconFlow requests (longer due to large models)
        pub const TIMEOUT: Duration = timeouts::EXTENDED;
    }

    /// Groq-specific defaults
    pub mod groq {
        use super::*;

        /// Default base URL for Groq API
        pub const BASE_URL: &str = "https://api.groq.com/openai/v1";

        /// Default model for Groq
        pub const DEFAULT_MODEL: &str = "llama-3.3-70b-versatile";

        /// Default timeout for Groq requests (fast inference)
        pub const TIMEOUT: Duration = timeouts::FAST;
    }
}

/// Model parameter defaults
pub mod model_params {
    /// Default temperature for text generation
    pub const TEMPERATURE: f32 = 0.7;

    /// Default top-p for nucleus sampling
    pub const TOP_P: f32 = 0.9;

    /// Default top-k for top-k sampling
    pub const TOP_K: u32 = 50;

    /// Default maximum tokens to generate
    pub const MAX_TOKENS: u32 = 2048;

    /// Default presence penalty
    pub const PRESENCE_PENALTY: f32 = 0.0;

    /// Default frequency penalty
    pub const FREQUENCY_PENALTY: f32 = 0.0;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timeout_hierarchy() {
        // Ensure timeouts are in logical order
        assert!(timeouts::FAST < timeouts::STANDARD);
        assert!(timeouts::STANDARD < timeouts::EXTENDED);
        assert!(timeouts::EXTENDED < timeouts::LONG_RUNNING);
        assert!(timeouts::LONG_RUNNING < timeouts::MAXIMUM);
    }

    #[test]
    fn test_http_defaults() {
        assert_eq!(http::REQUEST_TIMEOUT, Duration::from_secs(60));
        assert_eq!(http::CONNECT_TIMEOUT, Duration::from_secs(10));
        assert_eq!(http::USER_AGENT, "siumai/0.1.0");
    }

    #[test]
    fn test_model_timeout_assignments() {
        // Small models should use fast timeout
        assert_eq!(model_timeouts::SMALL_MODELS, timeouts::FAST);

        // Large models should use extended timeout
        assert_eq!(model_timeouts::LARGE_MODELS, timeouts::EXTENDED);

        // Reasoning models need extended time
        assert_eq!(model_timeouts::REASONING_MODELS, timeouts::EXTENDED);
    }
}
