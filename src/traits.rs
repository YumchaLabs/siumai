//! Core Trait Definitions
//!
//! Defines the core capability traits for the LLM library, using a capability separation design pattern.
//! This follows the design principles of trait separation, parameter sharing, and provider-specific extensions.

use async_trait::async_trait;
use std::collections::HashMap;

use crate::error::LlmError;
use crate::stream::ChatStream;
use crate::stream::ChatStreamHandle;
use crate::types::*;

/// Core chat capability trait - the most fundamental LLM capability.
///
/// This trait provides only the essential chat functionality that all LLM providers must implement.
/// It follows the single responsibility principle by focusing solely on core chat operations.
///
/// # API References
/// - OpenAI: <https://platform.openai.com/docs/guides/tools>
/// - Anthropic: <https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview>
/// - xAI: <https://docs.x.ai/docs/guides/function-calling>
#[async_trait]
pub trait ChatCapability: Send + Sync {
    /// Sends a chat request to the provider with a sequence of messages.
    ///
    /// # Arguments
    /// * `messages` - The conversation history as a list of chat messages
    ///
    /// # Returns
    /// The provider's response or an error
    ///
    /// # Default Implementation
    /// By default, this calls `chat_with_tools` with no tools.
    async fn chat(&self, messages: Vec<ChatMessage>) -> Result<ChatResponse, LlmError> {
        self.chat_with_tools(messages, None).await
    }

    /// Sends a chat request to the provider with a sequence of messages and tools.
    ///
    /// # Arguments
    /// * `messages` - The conversation history as a list of chat messages
    /// * `tools` - Optional list of tools to use in the chat
    ///
    /// # Returns
    /// The provider's response or an error
    async fn chat_with_tools(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatResponse, LlmError>;

    /// Sends a streaming chat request to the provider.
    ///
    /// # Arguments
    /// * `messages` - The conversation history as a list of chat messages
    /// * `tools` - Optional list of tools to use in the chat
    ///
    /// # Returns
    /// A stream of chat events
    async fn chat_stream(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatStream, LlmError>;

    /// Sends a streaming chat request and returns a first-class cancellation handle
    /// together with the stream. Calling `cancel()` will stop the underlying HTTP
    /// stream immediately so providers stop generating tokens.
    ///
    /// Default implementation wraps `chat_stream` with a cancellable adapter without
    /// requiring providers to change their implementations.
    async fn chat_stream_with_cancel(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatStreamHandle, LlmError> {
        let stream = self.chat_stream(messages, tools).await?;
        let (cancellable, cancel) = crate::utils::cancel::make_cancellable_stream(stream);
        Ok(ChatStreamHandle {
            stream: cancellable,
            cancel,
        })
    }
}

/// Extended chat capabilities providing convenience methods and advanced features.
///
/// This trait extends the core ChatCapability with useful convenience methods
/// that are commonly needed but not essential for basic chat functionality.
/// It follows the interface segregation principle by separating optional features.
#[async_trait]
pub trait ChatExtensions: ChatCapability {
    /// Send a chat request using a unified `ChatRequest` container.
    ///
    /// This default implementation forwards to `chat_with_tools(messages, tools)` and
    /// ignores advanced per-request options (e.g., provider_params/http_config) at this layer.
    /// Use provider-specific clients for full control when needed.
    async fn chat_request(&self, request: ChatRequest) -> Result<ChatResponse, LlmError> {
        self.chat_with_tools(request.messages, request.tools).await
    }

    /// Send a streaming chat request using a unified `ChatRequest` container.
    ///
    /// This default implementation forwards to `chat_stream(messages, tools)` and
    /// ignores advanced per-request options (e.g., provider_params/http_config) at this layer.
    /// Use provider-specific clients for full control when needed.
    async fn chat_stream_request(&self, request: ChatRequest) -> Result<ChatStream, LlmError> {
        self.chat_stream(request.messages, request.tools).await
    }

    /// Send a streaming chat request and get a cancellation handle.
    async fn chat_stream_request_with_cancel(
        &self,
        request: ChatRequest,
    ) -> Result<ChatStreamHandle, LlmError> {
        self.chat_stream_with_cancel(request.messages, request.tools)
            .await
    }
    /// Send a chat request with retry using the unified retry facade.
    ///
    /// This wraps `chat_with_tools` with `retry_api::retry_with` for convenient retries.
    async fn chat_with_retry(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
        options: crate::retry_api::RetryOptions,
    ) -> Result<ChatResponse, LlmError> {
        use crate::retry_api;

        // Capture clones for retry closure
        let msgs = messages;
        let tls = tools;

        retry_api::retry_with(
            || {
                let m = msgs.clone();
                let t = tls.clone();
                async move { self.chat_with_tools(m, t).await }
            },
            options,
        )
        .await
    }

    /// Get current memory contents if provider supports memory.
    ///
    /// # Returns
    /// Optional list of messages representing the current memory state
    async fn memory_contents(&self) -> Result<Option<Vec<ChatMessage>>, LlmError> {
        Ok(None)
    }

    /// Summarizes a conversation history into a concise 2-3 sentence summary.
    ///
    /// # Arguments
    /// * `messages` - The conversation messages to summarize
    ///
    /// # Returns
    /// A string containing the summary
    async fn summarize_history(&self, messages: Vec<ChatMessage>) -> Result<String, LlmError> {
        let prompt = format!(
            "Summarize in 2-3 sentences:\n{}",
            messages
                .iter()
                .map(|m| format!("{:?}: {}", m.role, m.content_text().unwrap_or("")))
                .collect::<Vec<_>>()
                .join("\n")
        );

        let request_messages = vec![ChatMessage::user(prompt).build()];
        let response = self.chat(request_messages).await?;

        response
            .content_text()
            .ok_or_else(|| LlmError::InternalError("No text in summary response".to_string()))
            .map(std::string::ToString::to_string)
    }

    /// Simple text completion - just send a prompt and get a response.
    ///
    /// # Arguments
    /// * `prompt` - The text prompt to send
    ///
    /// # Returns
    /// The response text
    ///
    /// # Example
    /// ```rust,no_run
    /// # use siumai::prelude::*;
    /// # #[tokio::main]
    /// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let client = quick_openai().await?;
    /// let response = client.ask("What is the capital of France?".to_string()).await?;
    /// println!("{}", response);
    /// # Ok(())
    /// # }
    /// ```
    async fn ask(&self, prompt: String) -> Result<String, LlmError> {
        let message = ChatMessage::user(prompt).build();
        let response = self.chat(vec![message]).await?;
        response
            .content_text()
            .ok_or_else(|| LlmError::InternalError("No text in response".to_string()))
            .map(std::string::ToString::to_string)
    }

    /// Simple text completion with retry.
    async fn ask_with_retry(
        &self,
        prompt: String,
        options: crate::retry_api::RetryOptions,
    ) -> Result<String, LlmError> {
        let message = ChatMessage::user(prompt).build();
        let response = self.chat_with_retry(vec![message], None, options).await?;
        response
            .content_text()
            .ok_or_else(|| LlmError::InternalError("No text in response".to_string()))
            .map(std::string::ToString::to_string)
    }

    /// Simple system-prompted completion.
    ///
    /// # Arguments
    /// * `system_prompt` - The system instruction
    /// * `user_prompt` - The user prompt
    ///
    /// # Returns
    /// The response text
    ///
    /// # Example
    /// ```rust,no_run
    /// # use siumai::prelude::*;
    /// # #[tokio::main]
    /// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let client = quick_openai().await?;
    /// let response = client.ask_with_system(
    ///     "You are a helpful assistant that responds in JSON".to_string(),
    ///     "List 3 colors".to_string()
    /// ).await?;
    /// # Ok(())
    /// # }
    /// ```
    async fn ask_with_system(
        &self,
        system_prompt: String,
        user_prompt: String,
    ) -> Result<String, LlmError> {
        let messages = vec![
            ChatMessage::system(system_prompt).build(),
            ChatMessage::user(user_prompt).build(),
        ];
        let response = self.chat(messages).await?;
        response
            .content_text()
            .ok_or_else(|| LlmError::InternalError("No text in response".to_string()))
            .map(std::string::ToString::to_string)
    }

    /// Continue a conversation with a new user message.
    ///
    /// # Arguments
    /// * `conversation` - Existing conversation messages
    /// * `new_message` - New user message to add
    ///
    /// # Returns
    /// The response and updated conversation
    ///
    /// # Example
    /// ```rust,no_run
    /// # use siumai::prelude::*;
    /// # #[tokio::main]
    /// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let client = quick_openai().await?;
    /// let mut conversation = vec![
    ///     ChatMessage::system("You are a helpful assistant").build()
    /// ];
    ///
    /// let (response, updated_conversation) = client
    ///     .continue_conversation(conversation, "Hello!".to_string())
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    async fn continue_conversation(
        &self,
        mut conversation: Vec<ChatMessage>,
        new_message: String,
    ) -> Result<(String, Vec<ChatMessage>), LlmError> {
        conversation.push(ChatMessage::user(new_message).build());

        let response = self.chat(conversation.clone()).await?;
        let response_text = response
            .content_text()
            .ok_or_else(|| LlmError::InternalError("No text in response".to_string()))?
            .to_string();

        conversation.push(ChatMessage::assistant(response_text.clone()).build());

        Ok((response_text, conversation))
    }

    /// Translate text to another language.
    ///
    /// # Arguments
    /// * `text` - Text to translate
    /// * `target_language` - Target language (e.g., "French", "Spanish", "Chinese")
    ///
    /// # Returns
    /// Translated text
    async fn translate(&self, text: String, target_language: String) -> Result<String, LlmError> {
        let prompt = format!("Translate the following text to {target_language}: {text}");
        self.ask(prompt).await
    }

    /// Explain a concept in simple terms.
    ///
    /// # Arguments
    /// * `concept` - The concept to explain
    /// * `audience` - Target audience (e.g., "a 5-year-old", "a beginner programmer")
    ///
    /// # Returns
    /// Simple explanation
    async fn explain(&self, concept: String, audience: Option<String>) -> Result<String, LlmError> {
        let audience_str = audience
            .map(|a| format!(" to {a}"))
            .unwrap_or_else(|| " in simple terms".to_string());

        let prompt = format!("Explain {concept}{audience_str}");
        self.ask(prompt).await
    }

    /// Generate creative content based on a prompt.
    ///
    /// # Arguments
    /// * `content_type` - Type of content (e.g., "story", "poem", "email")
    /// * `prompt` - Creative prompt
    ///
    /// # Returns
    /// Generated content
    async fn generate(&self, content_type: String, prompt: String) -> Result<String, LlmError> {
        let system_prompt = format!(
            "You are a creative writer. Generate a {content_type} based on the user's prompt."
        );
        self.ask_with_system(system_prompt, prompt).await
    }
}

/// Automatic implementation of `ChatExtensions` for all types that implement `ChatCapability`
impl<T: ChatCapability> ChatExtensions for T {}

/// Unified audio processing capability interface.
///
/// This interface provides a single entry point for all audio-related functionality,
/// including text-to-speech, speech-to-text, audio translation, and real-time processing.
/// Use the `supported_features` property to discover which features are available.
///
/// # Feature Discovery
/// Always check `supported_features()` before calling specific methods to ensure
/// the provider supports the desired functionality.
///
/// # API References
/// - OpenAI: <https://platform.openai.com/docs/guides/speech-to-text>
/// - Google: <https://cloud.google.com/speech-to-text/docs>
#[async_trait]
pub trait AudioCapability: Send + Sync {
    /// Get all audio features supported by this provider.
    fn supported_features(&self) -> &[AudioFeature];

    /// Convert text to speech with full configuration support.
    ///
    /// # Arguments
    /// * `request` - The TTS request configuration
    ///
    /// # Returns
    /// Audio response with generated audio data
    ///
    /// # Errors
    /// Returns `UnsupportedOperation` if text-to-speech is not supported.
    /// Check `supported_features()` first.
    async fn text_to_speech(&self, _request: TtsRequest) -> Result<TtsResponse, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "Text-to-speech not supported by this provider".to_string(),
        ))
    }

    /// Convert text to speech with streaming output.
    ///
    /// # Arguments
    /// * `request` - The TTS request configuration
    ///
    /// # Returns
    /// Stream of audio events
    ///
    /// # Errors
    /// Returns `UnsupportedOperation` if streaming TTS is not supported.
    async fn text_to_speech_stream(&self, _request: TtsRequest) -> Result<AudioStream, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "Streaming text-to-speech not supported by this provider".to_string(),
        ))
    }

    /// Convert speech to text with full configuration support.
    ///
    /// # Arguments
    /// * `request` - The STT request configuration
    ///
    /// # Returns
    /// Text response with transcribed content
    ///
    /// # Errors
    /// Returns `UnsupportedOperation` if speech-to-text is not supported.
    async fn speech_to_text(&self, _request: SttRequest) -> Result<SttResponse, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "Speech-to-text not supported by this provider".to_string(),
        ))
    }

    /// Convert speech to text with streaming output (optional).
    ///
    /// Default implementation returns `UnsupportedOperation`. Providers that
    /// support streaming STT can override this method.
    async fn speech_to_text_stream(&self, _request: SttRequest) -> Result<AudioStream, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "Streaming speech-to-text not supported by this provider".to_string(),
        ))
    }

    /// Translate audio to English text.
    ///
    /// # Arguments
    /// * `request` - The audio translation request
    ///
    /// # Returns
    /// Text response with translated content
    ///
    /// # Errors
    /// Returns `UnsupportedOperation` if audio translation is not supported.
    async fn translate_audio(
        &self,
        _request: AudioTranslationRequest,
    ) -> Result<SttResponse, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "Audio translation not supported by this provider".to_string(),
        ))
    }

    /// Get available voices for this provider.
    ///
    /// # Returns
    /// List of available voice configurations
    async fn get_voices(&self) -> Result<Vec<VoiceInfo>, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "Voice listing not supported by this provider".to_string(),
        ))
    }

    /// Get supported languages for transcription and translation.
    ///
    /// # Returns
    /// List of supported language information
    async fn get_supported_languages(&self) -> Result<Vec<LanguageInfo>, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "Language listing not supported by this provider".to_string(),
        ))
    }

    /// Get supported input/output audio formats.
    ///
    /// # Returns
    /// List of supported audio format strings
    fn get_supported_audio_formats(&self) -> Vec<String> {
        vec!["mp3".to_string(), "wav".to_string(), "ogg".to_string()] // Default formats
    }

    // === Convenience Methods ===

    /// Simple text-to-speech conversion (convenience method).
    ///
    /// # Arguments
    /// * `text` - The text to convert to speech
    ///
    /// # Returns
    /// Audio data as bytes
    async fn speech(&self, text: String) -> Result<Vec<u8>, LlmError> {
        let request = TtsRequest::new(text);
        let response = self.text_to_speech(request).await?;
        Ok(response.audio_data)
    }

    /// Simple audio transcription (convenience method).
    ///
    /// # Arguments
    /// * `audio` - The audio data to transcribe
    ///
    /// # Returns
    /// Transcribed text
    async fn transcribe(&self, audio: Vec<u8>) -> Result<String, LlmError> {
        let request = SttRequest::from_audio(audio);
        let response = self.speech_to_text(request).await?;
        Ok(response.text)
    }

    /// Simple file transcription (convenience method).
    ///
    /// # Arguments
    /// * `file_path` - Path to the audio file
    ///
    /// # Returns
    /// Transcribed text
    async fn transcribe_file(&self, file_path: String) -> Result<String, LlmError> {
        let request = SttRequest::from_file(file_path);
        let response = self.speech_to_text(request).await?;
        Ok(response.text)
    }

    /// Simple audio translation (convenience method).
    ///
    /// # Arguments
    /// * `audio` - The audio data to translate
    ///
    /// # Returns
    /// Translated text in English
    async fn translate(&self, audio: Vec<u8>) -> Result<String, LlmError> {
        let request = AudioTranslationRequest::from_audio(audio);
        let response = self.translate_audio(request).await?;
        Ok(response.text)
    }

    /// Simple file translation (convenience method).
    ///
    /// # Arguments
    /// * `file_path` - Path to the audio file
    ///
    /// # Returns
    /// Translated text in English
    async fn translate_file(&self, file_path: String) -> Result<String, LlmError> {
        let request = AudioTranslationRequest::from_file(file_path);
        let response = self.translate_audio(request).await?;
        Ok(response.text)
    }
}

/// Vision capability trait for image analysis and generation.
///
/// This trait provides image understanding and generation capabilities.
/// Different providers may support different aspects of vision processing.
///
/// # API References
/// - OpenAI: <https://platform.openai.com/docs/guides/vision>
/// - Google: <https://cloud.google.com/vision/docs>
/// - Anthropic: <https://docs.anthropic.com/en/docs/vision>
#[async_trait]
pub trait VisionCapability: Send + Sync {
    /// Analyzes an image with optional text prompt.
    ///
    /// # Arguments
    /// * `request` - The vision analysis request
    ///
    /// # Returns
    /// Analysis results including descriptions, detected objects, etc.
    async fn analyze_image(&self, request: VisionRequest) -> Result<VisionResponse, LlmError>;

    /// Generates an image from a text prompt.
    ///
    /// # Arguments
    /// * `request` - The image generation request
    ///
    /// # Returns
    /// Generated image data and metadata
    async fn generate_image(&self, request: ImageGenRequest) -> Result<ImageResponse, LlmError>;

    /// Get supported image formats for input.
    fn get_supported_input_formats(&self) -> Vec<String> {
        vec!["jpeg".to_string(), "png".to_string(), "webp".to_string()]
    }

    /// Get supported image formats for output.
    fn get_supported_output_formats(&self) -> Vec<String> {
        vec!["png".to_string(), "jpeg".to_string()]
    }
}

/// Core embedding capability trait for vector embeddings.
///
/// This trait provides essential text embedding functionality for semantic search,
/// similarity comparison, and other vector-based operations. It follows the
/// single responsibility principle by focusing on core embedding operations.
///
/// # API References
/// - OpenAI: <https://platform.openai.com/docs/guides/embeddings>
/// - Google: <https://cloud.google.com/vertex-ai/docs/generative-ai/embeddings>
/// - Anthropic: Currently not supported
#[async_trait]
pub trait EmbeddingCapability: Send + Sync {
    /// Generates embeddings for the given input texts.
    ///
    /// # Arguments
    /// * `input` - List of strings to generate embeddings for
    ///
    /// # Returns
    /// Embedding response with vectors and metadata
    async fn embed(&self, input: Vec<String>) -> Result<EmbeddingResponse, LlmError>;

    /// Get the dimension of embeddings produced by this provider.
    ///
    /// # Returns
    /// Number of dimensions in the embedding vectors
    fn embedding_dimension(&self) -> usize;

    /// Get the maximum number of tokens that can be embedded at once.
    ///
    /// # Returns
    /// Maximum token limit per embedding request
    fn max_tokens_per_embedding(&self) -> usize {
        8192 // Common default
    }

    /// Get supported embedding models for this provider.
    ///
    /// # Returns
    /// List of available embedding model names
    fn supported_embedding_models(&self) -> Vec<String> {
        vec!["default".to_string()]
    }
}

/// Extended embedding capabilities providing advanced features.
///
/// This trait extends the core EmbeddingCapability with advanced features
/// that are commonly needed but not essential for basic embedding functionality.
/// It follows the interface segregation principle by separating optional features.
#[async_trait]
pub trait EmbeddingExtensions: EmbeddingCapability {
    /// Generate embeddings with advanced configuration.
    ///
    /// # Arguments
    /// * `request` - Detailed embedding request with configuration
    ///
    /// # Returns
    /// Embedding response with vectors and metadata
    async fn embed_with_config(
        &self,
        request: EmbeddingRequest,
    ) -> Result<EmbeddingResponse, LlmError> {
        // Default implementation falls back to basic embed
        self.embed(request.input).await
    }

    /// Process multiple embedding requests in batch.
    ///
    /// # Arguments
    /// * `requests` - Batch of embedding requests
    ///
    /// # Returns
    /// Batch response with individual results
    async fn embed_batch(
        &self,
        requests: BatchEmbeddingRequest,
    ) -> Result<BatchEmbeddingResponse, LlmError> {
        // Default implementation processes sequentially
        let mut responses = Vec::new();

        for request in requests.requests {
            let result = self
                .embed_with_config(request)
                .await
                .map_err(|e| e.to_string());
            responses.push(result);

            // Fail fast if enabled
            if requests.batch_options.fail_fast && responses.last().unwrap().is_err() {
                break;
            }
        }

        Ok(BatchEmbeddingResponse {
            responses,
            metadata: HashMap::new(),
        })
    }

    /// Get detailed information about available embedding models.
    ///
    /// # Returns
    /// List of embedding model information
    async fn list_embedding_models(&self) -> Result<Vec<EmbeddingModelInfo>, LlmError> {
        // Default implementation returns basic info
        let models = self.supported_embedding_models();
        let model_infos = models
            .into_iter()
            .map(|id| {
                EmbeddingModelInfo::new(
                    id.clone(),
                    id,
                    self.embedding_dimension(),
                    self.max_tokens_per_embedding(),
                )
            })
            .collect();

        Ok(model_infos)
    }

    /// Calculate similarity between two embedding vectors.
    ///
    /// # Arguments
    /// * `embedding1` - First embedding vector
    /// * `embedding2` - Second embedding vector
    ///
    /// # Returns
    /// Cosine similarity score between -1 and 1
    fn calculate_similarity(
        &self,
        embedding1: &[f32],
        embedding2: &[f32],
    ) -> Result<f32, LlmError> {
        if embedding1.len() != embedding2.len() {
            return Err(LlmError::InvalidInput(
                "Embedding vectors must have the same dimension".to_string(),
            ));
        }

        let dot_product: f32 = embedding1
            .iter()
            .zip(embedding2.iter())
            .map(|(a, b)| a * b)
            .sum();

        let norm1: f32 = embedding1.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm2: f32 = embedding2.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm1 == 0.0 || norm2 == 0.0 {
            return Err(LlmError::InvalidInput(
                "Cannot calculate similarity for zero vectors".to_string(),
            ));
        }

        Ok(dot_product / (norm1 * norm2))
    }
}

/// Image generation capability trait.
///
/// This trait provides image generation, editing, and variation creation
/// capabilities across different providers.
///
/// # API References
/// - OpenAI: <https://platform.openai.com/docs/api-reference/images>
/// - Stability AI: <https://platform.stability.ai/docs/api-reference>
#[async_trait]
pub trait ImageGenerationCapability: Send + Sync {
    /// Generate images from text prompts.
    ///
    /// # Arguments
    /// * `request` - The image generation request
    ///
    /// # Returns
    /// Generated images with metadata
    async fn generate_images(
        &self,
        request: ImageGenerationRequest,
    ) -> Result<ImageGenerationResponse, LlmError>;

    /// Edit an existing image based on a prompt.
    ///
    /// # Arguments
    /// * `request` - The image editing request
    ///
    /// # Returns
    /// Edited images with metadata
    async fn edit_image(
        &self,
        _request: ImageEditRequest,
    ) -> Result<ImageGenerationResponse, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "Image editing not supported by this provider".to_string(),
        ))
    }

    /// Create variations of an existing image.
    ///
    /// # Arguments
    /// * `request` - The image variation request
    ///
    /// # Returns
    /// Image variations with metadata
    async fn create_variation(
        &self,
        _request: ImageVariationRequest,
    ) -> Result<ImageGenerationResponse, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "Image variations not supported by this provider".to_string(),
        ))
    }

    /// Get supported image sizes for this provider.
    fn get_supported_sizes(&self) -> Vec<String>;

    /// Get supported response formats for this provider.
    fn get_supported_formats(&self) -> Vec<String>;

    /// Check if the provider supports image editing.
    fn supports_image_editing(&self) -> bool {
        false
    }

    /// Check if the provider supports image variations.
    fn supports_image_variations(&self) -> bool {
        false
    }

    /// Simple image generation (convenience method).
    ///
    /// # Arguments
    /// * `prompt` - Text description of the image to generate
    /// * `size` - Optional image size (e.g., "1024x1024")
    /// * `count` - Number of images to generate (default: 1)
    ///
    /// # Returns
    /// List of image URLs or base64 data
    async fn generate_image(
        &self,
        prompt: String,
        size: Option<String>,
        count: Option<u32>,
    ) -> Result<Vec<String>, LlmError> {
        let request = ImageGenerationRequest {
            prompt,
            size,
            count: count.unwrap_or(1),
            ..Default::default()
        };

        let response = self.generate_images(request).await?;
        Ok(response
            .images
            .into_iter()
            .filter_map(|img| img.url)
            .collect())
    }
}

/// File management capability for uploading and managing files.
///
/// This interface provides a unified API for file operations across different
/// providers (OpenAI, Anthropic, etc.).
///
/// # API References
/// - OpenAI: <https://platform.openai.com/docs/api-reference/files>
/// - Anthropic: <https://docs.anthropic.com/en/api/messages>
#[async_trait]
pub trait FileManagementCapability: Send + Sync {
    /// Upload a file to the provider's storage.
    ///
    /// # Arguments
    /// * `request` - The file upload request
    ///
    /// # Returns
    /// File object with metadata
    async fn upload_file(&self, request: FileUploadRequest) -> Result<FileObject, LlmError>;

    /// List files with optional filtering.
    ///
    /// # Arguments
    /// * `query` - Optional query parameters for filtering
    ///
    /// # Returns
    /// Paginated list of files
    async fn list_files(&self, query: Option<FileListQuery>) -> Result<FileListResponse, LlmError>;

    /// Retrieve file metadata.
    ///
    /// # Arguments
    /// * `file_id` - The file identifier
    ///
    /// # Returns
    /// File object with metadata
    async fn retrieve_file(&self, file_id: String) -> Result<FileObject, LlmError>;

    /// Delete a file permanently.
    ///
    /// # Arguments
    /// * `file_id` - The file identifier
    ///
    /// # Returns
    /// Deletion confirmation
    async fn delete_file(&self, file_id: String) -> Result<FileDeleteResponse, LlmError>;

    /// Get file content as bytes.
    ///
    /// # Arguments
    /// * `file_id` - The file identifier
    ///
    /// # Returns
    /// Raw file content
    async fn get_file_content(&self, file_id: String) -> Result<Vec<u8>, LlmError>;
}

/// Content moderation capability.
///
/// This trait provides content moderation functionality to check for
/// policy violations, harmful content, etc.
///
/// # API References
/// - OpenAI: <https://platform.openai.com/docs/api-reference/moderations>
#[async_trait]
pub trait ModerationCapability: Send + Sync {
    /// Moderate content for policy violations.
    ///
    /// # Arguments
    /// * `request` - The moderation request
    ///
    /// # Returns
    /// Moderation results with flagged categories
    async fn moderate(&self, request: ModerationRequest) -> Result<ModerationResponse, LlmError>;

    /// Get supported moderation categories.
    ///
    /// # Returns
    /// List of moderation categories this provider supports
    fn supported_categories(&self) -> Vec<String> {
        vec![
            "hate".to_string(),
            "hate/threatening".to_string(),
            "harassment".to_string(),
            "harassment/threatening".to_string(),
            "self-harm".to_string(),
            "self-harm/intent".to_string(),
            "self-harm/instructions".to_string(),
            "sexual".to_string(),
            "sexual/minors".to_string(),
            "violence".to_string(),
            "violence/graphic".to_string(),
        ]
    }
}

/// Model listing capability.
///
/// This trait provides functionality to list and get information about
/// available models from the provider.
///
/// # API References
/// - OpenAI: <https://platform.openai.com/docs/api-reference/models>
/// - Anthropic: Models are typically hardcoded
#[async_trait]
pub trait ModelListingCapability: Send + Sync {
    /// Get available models from the provider.
    ///
    /// # Returns
    /// List of available models with metadata
    async fn list_models(&self) -> Result<Vec<ModelInfo>, LlmError>;

    /// Get detailed information about a specific model.
    ///
    /// # Arguments
    /// * `model_id` - The model identifier
    ///
    /// # Returns
    /// Detailed model information
    async fn get_model(&self, model_id: String) -> Result<ModelInfo, LlmError>;

    /// Check if a model is available.
    ///
    /// # Arguments
    /// * `model_id` - The model identifier
    ///
    /// # Returns
    /// Whether the model is available
    async fn is_model_available(&self, model_id: String) -> Result<bool, LlmError> {
        match self.get_model(model_id).await {
            Ok(_) => Ok(true),
            Err(LlmError::NotFound(_)) => Ok(false),
            Err(e) => Err(e),
        }
    }
}

/// Text completion capability (non-chat).
///
/// This trait provides traditional text completion functionality,
/// as opposed to conversational chat.
///
/// # API References
/// - OpenAI: <https://platform.openai.com/docs/api-reference/completions>
#[async_trait]
pub trait CompletionCapability: Send + Sync {
    /// Generate text completion from a prompt.
    ///
    /// # Arguments
    /// * `request` - The completion request
    ///
    /// # Returns
    /// Generated completion text
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, LlmError>;

    /// Generate streaming text completion.
    ///
    /// # Arguments
    /// * `request` - The completion request
    ///
    /// # Returns
    /// Stream of completion events
    async fn complete_stream(
        &self,
        _request: CompletionRequest,
    ) -> Result<CompletionStream, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "Streaming completion not supported by this provider".to_string(),
        ))
    }
}

/// Application-level timeout support for LLM operations.
///
/// This trait provides timeout control for complete operations including retries,
/// complementing the HTTP-level timeouts in `HttpConfig`.
///
/// ## Why Two Timeout Levels?
///
/// 1. **HTTP timeout** (via `HttpConfig`):
///    - Controls individual HTTP request timeout (default: 30s)
///    - Prevents single requests from hanging
///    - Does NOT control retry duration
///
/// 2. **Application timeout** (via this trait):
///    - Controls the TOTAL operation time including retries
///    - Example: 3 retries with exponential backoff could take 5+ minutes
///    - Useful for strict time constraints (e.g., web API handlers)
///
/// ## Example Scenario
///
/// ```text
/// Without application timeout:
/// - HTTP timeout: 30s per request
/// - Retry policy: 3 attempts with exponential backoff
/// - Worst case: 30s + 60s + 120s = 210s (3.5 minutes)
///
/// With application timeout of 60s:
/// - Operation fails after 60s regardless of retry state
/// - Provides predictable maximum latency
/// ```
///
/// ## Usage Example
///
/// ```rust,no_run
/// use siumai::traits::TimeoutCapability;
/// use siumai::types::{ChatMessage, Tool};
/// use std::time::Duration;
///
/// # async fn example(client: impl TimeoutCapability) -> Result<(), Box<dyn std::error::Error>> {
/// # let messages: Vec<ChatMessage> = vec![];
/// # let tools: Option<Vec<Tool>> = None;
/// // Strict timeout for web API handler
/// let response = client.chat_with_timeout(
///     messages.clone(),
///     tools.clone(),
///     Duration::from_secs(10)  // Fail fast for user-facing API
/// ).await?;
///
/// // Longer timeout for background job
/// let response = client.chat_with_timeout(
///     messages,
///     tools,
///     Duration::from_secs(300)  // Allow full retry cycle
/// ).await?;
/// # Ok(())
/// # }
/// ```
#[async_trait]
pub trait TimeoutCapability: ChatCapability + Send + Sync {
    /// Chat with application-level timeout.
    ///
    /// Controls the total operation time including all retries.
    /// Use this when you need strict time bounds regardless of retry behavior.
    ///
    /// # Arguments
    /// * `messages` - The conversation history
    /// * `tools` - Optional tools to use
    /// * `timeout` - Maximum total time for the operation (including retries)
    ///
    /// # Returns
    /// Chat response or timeout error
    async fn chat_with_timeout(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
        timeout: std::time::Duration,
    ) -> Result<ChatResponse, LlmError> {
        tokio::time::timeout(timeout, self.chat_with_tools(messages, tools))
            .await
            .map_err(|_| {
                LlmError::TimeoutError(format!(
                    "Operation timed out after {:?} (including retries)",
                    timeout
                ))
            })?
    }

    /// Streaming chat with timeout for initial response.
    ///
    /// Note: This only controls the time to receive the FIRST chunk.
    /// Once streaming starts, the timeout no longer applies.
    ///
    /// # Arguments
    /// * `messages` - The conversation history
    /// * `tools` - Optional tools to use
    /// * `timeout` - Maximum time to wait for the stream to start
    ///
    /// # Returns
    /// Chat stream or timeout error
    async fn chat_stream_with_timeout(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
        timeout: std::time::Duration,
    ) -> Result<ChatStream, LlmError> {
        tokio::time::timeout(timeout, self.chat_stream(messages, tools))
            .await
            .map_err(|_| {
                LlmError::TimeoutError(format!(
                    "Stream initialization timed out after {:?}",
                    timeout
                ))
            })?
    }
}

// Blanket implementation for all types that implement ChatCapability
impl<T> TimeoutCapability for T where T: ChatCapability + Send + Sync {}

/// OpenAI-specific capabilities.
///
/// This trait provides OpenAI-specific functionality that's not available
/// in other providers.
///
/// # API References
/// - OpenAI: <https://platform.openai.com/docs/api-reference>
#[async_trait]
pub trait OpenAiCapability: Send + Sync {
    /// Chat with structured output using JSON schema.
    ///
    /// # Arguments
    /// * `messages` - The conversation messages
    /// * `schema` - JSON schema for structured output
    ///
    /// # Returns
    /// Structured response matching the schema
    async fn chat_with_structured_output(
        &self,
        messages: Vec<ChatMessage>,
        schema: JsonSchema,
    ) -> Result<StructuredResponse, LlmError>;

    /// Create a batch processing job.
    ///
    /// # Arguments
    /// * `requests` - List of requests to process in batch
    ///
    /// # Returns
    /// Batch job information
    async fn create_batch(&self, requests: Vec<BatchRequest>) -> Result<BatchResponse, LlmError>;

    /// Use the Responses API instead of Chat Completions.
    ///
    /// # Arguments
    /// * `messages` - The conversation messages
    /// * `tools` - Optional built-in tools (web search, file search, etc.)
    ///
    /// # Returns
    /// Response from the Responses API
    async fn chat_with_responses_api(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<OpenAiBuiltInTool>>,
    ) -> Result<ChatResponse, LlmError>;
}

/// Anthropic-specific capabilities.
#[async_trait]
pub trait AnthropicCapability: Send + Sync {
    /// Caches prompts.
    async fn chat_with_cache(
        &self,
        request: ChatRequest,
        cache_config: CacheConfig,
    ) -> Result<ChatResponse, LlmError>;

    /// Thinking process.
    async fn chat_with_thinking(&self, request: ChatRequest) -> Result<ThinkingResponse, LlmError>;
}

/// OpenAI-specific embedding capabilities.
#[async_trait]
pub trait OpenAiEmbeddingCapability: EmbeddingCapability {
    /// Generate embeddings with custom dimensions (for text-embedding-3 models).
    ///
    /// # Arguments
    /// * `input` - List of strings to generate embeddings for
    /// * `dimensions` - Custom output dimensions (1-3072 for text-embedding-3-large)
    ///
    /// # Returns
    /// Embedding response with custom-sized vectors
    async fn embed_with_dimensions(
        &self,
        input: Vec<String>,
        dimensions: u32,
    ) -> Result<EmbeddingResponse, LlmError>;

    /// Generate embeddings with specific encoding format.
    ///
    /// # Arguments
    /// * `input` - List of strings to generate embeddings for
    /// * `format` - Encoding format (float or base64)
    ///
    /// # Returns
    /// Embedding response in specified format
    async fn embed_with_format(
        &self,
        input: Vec<String>,
        format: EmbeddingFormat,
    ) -> Result<EmbeddingResponse, LlmError>;
}

/// Document reranking capability trait.
///
/// This trait provides document reranking functionality, which is useful for
/// improving search relevance by reordering documents based on their relevance
/// to a specific query. This is commonly used in RAG (Retrieval-Augmented Generation)
/// systems to improve the quality of retrieved documents.
///
/// # API References
/// - SiliconFlow: <https://docs.siliconflow.cn/cn/api-reference/rerank/create-rerank>
/// - Cohere: <https://docs.cohere.com/reference/rerank>
/// - Jina AI: <https://jina.ai/reranker/>
#[async_trait]
pub trait RerankCapability: Send + Sync {
    /// Rerank documents based on their relevance to a query.
    ///
    /// # Arguments
    /// * `request` - The rerank request containing query and documents
    ///
    /// # Returns
    /// Reranked documents with relevance scores
    async fn rerank(&self, request: RerankRequest) -> Result<RerankResponse, LlmError>;

    /// Get the maximum number of documents that can be reranked in a single request.
    ///
    /// # Returns
    /// Maximum document count, or None if unlimited
    fn max_documents(&self) -> Option<u32> {
        None
    }

    /// Get supported rerank models for this provider.
    ///
    /// # Returns
    /// List of available rerank model names
    fn supported_models(&self) -> Vec<String> {
        vec![]
    }
}

/// Gemini-specific capabilities.
#[async_trait]
pub trait GeminiCapability: Send + Sync {
    /// Search-augmented generation.
    async fn chat_with_search(
        &self,
        request: ChatRequest,
        search_config: SearchConfig,
    ) -> Result<ChatResponse, LlmError>;

    /// Code execution.
    async fn execute_code(
        &self,
        code: String,
        language: String,
    ) -> Result<ExecutionResponse, LlmError>;
}

/// Gemini-specific embedding capabilities.
#[async_trait]
pub trait GeminiEmbeddingCapability: EmbeddingCapability {
    /// Generate embeddings with task type optimization.
    ///
    /// # Arguments
    /// * `input` - List of strings to generate embeddings for
    /// * `task_type` - Task type for optimization
    ///
    /// # Returns
    /// Task-optimized embedding response
    async fn embed_with_task_type(
        &self,
        input: Vec<String>,
        task_type: EmbeddingTaskType,
    ) -> Result<EmbeddingResponse, LlmError>;

    /// Generate embeddings with title context.
    ///
    /// # Arguments
    /// * `input` - List of strings to generate embeddings for
    /// * `title` - Title for context
    ///
    /// # Returns
    /// Context-aware embedding response
    async fn embed_with_title(
        &self,
        input: Vec<String>,
        title: String,
    ) -> Result<EmbeddingResponse, LlmError>;

    /// Generate embeddings with custom output dimensions.
    ///
    /// # Arguments
    /// * `input` - List of strings to generate embeddings for
    /// * `output_dimensionality` - Custom output dimensions
    ///
    /// # Returns
    /// Embedding response with custom dimensions
    async fn embed_with_output_dimensionality(
        &self,
        input: Vec<String>,
        output_dimensionality: u32,
    ) -> Result<EmbeddingResponse, LlmError>;
}

/// Ollama-specific embedding capabilities.
#[async_trait]
pub trait OllamaEmbeddingCapability: EmbeddingCapability {
    /// Generate embeddings with model-specific options.
    ///
    /// # Arguments
    /// * `input` - List of strings to generate embeddings for
    /// * `model` - Specific model to use
    /// * `options` - Model-specific options
    ///
    /// # Returns
    /// Embedding response with model-specific processing
    async fn embed_with_model_options(
        &self,
        input: Vec<String>,
        model: String,
        options: HashMap<String, serde_json::Value>,
    ) -> Result<EmbeddingResponse, LlmError>;

    /// Generate embeddings with truncation control.
    ///
    /// # Arguments
    /// * `input` - List of strings to generate embeddings for
    /// * `truncate` - Whether to truncate input to fit context length
    ///
    /// # Returns
    /// Embedding response with truncation handling
    async fn embed_with_truncation(
        &self,
        input: Vec<String>,
        truncate: bool,
    ) -> Result<EmbeddingResponse, LlmError>;

    /// Generate embeddings with keep-alive control.
    ///
    /// # Arguments
    /// * `input` - List of strings to generate embeddings for
    /// * `keep_alive` - Duration to keep model loaded
    ///
    /// # Returns
    /// Embedding response with model lifecycle control
    async fn embed_with_keep_alive(
        &self,
        input: Vec<String>,
        keep_alive: String,
    ) -> Result<EmbeddingResponse, LlmError>;
}

/// Core provider trait.
pub trait LlmProvider: Send + Sync {
    /// Provider name.
    fn provider_name(&self) -> &'static str;

    /// List of supported models.
    fn supported_models(&self) -> Vec<String>;

    /// Gets capability information.
    fn capabilities(&self) -> ProviderCapabilities;

    /// Gets the HTTP client.
    fn http_client(&self) -> &reqwest::Client;
}

/// Provider capability information.
#[derive(Debug, Clone, Default)]
pub struct ProviderCapabilities {
    pub chat: bool,
    pub audio: bool,
    pub vision: bool,
    pub tools: bool,
    pub embedding: bool,
    pub streaming: bool,
    pub file_management: bool,
    pub custom_features: HashMap<String, bool>,
}

impl ProviderCapabilities {
    /// Creates new capability information.
    pub fn new() -> Self {
        Self::default()
    }

    /// Enables chat capability.
    pub const fn with_chat(mut self) -> Self {
        self.chat = true;
        self
    }

    /// Enables audio capability.
    pub const fn with_audio(mut self) -> Self {
        self.audio = true;
        self
    }

    /// Enables vision capability.
    pub const fn with_vision(mut self) -> Self {
        self.vision = true;
        self
    }

    /// Enables tool capability.
    pub const fn with_tools(mut self) -> Self {
        self.tools = true;
        self
    }

    /// Enables embedding capability.
    pub const fn with_embedding(mut self) -> Self {
        self.embedding = true;
        self
    }

    /// Enables streaming.
    pub const fn with_streaming(mut self) -> Self {
        self.streaming = true;
        self
    }

    /// Enables file management capability.
    pub const fn with_file_management(mut self) -> Self {
        self.file_management = true;
        self
    }

    /// Adds a custom feature.
    pub fn with_custom_feature(mut self, name: impl Into<String>, enabled: bool) -> Self {
        self.custom_features.insert(name.into(), enabled);
        self
    }

    /// Checks if a feature is supported.
    pub fn supports(&self, feature: &str) -> bool {
        match feature {
            "chat" => self.chat,
            "audio" => self.audio,
            "vision" => self.vision,
            "tools" => self.tools,
            "embedding" => self.embedding,
            "streaming" => self.streaming,
            "file_management" => self.file_management,
            _ => self.custom_features.get(feature).copied().unwrap_or(false),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_capabilities() {
        let caps = ProviderCapabilities::new()
            .with_chat()
            .with_streaming()
            .with_custom_feature("custom_feature", true);

        assert!(caps.supports("chat"));
        assert!(caps.supports("streaming"));
        assert!(caps.supports("custom_feature"));
        assert!(!caps.supports("audio"));
    }

    // Test that all capability traits are Send + Sync
    #[test]
    fn test_capability_traits_are_send_sync() {
        use std::sync::Arc;

        // Test that trait objects can be used in Arc (requires Send + Sync)
        fn test_arc_usage() {
            // These should compile without errors if traits have Send + Sync
            let _: Option<Arc<dyn ChatCapability>> = None;
            let _: Option<Arc<dyn AudioCapability>> = None;
            let _: Option<Arc<dyn VisionCapability>> = None;
            let _: Option<Arc<dyn EmbeddingCapability>> = None;
            let _: Option<Arc<dyn ImageGenerationCapability>> = None;
            let _: Option<Arc<dyn FileManagementCapability>> = None;
            let _: Option<Arc<dyn ModerationCapability>> = None;
            let _: Option<Arc<dyn ModelListingCapability>> = None;
            let _: Option<Arc<dyn CompletionCapability>> = None;
            let _: Option<Arc<dyn OpenAiCapability>> = None;
            let _: Option<Arc<dyn AnthropicCapability>> = None;
            let _: Option<Arc<dyn GeminiCapability>> = None;
        }

        test_arc_usage();
    }

    // Test actual multi-threading with capability traits
    #[tokio::test]
    async fn test_capability_traits_multithreading() {
        use std::sync::Arc;
        use tokio::task;

        // Create a mock capability that we can share across threads
        struct MockCapability;

        #[async_trait::async_trait]
        impl ChatCapability for MockCapability {
            async fn chat_with_tools(
                &self,
                _messages: Vec<ChatMessage>,
                _tools: Option<Vec<Tool>>,
            ) -> Result<ChatResponse, crate::error::LlmError> {
                Ok(ChatResponse {
                    id: Some("mock-id".to_string()),
                    content: MessageContent::Text("Mock response".to_string()),
                    model: Some("mock-model".to_string()),
                    usage: None,
                    finish_reason: Some(crate::types::FinishReason::Stop),
                    tool_calls: None,
                    thinking: None,
                    metadata: std::collections::HashMap::new(),
                })
            }

            async fn chat_stream(
                &self,
                _messages: Vec<ChatMessage>,
                _tools: Option<Vec<Tool>>,
            ) -> Result<crate::stream::ChatStream, crate::error::LlmError> {
                Err(crate::error::LlmError::UnsupportedOperation(
                    "Mock streaming not implemented".to_string(),
                ))
            }
        }

        let capability: Arc<dyn ChatCapability> = Arc::new(MockCapability);

        // Spawn multiple tasks that use the capability concurrently
        let mut handles = Vec::new();

        for i in 0..5 {
            let capability_clone = capability.clone();
            let handle = task::spawn(async move {
                // This tests that the capability can be used across thread boundaries
                let messages = vec![ChatMessage::user("Test message").build()];
                let result = capability_clone.chat_with_tools(messages, None).await;
                assert!(result.is_ok());
                i // Return task id for verification
            });
            handles.push(handle);
        }

        // Wait for all tasks to complete
        let mut results = Vec::new();
        for handle in handles {
            let result = handle.await.unwrap();
            results.push(result);
        }

        // Verify all tasks completed
        assert_eq!(results.len(), 5);
        for (i, result) in results.iter().enumerate() {
            assert_eq!(*result, i);
        }
    }
}
