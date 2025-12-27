//! Aggregator for HTTP 401 retry tests under tests/retry_401/.

#[path = "support/mod.rs"]
mod support;

#[path = "retry_401/http_common_retry_401.rs"]
mod http_common_retry_401;

#[path = "retry_401/audio_tts_retry_401.rs"]
mod audio_tts_retry_401;

#[path = "retry_401/audio_stt_retry_401.rs"]
mod audio_stt_retry_401;

#[path = "retry_401/executors_files_retry_401.rs"]
mod executors_files_retry_401;

#[path = "retry_401/executors_image_retry_401.rs"]
mod executors_image_retry_401;
