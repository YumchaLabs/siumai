pub(crate) use crate::standards::openai::audio::{
    ensure_openai_sse_content_type, openai_speech_audio_delta, openai_speech_audio_done,
    openai_sse_error_event, openai_sse_event_type, openai_sse_json_config,
    openai_sse_should_ignore_event_type, openai_stt_force_stream_true,
    openai_transcript_text_delta, openai_transcript_text_done, openai_transcript_text_segment,
    openai_tts_force_stream_format_sse,
};
