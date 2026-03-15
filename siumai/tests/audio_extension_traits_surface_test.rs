use siumai::Provider;
use siumai::experimental::client::LlmClient;
use siumai::extensions::{SpeechExtras, TranscriptionExtras};

#[cfg(feature = "openai")]
#[tokio::test]
async fn openai_public_client_exposes_speech_and_transcription_extras() {
    let client = Provider::openai()
        .api_key("test-key")
        .model("gpt-4o-mini")
        .build()
        .await
        .expect("openai client");

    assert!(client.as_speech_capability().is_some());
    let speech_extras = client.as_speech_extras().expect("speech extras");
    let _: &dyn SpeechExtras = speech_extras;

    assert!(client.as_transcription_capability().is_some());
    let transcription_extras = client
        .as_transcription_extras()
        .expect("transcription extras");
    let _: &dyn TranscriptionExtras = transcription_extras;
}

#[cfg(feature = "openai")]
#[tokio::test]
async fn fireworks_public_client_exposes_transcription_extras_without_speech_extras() {
    let client = Provider::openai()
        .fireworks()
        .api_key("test-key")
        .model("whisper-v3")
        .build()
        .await
        .expect("fireworks client");

    assert!(client.as_speech_capability().is_none());
    assert!(client.as_speech_extras().is_none());

    assert!(client.as_transcription_capability().is_some());
    let transcription_extras = client
        .as_transcription_extras()
        .expect("transcription extras");
    let _: &dyn TranscriptionExtras = transcription_extras;
}

#[cfg(feature = "xai")]
#[tokio::test]
async fn xai_public_client_exposes_speech_family_without_audio_extras() {
    let client = Provider::xai()
        .api_key("test-key")
        .model("grok-4")
        .build()
        .await
        .expect("xai client");

    assert!(client.as_speech_capability().is_some());
    assert!(client.as_speech_extras().is_none());

    assert!(client.as_transcription_capability().is_none());
    assert!(client.as_transcription_extras().is_none());
}

#[cfg(feature = "groq")]
#[tokio::test]
async fn groq_public_client_exposes_audio_family_without_audio_extras() {
    let client = Provider::groq()
        .api_key("test-key")
        .model("whisper-large-v3")
        .build()
        .await
        .expect("groq client");

    assert!(client.as_speech_capability().is_some());
    assert!(client.as_speech_extras().is_none());

    assert!(client.as_transcription_capability().is_some());
    assert!(client.as_transcription_extras().is_none());
}

#[cfg(feature = "minimaxi")]
#[tokio::test]
async fn minimaxi_public_client_exposes_speech_family_without_audio_extras() {
    let client = Provider::minimaxi()
        .api_key("test-key")
        .model("speech-2.5-hd")
        .build()
        .await
        .expect("minimaxi client");

    assert!(client.as_speech_capability().is_some());
    assert!(client.as_speech_extras().is_none());

    assert!(client.as_transcription_capability().is_none());
    assert!(client.as_transcription_extras().is_none());
}
