// Example: Using Registry speech/transcription models
// Run with: cargo run --example speech --features openai

use siumai::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(not(feature = "openai"))]
    {
        eprintln!("Enable --features openai to run this example");
        return Ok(());
    }

    #[cfg(feature = "openai")]
    {
        if std::env::var("OPENAI_API_KEY").is_err() {
            eprintln!("Set OPENAI_API_KEY to run this example");
            return Ok(());
        }
        let reg = siumai::registry::helpers::create_registry_with_defaults();

        // TTS
        let sp = reg.speech_model("openai:gpt-4o-mini-tts")?;
        let tts = sp.text_to_speech(TtsRequest::new("Hello from Siumai")).await?;
        println!("tts bytes: {}", tts.audio_data.len());

        // STT (demo only)
        let tr = reg.transcription_model("openai:gpt-4o-mini-transcribe")?;
        // let text = tr.speech_to_text(SttRequest::from_audio(tts.audio_data.clone())).await?;
        // println!("stt: {}", text.text);
        println!("stt: <demo>");
    }

    Ok(())
}

