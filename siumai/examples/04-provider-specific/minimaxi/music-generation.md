# MiniMaxi Music Generation Example

This example demonstrates how to use the MiniMaxi music generation capability.

## Basic Usage

```rust
use siumai::prelude::*;
use siumai::types::music::MusicGenerationRequest;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create MiniMaxi client
    let client = LlmBuilder::new()
        .minimaxi()
        .api_key("your-api-key")
        .build()
        .await?;

    // Create music generation request
    let request = MusicGenerationRequest::new(
        "music-2.0",
        "Indie folk, melancholic, introspective, longing, solitary walk, coffee shop",
        r#"[verse]
Streetlights flicker, the night breeze sighs
Footsteps echo on empty streets
A lone figure walks, lost in thought
Coffee shop lights beckon warmly

[chorus]
In this quiet moment, I find myself
Between the shadows and the light
Searching for meaning in the silence
Of this endless night"#
    );

    // Generate music
    let response = client.generate_music(request).await?;

    // Save the audio file
    std::fs::write("generated_music.mp3", &response.audio_data)?;

    // Print metadata
    println!("Music generated successfully!");
    if let Some(duration) = response.metadata.music_duration {
        println!("Duration: {} ms", duration);
    }
    if let Some(sample_rate) = response.metadata.music_sample_rate {
        println!("Sample rate: {} Hz", sample_rate);
    }
    if let Some(size) = response.metadata.music_size {
        println!("File size: {} bytes", size);
    }

    Ok(())
}
```

## Custom Audio Settings

```rust
use siumai::prelude::*;
use siumai::types::music::{MusicGenerationRequest, MusicAudioSetting};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = LlmBuilder::new()
        .minimaxi()
        .api_key("your-api-key")
        .build()
        .await?;

    // Create request with custom audio settings
    let request = MusicGenerationRequest::new(
        "music-2.0",
        "Upbeat electronic dance music, energetic, festival vibes",
        "[intro]\n[build-up]\n[drop]\n[outro]"
    )
    .with_sample_rate(48000)
    .with_bitrate(320000)
    .with_format("mp3");

    let response = client.generate_music(request).await?;
    std::fs::write("edm_track.mp3", &response.audio_data)?;

    println!("EDM track generated!");

    Ok(())
}
```

## Structured Lyrics

MiniMaxi supports structured lyrics with tags:

- `[Intro]` - Introduction
- `[Verse]` - Verse section
- `[Chorus]` - Chorus section
- `[Bridge]` - Bridge section
- `[Outro]` - Ending section

```rust
use siumai::prelude::*;
use siumai::types::music::MusicGenerationRequest;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = LlmBuilder::new()
        .minimaxi()
        .api_key("your-api-key")
        .build()
        .await?;

    let lyrics = r#"[Intro]
Soft piano melody begins

[Verse]
Walking through the memories
Of days gone by
Every moment captured
In the corner of my eye

[Chorus]
Time keeps moving forward
But my heart stays here
In this moment frozen
Crystal clear

[Bridge]
If I could turn back time
I'd hold you close again
But all I have are memories
And this refrain

[Chorus]
Time keeps moving forward
But my heart stays here
In this moment frozen
Crystal clear

[Outro]
Soft piano fades away"#;

    let request = MusicGenerationRequest::new(
        "music-2.0",
        "Emotional ballad, piano-driven, nostalgic, heartfelt",
        lyrics
    );

    let response = client.generate_music(request).await?;
    std::fs::write("ballad.mp3", &response.audio_data)?;

    println!("Ballad generated!");

    Ok(())
}
```

## Different Music Styles

### Rock

```rust
let request = MusicGenerationRequest::new(
    "music-2.0",
    "Hard rock, electric guitar, powerful drums, energetic",
    "[verse]\nThunder rolls across the sky\n[chorus]\nWe rise!"
);
```

### Jazz

```rust
let request = MusicGenerationRequest::new(
    "music-2.0",
    "Smooth jazz, saxophone, laid-back, sophisticated",
    "[verse]\nMoonlight on the water\n[chorus]\nSweet melody"
);
```

### Classical

```rust
let request = MusicGenerationRequest::new(
    "music-2.0",
    "Classical orchestral, strings, majestic, cinematic",
    "[intro]\n[movement 1]\n[movement 2]\n[finale]"
);
```

### Hip-Hop

```rust
let request = MusicGenerationRequest::new(
    "music-2.0",
    "Hip-hop, urban beats, rhythmic, modern",
    "[verse]\nCity lights and late nights\n[chorus]\nWe on top"
);
```

## Error Handling

```rust
use siumai::prelude::*;
use siumai::types::music::MusicGenerationRequest;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = LlmBuilder::new()
        .minimaxi()
        .api_key("your-api-key")
        .build()
        .await?;

    let request = MusicGenerationRequest::new(
        "music-2.0",
        "Ambient electronic, atmospheric",
        "[verse]\nFloating through space"
    );

    match client.generate_music(request).await {
        Ok(response) => {
            std::fs::write("ambient.mp3", &response.audio_data)?;
            println!("Music generated successfully!");
        }
        Err(e) => {
            eprintln!("Failed to generate music: {}", e);
        }
    }

    Ok(())
}
```

## Supported Models

```rust
use siumai::prelude::*;
use siumai::traits::MusicGenerationCapability;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = LlmBuilder::new()
        .minimaxi()
        .api_key("your-api-key")
        .build()
        .await?;

    // Get supported models
    let models = client.get_supported_music_models();
    println!("Supported music models: {:?}", models);
    // Output: ["music-2.0"]

    Ok(())
}
```

## Notes

- **Prompt Length**: 10-2000 characters
- **Lyrics Length**: 10-3000 characters
- **Default Sample Rate**: 44100 Hz
- **Default Bitrate**: 256000 bps
- **Default Format**: MP3
- **Model**: music-2.0 (latest version)

## API Reference

### MusicGenerationRequest

- `new(model, prompt, lyrics)` - Create a new request
- `with_sample_rate(rate)` - Set sample rate (e.g., 44100, 48000)
- `with_bitrate(bitrate)` - Set bitrate (e.g., 256000, 320000)
- `with_format(format)` - Set audio format (e.g., "mp3")
- `with_audio_setting(setting)` - Set complete audio settings

### MusicGenerationResponse

- `audio_data: Vec<u8>` - Generated audio data
- `metadata: MusicMetadata` - Music metadata

### MusicMetadata

- `music_duration: Option<u32>` - Duration in milliseconds
- `music_sample_rate: Option<u32>` - Sample rate
- `music_channel: Option<u32>` - Number of channels
- `bitrate: Option<u32>` - Bitrate
- `music_size: Option<u32>` - File size in bytes

