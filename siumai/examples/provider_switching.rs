use siumai::Siumai;
use siumai::providers::anthropic::models::CLAUDE_SONNET_5;
use siumai::providers::google::models::GEMINI_3_5_FLASH;
use siumai::providers::openai::models::GPT_5_6;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Provider and model selection change; the application call does not.
    // Futures are intentionally dropped so this example uses only synthetic
    // credentials and remains deterministic and network-free when checked.
    {
        let ai = Siumai::builder()
            .openai()
            .api_key("example-openai-key")
            .build()?;
        let client = ai.language(GPT_5_6)?;
        drop(client.generate("Hello"));
    }

    {
        let ai = Siumai::builder()
            .anthropic()
            .api_key("example-anthropic-key")
            .build()?;
        let client = ai.language(CLAUDE_SONNET_5)?;
        drop(client.generate("Hello"));
    }

    {
        let ai = Siumai::builder()
            .gemini()
            .api_key("example-gemini-key")
            .build()?;
        let client = ai.language(GEMINI_3_5_FLASH)?;
        drop(client.generate("Hello"));
    }

    Ok(())
}
