//! MCP Streamable HTTP Transport Example
//!
//! This example demonstrates how to connect to MCP servers using
//! the current streamable HTTP transport.
//!
//! ## Transports
//!
//! ### Streamable HTTP
//! - Best for: Stateless operations, RESTful APIs
//! - Use case: Remote MCP servers with HTTP endpoints
//!
//! ## Setup
//!
//! 1. Add dependencies:
//!    ```toml
//!    siumai = { version = "0.11", features = ["openai"] }
//!    siumai-extras = { version = "0.11", features = ["mcp"] }
//!    ```
//!
//! 2. Start an MCP server with streamable HTTP transport
//!
//! 3. Run this example:
//!    ```bash
//!    MCP_TRANSPORT=http MCP_URL=http://localhost:3000/mcp \
//!      cargo run --example sse-http-examples --features openai
//!    ```

// prelude not needed for the stubbed example

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🌐 MCP Streamable HTTP Transport Example\n");
    println!("❌ This example requires siumai-extras with the 'mcp' feature");
    println!("\nTo use this example:");
    println!("1. Add to your Cargo.toml:");
    println!("   siumai-extras = {{ version = \"0.11\", features = [\"mcp\"] }}");
    println!("\n2. Import and use:");
    println!("   use siumai_extras::mcp::mcp_tools_from_http;");
    println!("\n3. Connect to MCP server:");
    println!(
        "   let (tools, resolver) = mcp_tools_from_http(\"http://localhost:3000/mcp\").await?;"
    );
    println!(
        "\nSee siumai-extras/README.md and https://docs.rs/siumai-extras/latest/siumai_extras/mcp/ for details."
    );

    Ok(())

    // Uncomment below when siumai-extras mcp feature is available:
    /*
    use siumai_extras::orchestrator::{generate, step_count_is};
    use siumai_extras::mcp::mcp_tools_from_http;

    // Get transport type from environment
    let transport = std::env::var("MCP_TRANSPORT")
        .unwrap_or_else(|_| "http".to_string());

    let url = std::env::var("MCP_URL")
        .unwrap_or_else(|_| "http://localhost:3000/mcp".to_string());

    println!("Transport: {}", transport.to_uppercase());
    println!("URL: {}\n", url);

    println!("Step 1: Connecting to MCP server...");

    let (tools, resolver) = match transport.as_str() {
        "http" => {
            println!("  Using streamable HTTP transport");
            mcp_tools_from_http(&url).await?
        }
        _ => {
            println!("❌ Unknown transport: {}", transport);
            println!("   Supported: http");
            return Ok(());
        }
    };

    println!("✅ Connected!");
    println!("\nStep 2: Discovered {} tools:", tools.len());
    for tool in &tools {
        println!("  - {}", tool.name);
        if let Some(desc) = &tool.description {
            println!("    {}", desc);
        }
    }

    println!("\nStep 3: Creating model handle via registry...");
    // Ensure `OPENAI_API_KEY` is set in your environment.
    let client = siumai::registry_global().language_model("openai:gpt-4o-mini")?;

    println!("\nStep 4: Running orchestration...");
    let messages = vec![
        user!("List all available tools and demonstrate using one of them.")
    ];

    let (response, steps) = generate(
        &client,
        messages,
        Some(tools),
        Some(&resolver),
        vec![step_count_is(10)],
        Default::default(),
    )
    .await?;

    println!("\n📝 Response:");
    println!("{}", response.content_text().unwrap_or("No text content"));

    println!("\n📊 Stats:");
    println!("  - Steps: {}", steps.len());
    println!("  - Tool calls: {}",
        steps.iter().map(|s| s.tool_calls.len()).sum::<usize>());

    println!("\n✅ Example completed!");
    */
}
