//! MCP SSE and HTTP Transport Examples
//!
//! This example demonstrates how to connect to MCP servers using
//! different transport protocols: SSE and HTTP.
//!
//! ## Transports
//!
//! ### SSE (Server-Sent Events)
//! - Best for: Real-time updates, long-lived connections
//! - Use case: Remote MCP servers with streaming capabilities
//!
//! ### HTTP (Streamable HTTP)
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
//! 2. Start an MCP server with SSE or HTTP transport
//!
//! 3. Run this example:
//!    ```bash
//!    # For SSE
//!    MCP_TRANSPORT=sse MCP_URL=http://localhost:8080/sse \
//!      cargo run --example sse-http-examples --features openai
//!    
//!    # For HTTP
//!    MCP_TRANSPORT=http MCP_URL=http://localhost:3000/mcp \
//!      cargo run --example sse-http-examples --features openai
//!    ```

// prelude not needed for the stubbed example

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🌐 MCP SSE/HTTP Transport Examples\n");
    println!("❌ This example requires siumai-extras with the 'mcp' feature");
    println!("\nTo use this example:");
    println!("1. Add to your Cargo.toml:");
    println!("   siumai-extras = {{ version = \"0.11\", features = [\"mcp\"] }}");
    println!("\n2. Import and use:");
    println!("   use siumai_extras::mcp::{{mcp_tools_from_sse, mcp_tools_from_http}};");
    println!("\n3. Connect to MCP server:");
    println!(
        "   let (tools, resolver) = mcp_tools_from_sse(\"http://localhost:8080/sse\").await?;"
    );
    println!("\nSee siumai/docs/guides/MCP_INTEGRATION.md for complete guide.");

    Ok(())

    // Uncomment below when siumai-extras mcp feature is available:
    /*
    use siumai::orchestrator::{generate, step_count_is};
    use siumai_extras::mcp::{mcp_tools_from_sse, mcp_tools_from_http};

    // Get transport type from environment
    let transport = std::env::var("MCP_TRANSPORT")
        .unwrap_or_else(|_| "sse".to_string());

    let url = std::env::var("MCP_URL")
        .unwrap_or_else(|_| {
            if transport == "sse" {
                "http://localhost:8080/sse".to_string()
            } else {
                "http://localhost:3000/mcp".to_string()
            }
        });

    println!("Transport: {}", transport.to_uppercase());
    println!("URL: {}\n", url);

    println!("Step 1: Connecting to MCP server...");

    let (tools, resolver) = match transport.as_str() {
        "sse" => {
            println!("  Using SSE transport");
            mcp_tools_from_sse(&url).await?
        }
        "http" => {
            println!("  Using HTTP transport");
            mcp_tools_from_http(&url).await?
        }
        _ => {
            println!("❌ Unknown transport: {}", transport);
            println!("   Supported: sse, http");
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

    // Get API key
    let api_key = std::env::var("OPENAI_API_KEY")
        .expect("OPENAI_API_KEY environment variable not set");

    println!("\nStep 3: Creating Siumai client...");
    let client = Siumai::builder()
        .openai()
        .api_key(&api_key)
        .model("gpt-4o-mini")
        .build()
        .await?;

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
