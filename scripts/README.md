# Scripts

This directory contains utility scripts for the Siumai project.

## ⚡ Fast Local Test Loop (recommended during refactors)

### `test-fast.sh`

Runs a minimal “refactor safety net” test set without `siumai`'s default `all-providers` feature
(which is convenient but slow to compile).

```bash
./scripts/test-fast.sh

# Optional: enable a small provider subset for the facade crate only
SIUMAI_TEST_FACADE=1 SIUMAI_FEATURES="openai,google" ./scripts/test-fast.sh

# Optional: enable a small provider subset for split crates too
SIUMAI_PROVIDERS_FEATURES="openai,google" SIUMAI_REGISTRY_FEATURES="openai,google" ./scripts/test-fast.sh
```

### `test-full.sh`

Runs full workspace tests with `--all-features` (CI-aligned). Uses `cargo nextest` when available
(and the repository profile `ci` from `.config/nextest.toml`).

```bash
./scripts/test-full.sh
```

### `test-smoke.sh`

Runs a feature-gated “smoke” suite that compiles protocol paths (OpenAI/Anthropic/Gemini, etc.)
without paying the cost of `--all-features`.

```bash
./scripts/test-smoke.sh

# Customize which protocol/provider features are exercised (comma-separated feature list)
SIUMAI_CORE_FEATURES="openai,anthropic" \
SIUMAI_PROVIDERS_FEATURES="openai,anthropic" \
SIUMAI_REGISTRY_FEATURES="openai,anthropic" \
./scripts/test-smoke.sh

# Optional: include the facade crate (slower)
SIUMAI_TEST_FACADE=1 SIUMAI_FEATURES="openai" ./scripts/test-smoke.sh
```

## 🧪 Integration Test Scripts

### `run_integration_tests.sh` (Linux/macOS)

Interactive script to run real LLM integration tests with environment setup.

**Usage:**
```bash
./scripts/run_integration_tests.sh
```

**Features:**
- Automatically loads `.env` file if present
- Checks for existing API keys
- Prompts for missing API keys interactively
- Shows configuration summary
- Provides test selection menu
- Handles optional base URL overrides

### `run_integration_tests.bat` (Windows)

Windows batch file version of the integration test runner.

**Usage:**
```cmd
scripts\run_integration_tests.bat
```

**Features:**
- Automatically loads `.env` file if present
- Same functionality as the shell script
- Windows-compatible batch commands
- Interactive prompts for API keys
- Test selection menu

### `test_ollama.sh` (Linux/macOS) / `test_ollama.bat` (Windows)

Dedicated Ollama testing script with comprehensive model management.

**Usage:**
```bash
# Linux/macOS
./scripts/test_ollama.sh

# Windows
scripts\test_ollama.bat
```

**Features:**
- Checks Ollama server availability
- Verifies required model installation
- Automatically offers to pull missing models
- Tests all Ollama capabilities:
  - Non-streaming chat
  - Streaming chat
  - Reasoning (with thinking models)
  - Embeddings
- Provides detailed test results and optimization tips
- Uses simple questions to save time and tokens

**Required Models:**
- **Chat**: `llama3.2:3b` (default, lightweight)
- **Reasoning**: `deepseek-r1:8b` (for thinking capabilities)
- **Embedding**: `nomic-embed-text` (for embedding tests)

**Environment Variables:**
- `OLLAMA_BASE_URL`: Ollama server URL (default: `http://localhost:11434`)
- `OLLAMA_CHAT_MODEL`: Override default chat model
- `OLLAMA_REASONING_MODEL`: Override default reasoning model
- `OLLAMA_EMBEDDING_MODEL`: Override default embedding model

## 🔧 Prerequisites

Before running the integration test scripts:

1. **Rust and Cargo**: Ensure you have Rust installed
2. **API Keys**: Have your LLM provider API keys ready
3. **Internet Connection**: Tests make real API calls

## 📋 Supported Providers

The scripts support testing with these providers:

| Provider   | Environment Variable | Required | Notes |
|------------|---------------------|----------|-------|
| OpenAI     | `OPENAI_API_KEY`    | No       | API key required |
| Anthropic  | `ANTHROPIC_API_KEY` | No       | API key required |
| Gemini     | `GEMINI_API_KEY`    | No       | API key required |
| DeepSeek   | `DEEPSEEK_API_KEY`  | No       | API key required |
| OpenRouter | `OPENROUTER_API_KEY`| No       | API key required |
| Groq       | `GROQ_API_KEY`      | No       | API key required |
| xAI        | `XAI_API_KEY`       | No       | API key required |
| Ollama     | `OLLAMA_BASE_URL`   | No       | Local server required |

**Notes:**
- You only need API keys for providers you want to test. The scripts will automatically skip providers without API keys.
- For Ollama, ensure the server is running locally (`ollama serve`) and required models are installed.

## 🔧 Optional Configuration

### Base URL Overrides

For proxy or custom endpoint usage:

```bash
# OpenAI custom endpoint
export OPENAI_BASE_URL="https://your-proxy.com/v1"

# Anthropic custom endpoint  
export ANTHROPIC_BASE_URL="https://your-proxy.com"
```

These are only used if the environment variables are set.

## 🚀 Quick Start

1. **Set up environment variables** (optional):
   ```bash
   # Option 1: Create .env file from template
   cp .env.example .env
   # Edit .env file with your API keys

   # Option 2: Export variables directly
   export OPENAI_API_KEY="your-key"
   export ANTHROPIC_API_KEY="your-key"
   ```

2. **Make script executable** (Linux/macOS only):
   ```bash
   chmod +x scripts/run_integration_tests.sh
   ```

3. **Run the script**:
   ```bash
   # Linux/macOS
   ./scripts/run_integration_tests.sh
   
   # Windows
   scripts\run_integration_tests.bat
   ```

4. **Follow the prompts**:
   - Enter API keys when prompted (or skip)
   - Choose test type from menu
   - Review test results

## 💡 Tips

- **Start with one provider**: Test with just OpenAI or Anthropic first
- **Check API limits**: Be aware of rate limits and costs
- **Use test keys**: Consider using separate API keys for testing
- **Monitor usage**: Check your API usage after running tests

## 🔍 Troubleshooting

### Common Issues

1. **Permission denied** (Linux/macOS):
   ```bash
   chmod +x scripts/run_integration_tests.sh
   ```

2. **API key errors**: 
   - Verify your API keys are correct
   - Check if your account has sufficient credits
   - Ensure API keys have required permissions

3. **Network issues**:
   - Check internet connectivity
   - Verify firewall settings
   - Try with base URL overrides if using proxies

### Getting Help

If you encounter issues:

1. Check the test output for specific error messages
2. Verify your API keys and account status
3. Review the [main documentation](../README.md)
4. Check the [test documentation](../tests/README.md)

## 📁 File Structure

```
scripts/
├── README.md                    # This file
├── run_integration_tests.sh     # Linux/macOS script
└── run_integration_tests.bat    # Windows script
```
