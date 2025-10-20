#!/bin/bash

# Real LLM Integration Test Runner
# This script helps you run the integration tests with proper environment setup

set -e

echo "🚀 Real LLM Integration Test Runner"
echo "=================================="
echo ""

# Load .env file if it exists
if [ -f ".env" ]; then
    echo "📄 Loading environment variables from .env file..."
    set -a  # automatically export all variables
    source .env
    set +a  # stop automatically exporting
    echo "✅ .env file loaded"
    echo ""
else
    echo "💡 No .env file found. You can create one from .env.example"
    echo ""
fi

# Function to check if environment variable is set
check_env_var() {
    local var_name=$1
    local var_value=${!var_name}
    
    if [ -n "$var_value" ]; then
        echo "✅ $var_name is set"
        return 0
    else
        echo "❌ $var_name is not set"
        return 1
    fi
}

# Function to prompt for API key
prompt_for_key() {
    local var_name=$1
    local provider_name=$2
    
    echo ""
    echo "🔑 Enter your $provider_name API key (or press Enter to skip):"
    read -s api_key
    
    if [ -n "$api_key" ]; then
        export $var_name="$api_key"
        echo "✅ $var_name set for this session"
    else
        echo "⏭️ Skipping $provider_name"
    fi
}

echo "📋 Checking environment variables..."
echo ""

# Check which providers are configured
providers_configured=0
total_providers=8

# OpenAI
if check_env_var "OPENAI_API_KEY"; then
    ((providers_configured++))
else
    prompt_for_key "OPENAI_API_KEY" "OpenAI"
    if [ -n "$OPENAI_API_KEY" ]; then
        ((providers_configured++))
    fi
fi

# Anthropic
if check_env_var "ANTHROPIC_API_KEY"; then
    ((providers_configured++))
else
    prompt_for_key "ANTHROPIC_API_KEY" "Anthropic"
    if [ -n "$ANTHROPIC_API_KEY" ]; then
        ((providers_configured++))
    fi
fi

# Gemini
if check_env_var "GEMINI_API_KEY"; then
    ((providers_configured++))
else
    prompt_for_key "GEMINI_API_KEY" "Google Gemini"
    if [ -n "$GEMINI_API_KEY" ]; then
        ((providers_configured++))
    fi
fi

# DeepSeek
if check_env_var "DEEPSEEK_API_KEY"; then
    ((providers_configured++))
else
    prompt_for_key "DEEPSEEK_API_KEY" "DeepSeek"
    if [ -n "$DEEPSEEK_API_KEY" ]; then
        ((providers_configured++))
    fi
fi

# OpenRouter
if check_env_var "OPENROUTER_API_KEY"; then
    ((providers_configured++))
else
    prompt_for_key "OPENROUTER_API_KEY" "OpenRouter"
    if [ -n "$OPENROUTER_API_KEY" ]; then
        ((providers_configured++))
    fi
fi

# Groq
if check_env_var "GROQ_API_KEY"; then
    ((providers_configured++))
else
    prompt_for_key "GROQ_API_KEY" "Groq"
    if [ -n "$GROQ_API_KEY" ]; then
        ((providers_configured++))
    fi
fi

# xAI
if check_env_var "XAI_API_KEY"; then
    ((providers_configured++))
else
    prompt_for_key "XAI_API_KEY" "xAI"
    if [ -n "$XAI_API_KEY" ]; then
        ((providers_configured++))
    fi
fi

# Ollama
echo ""
echo "🦙 Checking Ollama availability..."
ollama_base_url=${OLLAMA_BASE_URL:-"http://localhost:11434"}
if curl -s "$ollama_base_url/api/tags" > /dev/null 2>&1; then
    echo "✅ Ollama is available at $ollama_base_url"
    export OLLAMA_BASE_URL="$ollama_base_url"
    ((providers_configured++))
else
    echo "❌ Ollama is not available at $ollama_base_url"
    echo "💡 To enable Ollama tests:"
    echo "   1. Install Ollama: https://ollama.ai"
    echo "   2. Start Ollama: ollama serve"
    echo "   3. Pull models: ollama pull llama3.2:3b && ollama pull deepseek-r1:8b && ollama pull nomic-embed-text"
fi

echo ""
echo "📊 Summary: $providers_configured/$total_providers providers configured"
echo ""

if [ $providers_configured -eq 0 ]; then
    echo "❌ No providers configured. Please set at least one API key."
    echo ""
    echo "Example:"
    echo "export OPENAI_API_KEY=\"your-api-key\""
    echo "export ANTHROPIC_API_KEY=\"your-api-key\""
    echo ""
    exit 1
fi

# Check for optional base URL overrides
echo "🔧 Optional configuration:"
if check_env_var "OPENAI_BASE_URL"; then
    echo "   Using custom OpenAI base URL: $OPENAI_BASE_URL"
fi
if check_env_var "ANTHROPIC_BASE_URL"; then
    echo "   Using custom Anthropic base URL: $ANTHROPIC_BASE_URL"
fi

echo ""
echo "🧪 Running integration tests..."
echo ""

# Ask user which test to run
echo "Which test would you like to run?"
echo "1) All available providers - Basic tests (Chat, Streaming, Embedding, Reasoning)"
echo "2) All capability tests - Comprehensive testing (includes Tools, Vision, Audio, etc.)"
echo "3) Specific capability test"
echo "4) Specific provider test"
echo "5) Provider interface tests (Provider::* vs Siumai::builder())"
echo ""
read -p "Enter your choice (1-5): " choice

case $choice in
    1)
        echo "🚀 Running basic provider tests..."
        cargo test test_all_available_providers -- --ignored --nocapture
        ;;
    2)
        echo "🚀 Running comprehensive capability tests..."
        echo ""
        echo "📋 Running basic provider tests..."
        cargo test test_all_available_providers -- --ignored --nocapture
        echo ""
        echo "🔧 Running tool capability tests..."
        cargo test test_all_provider_tools -- --ignored --nocapture
        echo ""
        echo "👁️ Running vision capability tests..."
        cargo test test_all_provider_vision -- --ignored --nocapture
        echo ""
        echo "🔊 Running audio capability tests..."
        cargo test test_all_provider_audio -- --ignored --nocapture
        echo ""
        echo "📦 Running provider interface tests..."
        cargo test test_all_provider_interfaces -- --ignored --nocapture
        ;;
    3)
        echo ""
        echo "Available capability tests:"
        echo "- test_all_provider_tools (Tool calling across all providers)"
        echo "- test_all_provider_vision (Vision/multimodal across supported providers)"
        echo "- test_all_provider_audio (Audio TTS/STT for OpenAI and Groq)"
        echo "- test_all_provider_interfaces (Provider::* vs Siumai::builder())"
        echo "- test_all_available_providers (Basic chat, streaming, embedding, reasoning)"
        echo ""
        read -p "Enter test name: " test_name
        cargo test $test_name -- --ignored --nocapture
        ;;
    4)
        echo ""
        echo "Available provider tests:"
        echo "Basic tests:"
        echo "- test_openai_integration"
        echo "- test_anthropic_integration"
        echo "- test_gemini_integration"
        echo "- test_deepseek_integration"
        echo "- test_openrouter_integration"
        echo "- test_groq_integration"
        echo "- test_xai_integration"
        echo "- test_ollama_integration"
        echo ""
        echo "Tool capability tests:"
        echo "- test_openai_tools"
        echo "- test_anthropic_tools"
        echo "- test_gemini_tools"
        echo "- test_xai_tools"
        echo "- test_ollama_tools"
        echo ""
        echo "Vision capability tests:"
        echo "- test_openai_vision"
        echo "- test_anthropic_vision"
        echo "- test_gemini_vision"
        echo "- test_xai_vision"
        echo ""
        echo "Audio capability tests:"
        echo "- test_openai_audio_capability"
        echo "- test_groq_audio_capability"
        echo ""
        echo "Provider interface tests:"
        echo "- test_openai_provider_interface"
        echo "- test_anthropic_provider_interface"
        echo "- test_gemini_provider_interface"
        echo "- test_ollama_provider_interface"
        echo ""
        read -p "Enter test name: " test_name
        cargo test $test_name -- --ignored --nocapture
        ;;
    5)
        echo "🚀 Running provider interface tests..."
        cargo test test_all_provider_interfaces -- --ignored --nocapture
        ;;
    *)
        echo "❌ Invalid choice. Exiting."
        exit 1
        ;;
esac

echo ""
echo "✅ Integration tests completed!"
echo ""
echo "📊 Test Coverage Summary:"
echo "- ✅ Basic Chat & Streaming: All providers"
echo "- ✅ Embedding: OpenAI, Gemini, Ollama"
echo "- ✅ Reasoning: OpenAI (o1), Anthropic (thinking), Gemini, DeepSeek, xAI, Ollama"
echo "- ✅ Tool Calling: OpenAI, Anthropic, Gemini, xAI, Ollama"
echo "- ✅ Vision: OpenAI, Anthropic, Gemini, xAI"
echo "- ✅ Audio: OpenAI, Groq"
echo "- ✅ Provider Interfaces: Provider::* vs Siumai::builder()"
echo ""
echo "💡 Tips:"
echo "- Tests automatically skip providers without API keys"
echo "- Some features may not be available for all API keys (this is normal)"
echo "- Check the output for any warnings about missing permissions"
echo "- Vision tests use public images from Wikimedia Commons"
echo "- Audio tests include TTS (Text-to-Speech) but skip STT (Speech-to-Text) without audio files"
echo "- Tool tests use simple calculator and weather tools"
echo ""
echo "🔗 For more information:"
echo "- Test documentation: siumai/tests/README.md"
echo "- Test coverage analysis: TEST_COVERAGE_ANALYSIS.md"
echo "- Individual test files: siumai/tests/*_test.rs"
echo ""
