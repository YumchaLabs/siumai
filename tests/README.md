# Tests

This directory contains all test files for the Siumai project, organized by functionality.

## Directory Structure

```
tests/
├── README.md                           # This file
├── streaming/                          # Streaming functionality tests
│   ├── stream_start_event_test.rs     # StreamStart event generation tests
│   ├── complete_stream_events_test.rs # Complete streaming event sequence tests
│   ├── streaming_integration_test.rs  # General streaming integration tests
│   └── tool_call_streaming_integration_test.rs # Tool call streaming tests
├── providers/                          # Provider-specific tests
│   ├── provider_interface_test.rs     # Provider interface compliance tests
│   ├── provider_headers_test.rs       # HTTP header validation tests
│   └── gemini_thinking_test.rs        # Gemini thinking capability tests
├── capabilities/                       # Feature capability tests
│   ├── audio_capability_test.rs       # Audio processing capability tests
│   ├── tool_capability_test.rs        # Tool calling capability tests
│   ├── vision_capability_test.rs      # Vision/image processing capability tests
│   ├── image_generation_test.rs       # Image generation capability tests
│   └── embedding_integration_tests.rs # Embedding generation tests
├── parameters/                         # Parameter handling tests
│   ├── parameter_validation_test.rs   # Parameter validation tests
│   ├── parameter_advanced_tests.rs    # Advanced parameter handling tests
│   ├── parameter_mapping_consistency.rs # Parameter mapping consistency tests
│   ├── parameter_internal_verification_test.rs # Internal parameter verification
│   └── siumai_parameter_passing_test.rs # Parameter passing tests
├── mock/                              # Mock testing framework
│   ├── mock_framework.rs             # HTTP mock testing framework
│   └── mock_streaming_provider.rs    # Mock streaming provider for testing
├── core/                              # Core functionality tests
│   ├── clone_support_test.rs         # Clone trait implementation tests
│   ├── config_validation_tests.rs    # Configuration validation tests
│   ├── concurrency_tests.rs          # Concurrency and thread safety tests
│   ├── network_error_tests.rs        # Network error handling tests
│   └── resource_management_tests.rs  # Resource management tests
├── integration/                       # Integration tests
│   └── siliconflow_rerank_test.rs    # SiliconFlow rerank integration tests
├── integration_tests.rs              # Core integration tests
├── real_llm_integration_test.rs       # Tests with real LLM providers (requires API keys)
├── request_builder_integration_test.rs # Request builder integration tests
├── request_builder_consistency.rs     # Request builder consistency tests
├── siumai_unified_interface_test.rs   # Unified interface tests
├── unified_reasoning_test.rs          # Unified reasoning capability tests
├── max_tokens_default_test.rs         # Max tokens default behavior tests
└── url_compatibility_test.rs          # URL compatibility tests
```

## Test Categories

### 🌊 Streaming Tests (`tests/streaming/`)
Tests for streaming functionality across all providers:
- **StreamStart Event Tests** - Verify metadata emission at stream beginning
- **Complete Event Sequence Tests** - Test full streaming event flows
- **Integration Tests** - General streaming functionality
- **Tool Call Streaming** - Tool call specific streaming tests

### 🔌 Provider Tests (`tests/providers/`)
Tests for provider-specific functionality:
- **Interface Compliance** - Ensure providers implement required interfaces
- **Header Validation** - HTTP header handling tests
- **Provider-Specific Features** - Tests for unique provider capabilities

### 🎯 Capability Tests (`tests/capabilities/`)
Tests for specific AI capabilities:
- **Audio Processing** - Audio input/output handling
- **Tool Calling** - Function calling capabilities
- **Vision** - Image processing and analysis
- **Image Generation** - Image creation capabilities
- **Embeddings** - Text embedding generation

### ⚙️ Parameter Tests (`tests/parameters/`)
Tests for parameter handling and validation:
- **Validation** - Parameter validation logic
- **Advanced Handling** - Complex parameter scenarios
- **Mapping Consistency** - Parameter mapping across providers
- **Internal Verification** - Internal parameter processing

### 🎭 Mock Tests (`tests/mock/`)
Mock testing framework and utilities:
- **HTTP Mock Framework** - Mock server for HTTP requests
- **Streaming Mock Provider** - Mock provider for streaming tests

### 🏗️ Core Tests (`tests/core/`)
Core functionality and infrastructure tests:
- **Clone Support** - Clone trait implementations
- **Configuration** - Configuration validation
- **Concurrency** - Thread safety and concurrent access
- **Network Errors** - Error handling and recovery
- **Resource Management** - Memory and resource cleanup

## Running Tests

### All Tests
```bash
cargo test
```

### By Category
```bash
# Streaming tests
cargo test --test "streaming/*"

# Provider tests
cargo test --test "providers/*"

# Capability tests
cargo test --test "capabilities/*"

# Parameter tests
cargo test --test "parameters/*"

# Mock tests
cargo test --test "mock/*"

# Core tests
cargo test --test "core/*"
```

### Individual Test Files
```bash
# StreamStart event tests
cargo test --test streaming/stream_start_event_test

# Complete streaming sequence tests
cargo test --test streaming/complete_stream_events_test

# Mock streaming provider tests
cargo test --test mock/mock_streaming_provider
```

## Test Requirements

Some tests require environment variables:
- `OPENAI_API_KEY` - For OpenAI integration tests
- `ANTHROPIC_API_KEY` - For Anthropic integration tests
- `GOOGLE_API_KEY` - For Google/Gemini integration tests

## Mock Testing

The mock framework provides utilities for:
- **HTTP Mock Servers** - Simulate API responses
- **Error Injection** - Test error handling scenarios
- **Network Failure Simulation** - Test network resilience
- **Rate Limiting Simulation** - Test rate limit handling
- **Authentication Failure Testing** - Test auth error scenarios
- **Streaming Event Simulation** - Test complete streaming flows

## Adding New Tests

When adding new tests, place them in the appropriate category:

1. **Streaming-related** → `tests/streaming/`
2. **Provider-specific** → `tests/providers/`
3. **Capability/feature** → `tests/capabilities/`
4. **Parameter handling** → `tests/parameters/`
5. **Mock/testing utilities** → `tests/mock/`
6. **Core functionality** → `tests/core/`

Follow the naming convention: `{feature}_{type}_test.rs`

## Recent Additions

### StreamStart Event Fix (v0.9.2)
- Added comprehensive StreamStart event tests
- Fixed missing StreamStart events across all providers
- Created complete streaming event sequence tests
- Added mock streaming provider for testing

The streaming tests now verify that all providers correctly emit:
1. **StreamStart** - With proper metadata at stream beginning
2. **ContentDelta** - Incremental content updates
3. **ToolCallDelta** - Tool call information (where supported)
4. **ThinkingDelta** - Reasoning content (where supported)
5. **UsageUpdate** - Token usage information
6. **StreamEnd** - Final response with complete data
