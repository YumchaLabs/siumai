# Siumai Extras

Optional utilities for the [siumai](https://github.com/YumchaLabs/siumai) LLM library.

## Features

This crate provides optional functionality that extends `siumai` without adding heavy dependencies to the core library:

- **`schema`** - JSON Schema validation for structured outputs
- **`telemetry`** - Advanced tracing and logging with `tracing-subscriber`
- **`server`** - Server adapters for Axum and other web frameworks
- **`all`** - Enable all features

## Installation

```toml
[dependencies]
siumai = "0.10"
siumai-extras = { version = "0.10", features = ["schema", "telemetry"] }
```

## Usage

### Schema Validation

```rust
use siumai_extras::schema::SchemaValidator;

// Validate JSON against a schema
let validator = SchemaValidator::new(schema)?;
validator.validate(&json_value)?;
```

### Telemetry

```rust
use siumai_extras::telemetry::init_subscriber;

// Initialize tracing subscriber
init_subscriber(config)?;
```

### Server Adapters

```rust
use siumai_extras::server::axum::to_sse_response;

// Convert ChatStream to Axum SSE response
let sse = to_sse_response(stream, options);
```

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](../LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](../LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

