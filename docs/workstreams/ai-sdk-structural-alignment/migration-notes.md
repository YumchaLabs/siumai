# AI SDK Structural Alignment - Migration Notes

Last updated: 2026-03-31

This note captures the public-facing migration impact from the `Usage` canonicalization pass.

## Scope

The stable `Usage` type now treats AI SDK-style fields as canonical:

- `inputTokens`
- `outputTokens`
- `raw`

Legacy totals are still supported for compatibility, but they are no longer public storage fields on
the Rust struct.

## Rust API impact

Before:

- callers could read `usage.prompt_tokens`, `usage.completion_tokens`, `usage.total_tokens`
- callers could build `Usage` with struct literals

Now:

- read legacy totals through compatibility accessors:
  - `usage.prompt_tokens()`
  - `usage.completion_tokens()`
  - `usage.total_tokens()`
- when printing/logging, prefer `unwrap_or(0)` only if the local surface truly wants an unknown ->
  `0` display fallback
- construct usage through:
  - `Usage::builder()`
  - `Usage::new(prompt, completion)`
  - `Usage::with_legacy_fields(prompt, completion, total)`

Recommended direction for new code:

- treat `input_tokens`, `output_tokens`, and `raw` as the source of truth
- use legacy totals only for compatibility with older tests, logs, or adapter code

## JSON / serde impact

Serde compatibility is intentionally preserved:

- known legacy totals still serialize as `prompt_tokens`, `completion_tokens`, `total_tokens`
- incoming legacy totals still deserialize into the compatibility layer
- unknown legacy totals are now omitted instead of being synthesized as zeroes

That means existing JSON payloads remain readable, but payloads built from partial AI SDK-style
usage data may now omit legacy totals when those values are genuinely unknown.

## Mechanical migration patterns

Replace field reads:

```rust
let total = usage.total_tokens;
```

With:

```rust
let total = usage.total_tokens();
```

Replace display-oriented field reads:

```rust
println!("{}", usage.total_tokens);
```

With:

```rust
println!("{}", usage.total_tokens().unwrap_or(0));
```

Replace struct literals:

```rust
let usage = Usage {
    prompt_tokens: Some(10),
    completion_tokens: Some(5),
    total_tokens: Some(15),
    ..Default::default()
};
```

With:

```rust
let usage = Usage::with_legacy_fields(10, 5, 15);
```

Or:

```rust
let usage = Usage::builder()
    .prompt_tokens(10)
    .completion_tokens(5)
    .total_tokens(15)
    .build();
```

## Why this changed

AI SDK uses optional usage slots. Keeping legacy totals as the public storage root made it too easy
to:

- synthesize zeroes for unknown provider totals
- keep writing new code against compatibility fields
- leak the old totals-first model into newer provider/protocol refactors

This migration makes the stable Rust surface match the intended AI SDK layering more closely while
keeping legacy JSON and accessor compatibility.
