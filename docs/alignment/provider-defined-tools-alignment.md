# Provider-defined Tools (Vercel AI SDK Alignment)

See also: `docs/alignment/provider-implementation-alignment.md`.

`siumai` models provider-defined tools after the Vercel AI SDK convention:

- A stable provider tool **id** (e.g. `openai.web_search`)
- A call-scoped custom **name** used by the client (e.g. `webSearch` or `mySearch`)
- Optional provider-specific **args**

In Rust, this is represented as `Tool::ProviderDefined(ProviderDefinedTool { id, name, args })`.

## Why do we have both `id` and `name`?

- `id` is the stable, provider-scoped identifier (`provider.tool_type`) used for routing and
  provider-specific behavior.
- `name` is the client-facing alias used by your application. It can be any string and exists to
  match Vercel's “custom tool name” behavior (fixtures frequently use camelCase).

This means you can keep your own tool naming stable even if provider-native tool names differ.

## Recommended APIs

### 1) Use provider tool factories (best for most cases)

Use `siumai::tools` (re-export of `siumai_core::tools`) to construct tools with Vercel-aligned
default names:

```rust
use siumai::prelude::*;
use siumai::tools;

let tools = vec![
    tools::openai::web_search(),   // id: openai.web_search, name: webSearch
    tools::openai::file_search(),  // id: openai.file_search, name: fileSearch
];

let req = ChatRequest::new(vec![ChatMessage::user("hi").build()]).with_tools(tools);
```

If you want to override the custom tool name, use the `_named(...)` variants:

```rust
use siumai::tools;

let web_search = tools::openai::web_search_named("mySearch");
```

### 1b) Use provider-scoped aliases (Vercel-style import paths)

If you prefer a provider-scoped namespace (closer to the AI SDK mental model), you can use
the stable alias `siumai::providers::<provider>::tools::*`:

```rust
use siumai::prelude::*;
use siumai::providers;

let tools = vec![
    providers::openai::tools::web_search(),
    providers::openai::tools::file_search(),
];

let req = ChatRequest::new(vec![ChatMessage::user("hi").build()]).with_tools(tools);
```

Note: `Gemini` is exposed as `siumai::providers::google` (Vercel-aligned alias of `providers::gemini`).

### 2) Use `Tool::provider_defined_id(...)` for dynamic selection

If your tool list is configured by strings (e.g. config files), you can create the tool from the
id and still get the same Vercel-aligned default name:

```rust
use siumai::types::Tool;

let tool = Tool::provider_defined_id("openai.web_search").unwrap();
```

### 3) Use `siumai::hosted_tools` when you want typed args builders

Some provider-defined tools have structured configuration. For those, prefer `siumai::hosted_tools`
to build a `Tool` with typed helpers (then call `.build()`):

```rust,no_run
use siumai::hosted_tools::openai;

let tool = openai::web_search()
    .with_search_context_size("high")
    .build();
```

Alternatively, you can access the same typed builders under the provider-scoped alias:

```rust,no_run
use siumai::providers;

let tool = providers::openai::hosted_tools::web_search()
    .with_search_context_size("high")
    .build();
```

## Notes

- `Tool::provider_defined(id, name)` remains available for full manual control, but prefer the
  factory APIs above to avoid naming mismatches during the refactor.
- `Tool::provider_defined_id(...)` can only construct tools that do not require mandatory args.
  For example, `google.file_search` and `google.vertex_rag_store` require args, so
  `Tool::provider_defined_id("google.file_search")` will return `None`.
- Google/Gemini File Search:
  - `fileSearchStoreNames` is required.
  - `metadataFilter` is a string expression (AIP-160).
  - Prefer typed builders:
    `siumai::hosted_tools::google::file_search().with_file_search_store_names(vec![...]).with_metadata_filter("source = \\\"test\\\"").build()`.
- Google Vertex RAG Store:
  - `ragCorpus` is required.
  - Prefer typed builders:
    `siumai::hosted_tools::google::vertex_rag_store("projects/.../ragCorpora/...").with_top_k(5).build()`.
