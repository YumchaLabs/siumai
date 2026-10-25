# Model advisory refresh

- Model evidence checked: 2026-10-25
- Scope: dated completion hints and lifecycle advice, not a provider capability refresh
- Introduced in: `0.11.0-beta.12`

The constants below are available in `0.11.0-beta.12`. The
[self-hosted HTTP guide](openai-compatible.md) documents the trusted-origin API introduced in the
same release and retains compatibility examples for `0.11.0-beta.11`.

Model identifiers remain open. Catalog entries describe basic family operations, not every
provider option or hosted tool. A new constant does not enable a feature, change the selected API
mode, or promise account availability. Retired and deprecated IDs remain constructible locally;
the configured endpoint decides whether to accept them.

## New hints

| Provider | Added IDs | Official evidence | Scope |
|---|---|---|---|
| OpenAI | `gpt-6-astra`, `gpt-6.1-sol`, `gpt-6-luna` | [Model catalog](https://developers.openai.com/api/docs/models), [GPT-6 guide](https://developers.openai.com/api/docs/guides/latest-model) | Exact-ID classification and basic Responses/Chat generation/streaming advisories; no new tool or reasoning-option claim |
| Anthropic | `claude-opus-5-5`, `claude-sonnet-5-5`, `claude-fable-5-1`, `claude-haiku-5-5` | [Models overview](https://platform.claude.com/docs/en/models/overview) | Messages hints and basic generation/streaming advisories; `current_models()` selects these four while preserving older constants and rows |
| Google Gemini | `gemini-3.8-flash`, `gemini-4-argon`, `gemini-nano-banana-2.1` | [Release notes](https://ai.google.dev/gemini-api/docs/changelog) | Hints only; no new stable-v1 catalog rows |
| DeepSeek | `deepseek-flash` | [Models](https://api-docs.deepseek.com/quick_start/pricing/), [Responses](https://api-docs.deepseek.com/guides/responses_api/), [Messages](https://api-docs.deepseek.com/guides/anthropic_api/) | Current call name added to the existing language-mode advisories; old V4 IDs remain unchanged |
| MiniMax | `MiniMax-M3.1-Flash` | [Release notes](https://platform.minimax.io/docs/release-notes/models), [Messages](https://platform.minimax.io/docs/api-reference/text-chat-anthropic), [Chat](https://platform.minimax.io/docs/api-reference/text-chat-openai), [Responses](https://platform.minimax.io/docs/api-reference/responses-create) | Basic advisories for the three existing language modes; no claim for new effort controls or service-tier combinations |
| Groq | `qwen/qwen3.8-27b` | [Model page](https://console.groq.com/docs/model/qwen/qwen3.8-27b) | Preview namespace only; excluded from the stable language catalog |

Existing protocol, transport, resource, and older-model verification dates are not advanced merely
because a new model hint was added. See provider-owned evidence for each individual catalog row.

## Lifecycle corrections

| Provider | IDs | Corrected advice | Official evidence |
|---|---|---|---|
| OpenAI | `gpt-image-1` | Retired on 2026-10-23; replacement hint `gpt-image-2.5-sunburst` | [Deprecations](https://developers.openai.com/api/docs/deprecations), [model page](https://developers.openai.com/api/docs/models/gpt-image-1) |
| Google Gemini | `gemini-3.5-flash` | Deprecated; replacement `gemini-3.6-flash` | [2026-10-08 release notes](https://ai.google.dev/gemini-api/docs/changelog) |
| Google Gemini | `gemini-3.1-flash-image` | Deprecated; replacement hint `gemini-nano-banana-2.1`; not marked retired | [2026-10-06 release notes](https://ai.google.dev/gemini-api/docs/changelog) |
| Google Gemini | `gemini-3.1-flash-tts-preview`, `gemini-2.5-flash-preview-tts`, `gemini-2.5-pro-preview-tts` | Deprecated, not retired; shutdown table lists 2026-11-17 | [Deprecations](https://ai.google.dev/gemini-api/docs/deprecations) |
| Groq | `groq/compound`, `groq/compound-mini` | Retired on 2026-09-21; no replacement asserted | [Deprecations](https://console.groq.com/docs/deprecations) |

The GPT Image 2.5 replacement constant is an ID hint, not a new verified image-options profile.
Catalog replacement links require an existing row in the same exact scope. Where the replacement
is only an ID hint (GPT Image 2.5 or Nano Banana 2.1), the lifecycle row deliberately leaves its
replacement unset; the official migration hint is recorded here instead of inventing a verified
row or weakening the catalog invariant.
Other dated catalog entries retain their existing evidence; this document is not an exhaustive
review of audio, media, or provider-native resource lifecycle changes.

## Deliberate limits

- Gemini's current implementation uses stable-v1 Interactions and explicit stable-v1 Generate
  Content. The [stable-v1 reference](https://ai.google.dev/api/interactions-api-v1) does not yet
  establish the new IDs for these exact modes. Existing `current_*` lists therefore remain dated
  hints, not a new recommendation to switch API versions. Their older IDs can carry deprecation
  advice without becoming local execution bans.
- New Gemini TTS models have a changed audio contract. They need a separate codec/format review;
  adding their names to the existing raw-PCM slice would overstate compatibility.
- New Anthropic and MiniMax models change thinking/tool/effort behavior. This refresh does not
  implement those new options or infer defaults from model names.
- DeepSeek's September announcement and current catalog differ about V4 Pro routing/lifecycle.
  Its existing advisory is not changed without resolving that evidence conflict. An old ID being
  forwarded by the provider is not the same as that ID being rejected.
- Groq's Qwen 3.6 withdrawal notice is account-scope-specific. This refresh does not turn it into
  a universal retirement claim or an account/availability catalog.

Official provider documentation is the primary evidence. Local `repo-ref/` checkouts remain
read-only secondary comparisons, not sources for automatic catalog generation or capability rules.
