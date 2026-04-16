# Mistral Package Surface Alignment - Milestones

Last updated: 2026-04-13

## MISA-M0 - Upstream package boundary locked

Acceptance criteria:

- the main AI SDK Mistral reference files are identified
- the package boundary is recorded explicitly as chat + embedding
- non-goals around completion/image are written down

Current state:

- `repo-ref/ai/packages/mistral/src/index.ts`, `mistral-provider.ts`, and
  `mistral-chat-options.ts` are the main reference files
- the upstream package is explicitly documented as language-model + embedding only
- completion/image remain intentionally out of scope on the public wrapper boundary

Status: completed

## MISA-M1 - Public/runtime boundary aligned

Acceptance criteria:

- public builder/provider/config/registry paths expose the audited families only
- completion stays rejected on the audited package boundary
- embedding remains first-class on the same wrapper path

Current state:

- `Provider::mistral()` and `Siumai::builder().mistral()` already expose chat/chat-stream and
  embedding on the same audited wrapper boundary
- registry/public-path tests lock the intentional absence of completion support
- public-path parity tests also lock the embedding route and override behavior

Status: completed

## MISA-M2 - Typed options and wire normalization aligned

Acceptance criteria:

- Mistral typed options exist on the public/provider-owned surface
- public typed options stay camelCase
- compat request shaping owns the final wire-key lowering

Current state:

- `MistralChatOptions`, `MistralLanguageModelOptions`, `MistralReasoningEffort`, and
  `MistralChatRequestExt` are already public
- the public typed option lane stays AI SDK-style camelCase
- shared compat normalization lowers known fields onto `safe_prompt`,
  `document_image_limit`, and `document_page_limit`

Status: completed

## MISA-M3 - Model/default/catalog surface aligned

Acceptance criteria:

- curated chat and embedding model constants follow the audited package subset
- embedding default reuse is explicit
- provider catalog reuses those package-owned constants/defaults

Current state:

- curated Mistral chat model constants and `mistral-embed` are already package-owned
- shared compat family defaults now reuse `mistral-embed` explicitly
- provider catalog/public-path coverage already pins the curated chat + embedding surface

Status: completed

## MISA-M4 - Documentation and future re-audit trigger recorded

Acceptance criteria:

- a dedicated workstream documents the boundary
- future follow-up no longer depends on rediscovering why the current split exists

Current state:

- dedicated `docs/workstreams/mistral-package-surface-alignment/` docs now exist
- TypeScript-only exports remain intentionally deferred
- the main future trigger is upstream AI SDK changing the current package boundary

Status: in progress
