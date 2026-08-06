# Siumai Repository Guide

## Scope

This file is the repository-wide guide for coding agents working anywhere in the Siumai workspace.
It records durable product boundaries, architectural responsibilities, engineering conventions,
and validation expectations. It intentionally excludes task status, release-specific migration
steps, and an inventory of every crate that happens to exist today.

Keep temporary implementation decisions in the relevant issue or `docs/plans/`. Record durable
architecture decisions in `docs/adr/` and describe the current architecture in
`docs/architecture/`. If a nested `AGENTS.md` is added later, it may specialize these rules for its
subtree but must not silently contradict the repository-wide contracts in this file.

The root `Cargo.toml` is the authority for workspace membership, shared package metadata, the Rust
edition, and the MSRV. `docs/architecture/` is the authority for the accepted current shape. Do not
infer intended ownership from directory age, compatibility aliases, or historical documentation,
and do not maintain a second hard-coded crate inventory here.

## Product model

Siumai is a Rust-first library workspace for integrating AI model providers. It offers two
complementary public paths:

- provider-owned APIs for faithful access to provider capabilities and native resources;
- provider-neutral model-family APIs, plus an optional facade, Registry, and runtime for portable
  application code.

The unified API is an ergonomic portability layer, not a replacement for provider APIs and not a
least-common-denominator client. A provider-specific capability should remain available through a
typed provider API or typed provider options even when it is not portable enough for a shared
request field.

Use upstream SDKs as references for product concepts, fixtures, and edge cases. Do not mechanically
copy their module layout or type system. Public ownership, traits, builders, errors, feature flags,
async boundaries, and extension mechanisms must be idiomatic for Rust.

## Communication and source language

- Communicate with maintainers in Chinese.
- Write source code, identifiers, code comments, rustdoc, examples, technical documentation,
  changelogs, release notes, and commit messages in English.
- Prefer precise current domain names over historical aliases or names inherited from a reference
  implementation.

## Workspace ownership

### Public assembly

- `siumai/` is the facade crate. It owns curated re-exports, feature aggregation, preludes, and the
  primary ergonomic entry points. It must not duplicate provider, protocol, transport, Registry, or
  runtime implementations.
- The root `README.md`, facade rustdoc, and facade examples describe the primary user journey. Keep
  them synchronized with actual features and public paths.

### Shared contracts

- `siumai-core/` owns the small provider-neutral model-family traits and shared request, response,
  error, options, usage, identity, tool, and stream contracts. It does not own provider
  construction, wire schemas, network execution, retries, Registry lookup, or workflow
  orchestration.
- A shared type or helper belongs in the lowest durable owning layer. Do not create generic utility
  packages merely to avoid choosing an owner, and do not preserve duplicate neutral type systems.

### Wire and transport

- `siumai-transport/` owns HTTP and WebSocket mechanics, endpoint policy, destination validation,
  redirects, retries, replay safety, body and frame bounds, and transport-level security.
- `siumai-protocol-*/` crates own wire schemas, request and response mapping, event decoding, and
  protocol stream state machines. A protocol crate does not represent a commercial provider,
  choose an account or route, or own host application policy.

### Providers and compatibility engines

- `siumai-provider-*/` crates own provider construction, credentials, technical endpoints,
  supported API modes, typed provider options, provider-specific codecs, model advisories, and
  provider-native resources.
- Dedicated compatibility-engine packages own reusable configured protocol execution, verified
  dialect profiles, and explicit custom-compatible escape hatches. They are not the public owners
  of branded providers and must not become duplicate provider crates.
- Compatibility engines or profiles may describe verified protocol compatibility. They must not
  claim native capability fidelity that their implementation and fixtures do not prove.
- A branded provider crate must not depend on another branded provider crate, the facade, or
  Registry.

### Routing and orchestration

- `siumai-registry/` owns deterministic local lookup over caller-configured provider registrations.
  It does not discover remote inventories, construct hidden providers during lookup, or choose
  business routes.
- `siumai-runtime/` owns provider-neutral multi-step execution such as tool loops, structured output,
  approvals, budgets, and durable run behavior. Provider-specific wire switches do not belong in
  runtime.

### Integrations

- `siumai-mcp/` owns lifecycle-safe MCP discovery and tool bindings across an untrusted boundary.
- `siumai-server/` owns thin server projections over runtime behavior and explicit server trust
  boundaries.
- Optional integrations should live with the narrow capability or external boundary they adapt.
  Do not collect unrelated integrations, orchestration, server behavior, and protocol conversion in
  a miscellaneous catch-all package.

### Repository support

- `config/architecture/` contains machine-checkable dependency and architecture policy.
- `scripts/` contains maintained repository automation and validation entry points.
- `.github/` contains CI, issue, pull-request, and release automation.
- `docs/` contains architecture, decisions, provider evidence, migration guidance, and task plans.
- `repo-ref/`, when present, is read-only reference material. Never edit it, depend on it, publish
  its contents, or treat its source layout as Siumai's architecture.

## Architecture rules

Keep responsibilities flowing from reusable foundations toward concrete assembly:

1. Core contracts remain small and provider-neutral.
2. Transport owns network mechanics; protocol crates own wire semantics.
3. Provider crates compose shared contracts, transport, and the protocols they need.
4. Registry and runtime operate on provider-neutral contracts instead of matching concrete
   providers.
5. The facade and integration crates assemble or project lower layers without reimplementing them.

Before adding a workspace dependency, inspect both manifests and
`config/architecture/dependency-policy.json`. Avoid dependency cycles, neutral-to-provider edges,
facade back-edges, provider-to-Registry coupling, and feature flags that activate unrelated crates.

Use the following placement test when ownership is unclear:

- A wire field, event, or decode rule belongs in the relevant protocol or provider codec.
- Credentials, signing, API paths, protocol defaults, technical addressing, and native resources
  belong in the provider crate.
- Destination safety, HTTP behavior, redirects, retries, replay rules, framing, and connection
  limits belong in transport.
- A semantic shared by multiple providers belongs in core only after its portability is
  demonstrated.
- Multi-call workflow policy belongs in runtime.
- Configured route lookup belongs in Registry.
- Re-exports and consumer feature assembly belong in the facade.
- Wire projection belongs in the protocol codec or the concrete integration that exposes that wire
  contract. Do not introduce a general cross-protocol bridge until multiple real consumers prove a
  shared, bounded loss model.

Put a rule in the lowest layer that has enough information to enforce it completely. Convenience is
not sufficient reason to move provider or transport behavior into a higher-level crate.

## Provider plane and host control plane

Siumai owns provider execution: request correctness, authentication, technical endpoint selection,
wire behavior, response decoding, and provider-native resource operations.

The host application owns business and deployment policy: aliases, default models, account and
region choice, availability, pricing, quota, compliance, health, weights, fallback, and remote
catalog caching. A provider may accept a caller-selected region, project, workspace, deployment, or
endpoint when the remote API requires it for addressing or signing. Do not turn those technical
inputs into an SDK-maintained availability catalog or business routing policy.

Provider construction and model acquisition must remain local and network-free. Remote model
catalogs, where supported, are explicit provider resource APIs rather than hidden builder or
Registry behavior.

## Public API and Rust ergonomics

- Treat provider-direct APIs and provider-neutral family APIs as first-class, complementary
  surfaces.
- Prefer small model-family traits over a universal client with capability booleans, provider
  matches, or downcast ladders.
- Keep configured providers long-lived, model-independent, cheap to clone, and safe to share.
  Creating a model handle should be synchronous and network-free.
- Keep model identifiers open. Known model constants and profiles are dated ergonomic hints, not
  exhaustive allowlists.
- Put only portable semantics in shared request types. Carry provider-specific call behavior
  through typed, namespaced provider options with provider-owned validation and merge rules. Put
  node-specific provider intent in typed annotations stored beside the message, content part, or
  tool they modify; do not use request-level numeric selectors or untyped recursive maps.
- Keep native resources with distinct lifecycle semantics—such as files, batches, catalogs,
  sessions, hosted tools, or media jobs—in provider-owned APIs until a useful portable contract is
  demonstrated.
- Use stable and experimental namespaces intentionally. Experimental APIs must not leak into stable
  preludes through broad glob re-exports.
- Prefer enums and newtypes for meaningful states, builders for non-trivial configuration, and
  concrete generics internally. Use trait objects only at deliberate runtime-erasure boundaries.
- Avoid public `Any`, stringly typed protected fields, panic-based configuration, capability
  matrices that must be manually synchronized, and public types whose useful state cannot be
  constructed or inspected.
- Treat public compatibility as release policy. When a task changes a public surface, update its
  rustdoc, examples, feature documentation, tests, and user migration guidance together.

## Provider implementation and support claims

- Use current official provider documentation as the primary source for authentication, endpoints,
  API modes, request fields, response fields, streaming behavior, and model-specific rules.
- Record the official source and verification date for named compatibility, model, or capability
  claims. Reference SDKs and community reports are secondary evidence.
- Use the canonical vendor or product name as the public provider identity. Infrastructure service
  names remain private endpoint or wire details unless callers independently configure them as a
  product.
- Prefer a native implementation when semantics differ materially, a verified compatibility
  profile when the shared dialect is accurate, and explicit custom-compatible configuration as the
  fallback.
- Unknown future model IDs should remain callable with protocol-baseline behavior when they can be
  encoded safely. Do not infer unverified capabilities or defaults from model-name patterns.
- Expose provider capabilities such as prompt caching, Responses-style APIs, reasoning controls,
  hosted tools, or provider metadata through typed provider APIs or options. Promote a field into a
  shared request only when its semantics are genuinely portable.
- Provider options must not override credentials, authorization headers, endpoints, signing input,
  or transport policy through untyped extra JSON or header maps.
- Support claims describe technical evidence: provider, technical platform, model family, API mode,
  fidelity, public stability, source, and verification date. They do not promise account-specific
  or regional commercial availability.
- Test provider identity, construction, request encoding, response decoding, stream termination,
  typed options, unsupported combinations, future model IDs, and sanitized diagnostics with
  deterministic offline fixtures.

## Correctness, security, and resource bounds

- Errors must be typed, matchable, source-preserving, and sanitized by default.
- Credentials, authorization headers, signed URLs, private response bodies, and other secrets must
  never appear in `Debug`, `Display`, tracing, snapshots, or serialization.
- Preserve absent or unknown usage as unknown. Never normalize missing provider usage to zero.
- Reject invalid configuration and unencodable requests before network submission whenever
  possible.
- A stream reports setup failure through its outer result and produces one canonical terminal
  outcome after establishment. Unexpected EOF is an error unless the protocol explicitly defines
  it as successful completion.
- Retry only when submission state and semantic idempotency make replay safe. Treat failures after
  submission or after the first stream event conservatively.
- Bound bodies, frames, queues, buffers, diagnostic excerpts, retries, tool iterations, and
  integration-side work. Default tests must be deterministic, offline, and secret-free.
- Keep `unsafe` denied unless a narrowly scoped crate documents an unavoidable reason and receives
  explicit review.

## Features and dependencies

- Keep feature flags additive, narrowly scoped, and independently checkable.
- A facade provider feature should activate exactly the dependencies required for that provider.
  Do not add empty relay features solely to preserve historical names.
- Gate optional dependencies at the owning crate and keep `--no-default-features` useful.
- Add shared dependency versions at the workspace root when multiple crates genuinely share them;
  do not centralize one-off dependencies without benefit.
- After changing dependencies or features, inspect `cargo metadata`, the affected package graph,
  facade forwarding, and relevant CI matrices.

## Documentation and examples

- Root and crate READMEs explain supported user journeys and crate-specific responsibilities.
- `docs/architecture/` describes current architecture and accepted boundaries.
- `docs/adr/` records durable decisions and trade-offs.
- `docs/providers/` records support policy, dated evidence, and provider capability matrices.
- `docs/migration/` contains user-facing guidance for released breaking changes.
- `docs/plans/` contains task-scoped plans and is not standing repository policy.
- Keep a single current explanation for each contract. Retain historical material only when it is
  intentionally labeled and still useful.
- Examples must compile against public APIs, declare required features explicitly, avoid real
  credentials, and stay small enough to teach one user journey.

When implementation, tests, and documentation disagree, identify the owning contract instead of
silently choosing the most convenient source. Keep accepted decisions, executable contracts, and
user-facing documentation consistent.

## Working in a shared worktree

Before editing:

1. Run `git status --short` and preserve existing user and agent changes.
2. Read the root manifest, affected crate manifests, and affected public module roots.
3. Read only the architecture, ADR, provider evidence, or task plan relevant to the change.
4. Inspect feature gates, downstream dependents, and contract tests before changing a public type or
   dependency.
5. Decide which crate owns the behavior before adding a shared abstraction.

Other users and agents may edit the same checkout:

- Never discard or rewrite changes you did not create.
- Do not use `git restore`, `git checkout`, `git reset`, `git stash`, or broad deletion to clean the
  worktree unless the user explicitly authorizes the exact operation.
- Never attach `main` to an additional worktree.
- Split delegated work by non-overlapping file ownership where possible. The integrating agent owns
  review of the combined diff and serial verification.
- Stage files explicitly and leave unrelated modifications unstaged.

## Testing and validation

Run Cargo commands serially and reuse the workspace `target` directory. Do not run concurrent Cargo
builds or create alternate target directories without a measured reason.

Start with the smallest meaningful lane:

```bash
cargo fmt --all -- --check
cargo nextest run -p <crate> --all-features --test-threads 1
cargo clippy -p <crate> --all-targets --all-features -j 1 -- -D warnings
```

Expand validation according to the change:

- run affected dependent crates and facade contract tests;
- check relevant `--no-default-features` and feature combinations;
- inspect `cargo metadata` after dependency or feature changes;
- run doctests when rustdoc examples change;
- run protocol and provider fixture suites after wire changes;
- run workspace, MSRV, packaging, or security lanes for repository-wide changes.

Prefer `cargo nextest` for Rust test suites. Use `cargo test` for doctests or harnesses that nextest
does not support. Do not run live, credentialed, billable, or destructive provider tests without
explicit user authorization for that external interaction.

Use `cargo fmt --all` only when it will not rewrite unrelated work in a dirty shared checkout.
Otherwise format files in scope and use `cargo fmt --all -- --check` as the workspace gate. Finish
with `git diff --check`; also run `git diff --cached --check` when changes are staged.

## Scripts and automation

- Prefer focused, cross-platform Python 3 scripts for local repository tooling.
- Shell scripts are acceptable for CI or an explicitly guaranteed Unix environment. Do not require
  PowerShell for local validation.
- Scripts should orchestrate authoritative tools or validate bounded data and schema contracts.
  They must not grow into partial Rust parsers, type inference engines, call-graph analyzers, ABI
  analyzers, or protocol compilers.
- Reuse Cargo, Clippy, rustdoc, nextest, and established validators instead of duplicating their
  logic.
- Document maintained entry points and supported modes in `scripts/README.md`.

## Commits and repository hygiene

- Keep changes scoped and reviewable. Inspect the diff and verification results before committing.
- Use English Conventional Commit messages. Stage only the files that belong to the commit.
- Do not push, tag, publish crates, create releases, or open or modify pull requests without explicit
  authorization for that external action.
- Never commit credentials, local absolute paths, private payloads, build output, editor state, or
  files copied from `repo-ref/`.
- Do not add Compound Engineering badges to repository documentation or pull requests.
