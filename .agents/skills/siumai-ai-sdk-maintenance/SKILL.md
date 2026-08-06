---
name: siumai-ai-sdk-maintenance
description: Maintains Siumai provider behavior and public APIs against current official provider documentation, using the local Vercel AI SDK checkout only as secondary reference material. Use when refreshing a provider, model advisory, capability, protocol profile, support claim, facade export, or provider-onboarding guidance in the Siumai Rust workspace.
---

# Siumai Provider Maintenance

Use this skill inside the Siumai workspace. Treat the repository `AGENTS.md`, current
architecture documents, and provider support policy as the ownership contract.

## Evidence order

1. Read the relevant official provider documentation for authentication, endpoint, request,
   response, stream, error, resource lifecycle, and model-policy facts.
2. Record the provider, technical platform, family, API mode, fidelity, stability, source, and
   verification date in the provider-owned support declaration or documentation.
3. Use `repo-ref/ai` only as secondary prior art for concepts, edge cases, and fixture ideas. It is
   read-only reference material and never determines Siumai names, ownership, or model allowlists.

Do not encode account entitlement, commercial availability, pricing, quota, fallback, or regional
catalogs in provider runtime types. A caller-selected region, project, workspace, deployment, or
endpoint is allowed only when the remote protocol needs it for addressing or signing.

## Provider refresh workflow

1. Identify the owning provider, protocol, transport, family trait, and public facade path before
   editing.
2. Choose the smallest faithful implementation: native protocol/resource first, verified dialect
   profile when semantics are proven compatible, and generic compatibility as the explicit escape
   hatch.
3. Keep model IDs open. Add dated constants or advisories only for ergonomics and verified quirks;
   unknown future IDs remain callable with baseline behavior and no guessed capabilities.
4. Put portable semantics in `siumai-core`. Keep provider-specific controls in typed provider
   options, annotations, profiles, or native resources. Reject unsupported or ambiguous requests
   before transport submission.
5. Add a focused offline fixture for each changed wire contract and the necessary negative or
   lifecycle boundary. Prefer one representative test over a combinatorial matrix when the same
   codec path is already covered.
6. Update the provider support document and user-facing examples in the same change.
7. Run the narrow serial validation lane, then inspect `git diff --check` and the feature graph.

## Public-surface review

Check direct provider construction, explicit family/API-mode selection, optional Registry
registration, facade re-exports, typed options, annotations, native resources, stream termination,
sanitized diagnostics, and unknown future model behavior. Keep stable preludes curated; do not add
experimental APIs through broad wildcard exports.

## Local tooling

Use the repository's small Python entry points for orchestration and bounded manifest/fixture
checks. They may call Cargo, nextest, Clippy, rustdoc, or other authoritative tools, but must not
parse Rust or TypeScript source, infer call graphs, regenerate provider code, or duplicate compiler
and protocol logic.

The local AI SDK checkout can be resolved when secondary comparison is useful:

```bash
python3 .agents/skills/siumai-ai-sdk-maintenance/scripts/resolve_ai_sdk_repo.py
```

Do not use a model-catalog scraper as a release gate. Model freshness is proven by the official
source/date attached to the provider claim and by deterministic provider fixtures.

## Validation

Keep Cargo processes serial and reuse the workspace target directory. Start with the smallest
meaningful lane:

```bash
cargo fmt --all -- --check
cargo nextest run -p <crate> --all-features -j 1 --test-threads 1
cargo clippy -p <crate> --all-targets --all-features -j 1 -- -D warnings
```

Expand to dependent crates, facade feature combinations, doctests, MSRV, packaging, and workspace
gates only when the changed surface requires them.
