# Native Extension And Compat Retirement — Evidence And Gates

Status: Active
Last updated: 2026-05-25

## Smallest Current Repro

```powershell
rg -n "impl ProviderExtensionFactory|compat_.*_client_with_ctx|build_.*_extras_with_ctx|as_.*_extras" siumai-registry/src
```

This inventories the remaining extension factory and compatibility-client construction paths before
choosing a provider-native override.

## Gate Set

### Workstream Gate

```powershell
git diff --check -- docs/workstreams/native-extension-and-compat-retirement docs/workstreams/INDEX.md
```

### Registry Extension Factory Gate

```powershell
cargo check -p siumai-registry --tests --no-default-features --features openai
cargo nextest run -p siumai-registry --test factory_architecture_boundary_test --no-default-features --features openai --no-fail-fast
```

### Method-Style / Generic-Client Gate

Use focused guards selected during NECR-040. At minimum they must prove remaining
`ProviderCompatibilityFactory` and `compat_*_client*` paths are compatibility-only and not stable
family execution.

### ContentPart Root-Move Gate

Use focused content projection and public import tests selected during NECR-050. Any namespace
movement must preserve serde-facing `ChatMessage` and `ChatResponse` compatibility.

## Evidence Log

| Date | Task | Command / Evidence | Result | What it proves |
| --- | --- | --- | --- | --- |
| 2026-05-25 | NECR-010 | Workstream docs opened. | Pass | Establishes the follow-on lane and task split for the three requested refactor areas. |
| 2026-05-25 | NECR-010 | `git diff --check -- docs/workstreams/native-extension-and-compat-retirement docs/workstreams/INDEX.md`. | Pass; Git reported the expected LF-to-CRLF working-copy warning for `docs/workstreams/INDEX.md`. | Proves the new workstream docs and index have no whitespace-error diff. |

## Residual Risks

- Provider extension defaults may still adapt generic clients until each provider-native override is
  proven safe.
- Method-style construction and generic `LlmClient` remain available until ADR-0007 deletion
  prerequisites are satisfied.
- Low-level root `ContentPart` paths remain until ADR-0008 parity gates prove that the move is safe.
