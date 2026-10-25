# Beta.12 release review

- Date: 2026-10-25
- Target: the working tree after `v0.11.0-beta.11`, including the model advisory refresh and trusted custom provider origins.
- Secondary reference: local `repo-ref/ai` at `682e2dd714`.
- Publication boundary: the maintainer has authorized code and release PRs, merging after review and CI, and beta.12 publication through the existing release workflow.

## Review lanes

| Owner | Scope | Responsibility |
|---|---|---|
| Transport reviewer | HTTP, retries, SSE, resources, proxy, WebSocket | Trace trust, replay, cancellation, bounds, and terminal behavior. |
| OpenAI reviewer | OpenAI protocol, provider, compatible engine | Trace request encoding, tool/reasoning output, usage, history, and stream completion. |
| Anthropic/Gemini reviewer | Native protocols, compatible execution, Vertex | Trace provider-native state, usage, tools, streaming, and model advice. |
| Runtime reviewer | Core, runtime, Registry, facade, MCP, server | Trace execution, persistence, approvals, cancellation, error handling, and public boundaries. |
| Other-provider reviewer | Remaining configured providers and media | Trace native wire contracts, stream completion, model advice, and resource lifecycles. |
| Integrator | Dependencies, release metadata, fixes, verification | Confirm findings, integrate bounded fixes, and run Cargo serially. |

Reviewers inspect the complete working tree without editing or running Cargo. Reference SDK behavior is evidence for edge cases, not authority over Siumai ownership or official provider contracts.

## Dependency findings

The initial lockfile audit used RustSec database commit `21d909e4c8bd4e14c7aea58bf980c881a0053ab1`, updated 2026-10-24. It reported 13 vulnerability records, one unsoundness warning, and a yanked `chacha20` release. Multiple records affect the same package; a lockfile match does not prove that every vulnerable optional feature is enabled.

| Package | Initial version | Selected version | Evidence |
|---|---|---|---|
| `h2` | 0.4.15 | 0.4.20 | [RUSTSEC-2026-0258](https://rustsec.org/advisories/RUSTSEC-2026-0258.html) |
| `rustls` | 0.23.43 | 0.23.46 | [RUSTSEC-2026-0285](https://rustsec.org/advisories/RUSTSEC-2026-0285.html), [0346](https://rustsec.org/advisories/RUSTSEC-2026-0346.html), [0348](https://rustsec.org/advisories/RUSTSEC-2026-0348.html), [0349](https://rustsec.org/advisories/RUSTSEC-2026-0349.html) |
| `rustls-webpki` | 0.103.14 | 0.103.16 | [RUSTSEC-2026-0382](https://rustsec.org/advisories/RUSTSEC-2026-0382.html) |
| `webpki-roots`, `webpki-root-certs` | 1.0.9 | 1.0.11 | [RUSTSEC-2026-0381](https://rustsec.org/advisories/RUSTSEC-2026-0381.html), [0383](https://rustsec.org/advisories/RUSTSEC-2026-0383.html) |
| `tokio` | 1.53.1 | 1.54.1 | [RUSTSEC-2026-0386](https://rustsec.org/advisories/RUSTSEC-2026-0386.html) |
| `quinn-proto` | 0.11.16 | 0.11.19 | [RUSTSEC-2026-0338](https://rustsec.org/advisories/RUSTSEC-2026-0338.html), [0339](https://rustsec.org/advisories/RUSTSEC-2026-0339.html), [0340](https://rustsec.org/advisories/RUSTSEC-2026-0340.html), [0341](https://rustsec.org/advisories/RUSTSEC-2026-0341.html), [0342](https://rustsec.org/advisories/RUSTSEC-2026-0342.html) |
| `chacha20` | 0.10.1, yanked | 0.10.2 | crates.io registry metadata |

The enabled native dependency graph includes `h2` and `rustls` through reqwest; WebSocket transport also uses rustls. `quinn-proto` is present in the lockfile but absent from the current native all-features build graph. The refresh also selects reqwest 0.13.5 and rmcp 3.7.0 within existing compatible version ranges, without raising the declared Rust 1.95 MSRV.

Keep `sse-stream` on 0.2.6: rmcp 3.7.0's public HTTP-client trait still uses the 0.2 `Sse` type. Moving Siumai alone to 0.3 would introduce incompatible types instead of upgrading that boundary.

The first audit produced its vulnerability report but timed out while checking registry yanks. Final verification must rerun the complete audit rather than treating that report as a successful gate. GitHub's unauthenticated issue API also returned a rate limit response; the issue was read through its public page instead.

## Confirmed code findings

The following findings were independently reproduced by tracing the final working-tree call paths and comparing them with the relevant reference SDK behavior. They are fixed in the current working tree.

| Surface | Finding | Fix and regression coverage |
|---|---|---|
| OpenAI Responses stream | Function calls were emitted before the response terminal outcome; truncated arguments could replace a provider failure with a protocol error. | Keep item assembly separate from promotion, derive tool events from the final successful response projection for completed and incomplete outcomes, and preserve identical truncated arguments without promoting them. Failed/cancelled outcomes never emit executable tool calls. |
| OpenAI Chat Completions stream | Refusal deltas were stored as separate final-response content parts. A proposed terminal-only event would also lose observed refusal content on errors/EOF. | Accumulate one final-response refusal while retaining incremental refusal events for partial output on exceptional stream termination. |
| Anthropic Messages | Empty signed `thinking` blocks were rejected in response decoding and history replay. | Allow empty thinking text while requiring a non-empty signature; add direct signed empty-block round-trip coverage. |
| Gemini Generate Content | Signed and unsigned adjacent text parts could be merged, changing thought-signature placement; scalar `cachedContentTokenCount` was ignored. | Merge only unsigned adjacent parts and prefer the scalar cache count; add signature and usage assertions. |
| Gemini Interactions | Unknown future event types stopped an otherwise valid stream. | Skip unknown events and let the terminal event settle the stream; add a future-event fixture. |
| MCP | Separate invalidation and publication atomics could resurrect an invalidated catalog; a proposed execution mutex could delay notifications for an entire remote call. | Keep generation and active fingerprint in one short locked state. Admission linearizes against invalidation; admitted calls can finish without holding a catalog lock during network I/O. rmcp 3.7 behavior was checked for session recovery and tool-call replay. |
| Alibaba and ElevenLabs | Alibaba nested cache creation usage was not projected into standard usage; ElevenLabs language identification probability was exposed as transcription confidence. | Decode nested cache creation tokens and keep language probability in provider metadata; add provider contract assertions. |
| Transport | Request targets accepted leading whitespace/control characters and could escape a configured base path after URL normalization; relative call timeouts were not resolved at the transport fallback boundary. | Reject ambiguous target characters, enforce normalized base-path containment, and resolve call deadlines before transport controls are built. |

The transport reviewer also identified WebSocket follow-ups: tungstenite 0.30 has payload-bearing debug/trace logs, the split receiver cannot independently flush a queued close acknowledgement, and a retained split half can retain a socket after the session permit is released. The initial global `log/max_level_info` mitigation was rejected at merge review because it changes logging for the entire host application and conflicts with other static levels. The release documents target-specific host logging filters as a limitation of the experimental WebSocket features. Close acknowledgement and split socket ownership remain follow-up transport tasks; current provider actor tests prove bounded peer-close settlement and permit release.

## Validation

- `cargo audit --no-fetch --deny warnings`: 320 locked dependencies, 0 vulnerability records, 0 warnings after the dependency refresh.
- Targeted protocol/provider/transport/MCP nextest: 441 passed, 0 skipped.
- Repository script tests: 20 passed.
- Final workspace all-features nextest: 1488 passed, 0 skipped through `python3 -B scripts/test-workspace.py full --runner nextest`.
- Workspace all-targets all-features Clippy with `-D warnings`: passed.
- Workspace all-targets all-features check on Rust 1.95: passed.
- Workspace all-features doctests: passed.
- Cargo package file-list validation: 597 paths, no private artifacts.
- `cargo fmt --all -- --check`, `git diff --check`, and `git diff --cached --check`: passed.

## Release status

The initial review was local-only. The maintainer subsequently authorized committing, opening and merging the code and release PRs, and publishing beta.12 through the existing manual workflow. Independent final merge review approved the integrated fixes after the 1488-test suite and workspace Clippy passed. Code PR [#34](https://github.com/YumchaLabs/siumai/pull/34) merged as `bfa6c27e` after all four CI checks passed; the merged main also passed all four checks. release-plz generated release PR [#35](https://github.com/YumchaLabs/siumai/pull/35), which still requires final version/documentation review and current-head CI before merging. release-plz remains the versioning and publication authority.
