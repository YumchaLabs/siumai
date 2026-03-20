# Fearless Refactor V4 - Release Readiness Summary

Last updated: 2026-03-20

See also: `release-notes-draft.md`

## Executive summary

The V4 line is now best understood as **architecturally complete and stabilization-complete enough
for release**, not as a workstream that still needs another large internal redesign pass.

Current assessment:

- overall completion: approximately **98%–99%**
- architecture decision: **locked**
- public API story: **locked**
- primary provider story: **release-ready**
- remaining work: **next-cycle depth work, not V4 blocker work**

## What V4 has now achieved

### 1. Architecture is no longer generic-client-centered

The refactor goal was not “rename everything to match AI SDK”, but to finish the move toward a
family-model-centered design while keeping Rust-first public naming.

That goal is now materially achieved:

- family traits are the default execution contract
- registry handles behave like model-family objects rather than temporary migration wrappers
- provider-specific complexity remains inside provider/protocol crates
- `LlmClient` is no longer the architectural center of gravity

### 2. Builder retention policy is settled

The project explicitly keeps builders, but builders are no longer the architecture.

The current construction order is stable:

1. registry-first
2. config-first
3. builder convenience last

This is the right compromise for V4:

- ergonomics are preserved for users who want fluent setup
- the internal system no longer depends on builder-only behavior
- config-first and registry-first paths are the authoritative convergence points

### 3. Provider migration is broad enough for release

The important release question is not “is every provider equally deep internally?”.
The important question is “is the advertised public facade now explicit, guarded, and coherent?”.

That answer is now yes.

Fresh evidence in this cycle:

- OpenAI public-path contract sweep: **144 passed, 1 skipped**
- OpenAI live smoke (`gpt-5.2` + embeddings): **passed**
- Anthropic / Google / Vertex contract sweep: **238 passed, 0 skipped**
- Groq / xAI / DeepSeek / Ollama contract sweep: **275 passed, 0 skipped**

Together, those sweeps mean the current first-line provider story is no longer based on design
intent alone; it is backed by executable no-network contract coverage plus targeted live validation
where route-specific breakage actually mattered.

## Late stabilization fixes that changed the release judgment

Two late-cycle fixes matter because they were real contract gaps, not cosmetic cleanup:

- OpenAI native streaming `/responses` requests stopped sending Chat Completions-only
  `stream_options.include_usage`; live `gpt-5.2` validation proved the old behavior was wrong.
- Ollama request option parity was restored across fixture, public, and registry paths:
  - legacy nested `providerOptions["ollama"].extra_params` payloads are accepted again
  - registry embedding handles now preserve provider-specific request config on
    `embed_with_config(...)` request-aware paths before falling back to the generic family route

These are good examples of the current V4 state:

- the remaining issues are stabilization issues
- they are discoverable through focused parity/live validation
- they are fixable without reopening the architecture

## What is still incomplete

The remaining items are real, but they are mostly **scope boundaries or next-cycle depth work**.

### Non-blocking in-progress areas

- some milestone checkboxes still say `in progress` because they track depth completion, not because
  the architectural direction is unsettled
- provider package tier normalization still has cleanup room
- OpenAI-compatible audio-family expansion remains intentionally incomplete
- some secondary providers still have depth work beyond the already-guarded public surface

### Explicitly deferred boundaries

These are approved V4 boundaries, not hidden release blockers:

- do not rename existing spec types just to mirror AI SDK naming
- do not split tools into many more crates
- do not remove builders immediately
- do not introduce a Stable hosted-search surface until at least three providers converge on both
  request and response semantics

### Next-cycle backlog drivers

The following documents should drive the next focused pass rather than be folded back into V4:

- `typed-metadata-boundary-matrix.md`
- `provider-capability-alignment-matrix.md`

## Release recommendation

The recommendation for the V4 line is:

- treat V4 as **release-ready after final release-manager validation**
- do **not** reopen a large architecture refactor before release
- move remaining work into a smaller follow-up cycle focused on capability depth and boundary
  tightening

In other words:

**V4 should ship on the strength of its now-locked architecture and validated public contract, not
wait for perfect internal symmetry across every provider.**

## Suggested final gate before tagging

Before a release tag or release candidate cut, the remaining practical gate should be:

1. keep Tier 1 / Tier 2 CI green
2. keep the current provider contract sweeps green
3. compile the major examples on their recommended paths
4. rerun targeted live smoke only when route/auth/transport defaults changed

That is a release-management task, not another architecture task.

## Recommended next cycle after V4

After V4 release, the next cycle should be narrower:

1. use `typed-metadata-boundary-matrix.md` as the response-side typing backlog
2. use `provider-capability-alignment-matrix.md` as the provider-capability backlog
3. normalize provider package promotion rules where the package story is still noisier than needed
4. expand only the provider stories that gain new stable evidence, not the ones that merely look
   asymmetrical on paper

This keeps momentum without undoing the discipline that made V4 converge.
