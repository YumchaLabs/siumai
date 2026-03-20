# Protocol Bridge + Gateway Runtime - TODO

Last updated: 2026-03-20

This TODO list is intentionally organized as mergeable tracks.

## 0) Lock the boundary

- [ ] Confirm the workstream scope:
  - explicit protocol bridges
  - gateway runtime policy
  - no multi-tenant billing or admin control plane
- [ ] Confirm package boundaries:
  - protocol crates own wire formats and serializer/parser state machines
  - `siumai-core` owns bridge contracts and reports
  - `siumai-extras` owns gateway adapters and runtime policy
- [ ] Decide final module names for the bridge surface

## 1) Audit the current bridge path

- [ ] Document the current inbound and outbound paths for:
  - Anthropic Messages
  - OpenAI Responses
  - OpenAI Chat Completions
  - Gemini GenerateContent
- [ ] Mark which conversions are already exact, lossy, or implicit
- [ ] Inventory current customization points:
  - request-side
  - response-side
  - stream-side
- [ ] Identify stateful stream converters that must become explicit bridge dependencies

## 2) Define bridge contracts

- [ ] Add bridge contract types:
  - `BridgeTarget`
  - `BridgeMode`
  - `BridgeReport`
  - `BridgeWarning`
  - `BridgeResult<T>`
- [ ] Define a stable representation for lossy conversion reasons
- [ ] Define a stable representation for unsupported semantics
- [ ] Decide how provider metadata is carried across bridges

## 3) Make request bridges explicit

- [ ] Add explicit request bridge entry points for:
  - Anthropic Messages -> normalized request
  - OpenAI Responses -> normalized request
  - OpenAI Chat Completions -> normalized request
- [ ] Add direct compat helpers where they materially reduce loss:
  - Anthropic Messages -> OpenAI Responses
  - OpenAI Responses -> Anthropic Messages
- [ ] Ensure request bridges can emit `BridgeReport`
- [ ] Ensure request bridges can reject unsupported shapes in `Strict` mode

## 4) Make non-streaming response bridges explicit

- [ ] Add explicit normalized response -> target protocol converters
- [ ] Ensure tool calls, reasoning, structured output, and usage survive when possible
- [ ] Emit loss reports for dropped or downgraded semantics
- [ ] Add no-network tests for exact and lossy cases

## 5) Make streaming bridges explicit

- [ ] Add explicit stream bridge adapters built on V3 stream parts
- [ ] Ensure terminal events are preserved across protocol views
- [ ] Ensure finish reasons survive target serialization
- [ ] Ensure content block ordering is validated for Anthropic output
- [ ] Ensure OpenAI final finish chunk behavior is consistent
- [ ] Add no-network finalization tests for incomplete upstream termination

## 6) Add customization hooks

- [ ] Add request pre-bridge transform hook
- [ ] Add response post-bridge transform hook
- [ ] Add stream-part transform hook
- [ ] Add tool name / id remapping hook
- [ ] Add route-level override for `BridgeMode`
- [ ] Add a policy for handling lossy fields:
  - reject
  - warn and continue
  - silently drop

## 7) Add gateway runtime policy

- [ ] Define `GatewayBridgePolicy` in `siumai-extras`
- [ ] Cover:
  - body limits
  - upstream read limits
  - stream idle timeout
  - keepalive interval
  - header filtering
  - error passthrough
  - bridge strictness default
  - warning emission
- [ ] Integrate the policy with Axum helper surfaces
- [ ] Keep framework-agnostic pieces separate from Axum wrappers

## 8) Documentation and examples

- [ ] Add a bridge-focused architecture note after the contract lands
- [ ] Add runnable examples for:
  - Anthropic -> OpenAI Responses gateway
  - OpenAI Responses -> Anthropic gateway
  - custom lossy-policy handling
  - custom stream transform
- [ ] Update `docs/README.md` to include this workstream
- [ ] Add a migration note if any public gateway helpers change shape

## 9) Validation

- [ ] Add fixture-based bridge tests for request, response, and streaming paths
- [ ] Add explicit tests for lossy conversions
- [ ] Add tests for custom hooks
- [ ] Add tests for strict vs best-effort behavior
- [ ] Add gateway smoke coverage for JSON and SSE output paths

