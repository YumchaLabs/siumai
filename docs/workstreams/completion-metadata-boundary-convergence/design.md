# Completion Metadata Boundary Convergence - Design

Last updated: 2026-05-17

Status: closed

## Problem

The completion-family implementation is already first-class, but completion response metadata is
still implemented in duplicated provider-client helpers:

- native OpenAI completion has local `completion_provider_metadata`,
  `merge_completion_provider_metadata`, and `flatten_completion_stream_provider_metadata` helpers
- OpenAI-compatible completion has near-identical local helpers with one extra `sources` extraction
- chat response metadata already uses a protocol-level extraction seam, while completion metadata
  still lives inside concrete clients

This is a shallow module shape: the provider clients must know the details of completion metadata
extraction even though the behavior is OpenAI protocol-family behavior. The duplication also makes
future AI SDK parity fixes easy to apply to chat but miss on completion.

## Target State

Move reusable completion response/stream provider-metadata extraction into the OpenAI protocol
layer and let concrete provider clients call that seam.

The shared completion metadata helper owns:

- namespacing extracted completion metadata under the requested provider key
- preserving raw completion `choices[0].logprobs`
- preserving non-empty top-level `sources` when compatible providers return them
- merging provider-metadata maps across stream chunks
- exposing the current stream-finish provider-metadata map without provider-client-specific
  duplication

Provider clients still own:

- request construction
- provider-specific finish-reason and usage parsing
- provider id / metadata key selection
- response metadata headers and request id capture

## Scope

In scope:

- `siumai-protocol-openai` OpenAI standard/protocol helper surface
- native OpenAI completion client
- OpenAI-compatible completion client
- focused regression tests around native and compatible completion metadata behavior

Out of scope:

- changing completion prompt materialization rules
- changing provider-option normalization
- changing chat metadata extraction
- changing the shared runtime stream event model
- broad fixture expansion unrelated to completion metadata

## Architecture Direction

This follows ADR-0001 and ADR-0002:

- the protocol crate owns OpenAI-family wire semantics
- concrete provider crates remain adapters around provider identity, settings, and transport
- provider-specific metadata stays provider-namespaced, but shared extraction mechanics do not live
  in each provider client

## Validation

The lane is complete when:

- duplicated local completion metadata helpers are removed from the concrete clients
- native OpenAI and OpenAI-compatible completion responses still preserve raw `logprobs`
- OpenAI-compatible completion responses still preserve `sources`
- streaming completion finish metadata still carries the accumulated provider metadata
- related packages pass focused `nextest` gates

All validation criteria are satisfied by the closeout gates recorded in `evidence_and_gates.md`.
