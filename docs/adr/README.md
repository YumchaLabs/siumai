# Architecture Decision Records

Architecture Decision Records capture durable repository decisions, their context, alternatives,
and consequences. An ADR is authoritative only when its status is `Accepted` and it has not been
superseded by a later accepted decision or current architecture contract.

## Current decisions

- `0010-provider-plane-and-host-control-plane.md` — Provider crates own technical execution and
  addressing; host applications own account, commercial region availability, routing, defaults,
  compliance, pricing, quota, and fallback. Alibaba is the public provider identity; DashScope is
  an internal service/endpoint detail.
- `0011-protocol-projection-ownership.md` — Protocol codecs and concrete integrations own wire
  projection. Siumai does not maintain a generic cross-protocol bridge without demonstrated shared
  consumers and a bounded canonical loss model.
- `0012-provider-annotations-follow-semantic-nodes.md` — Call configuration remains separate from
  durable typed provider annotations. Message, content, and tool annotations live beside the
  semantic nodes they modify instead of using fragile request-level indexes or untyped maps.
- `0013-provider-identity-and-family-registration.md` — The base provider exposes only canonical
  identity. Exact execution scope and policy belong to non-empty per-family registration bindings;
  support profiles and manifests remain separate evidence surfaces.

## Conventions

- ADRs are written in English.
- New records start as `Proposed` and become `Accepted` only after an explicit project decision.
- Each ADR records the problem, considered options and trade-offs, decision, consequences, and any
  required migration or verification work.
- Task progress, branch state, and temporary implementation checklists belong in plans or issue
  tracking, not in ADRs.
