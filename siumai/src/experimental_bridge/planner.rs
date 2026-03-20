//! Bridge planning helpers.
//!
//! The planner decides which high-level path a bridge should take. It does not
//! perform protocol serialization itself.

use siumai_core::bridge::BridgeTarget;

use super::request::pairs::{DirectRequestBridgePair, direct_request_bridge_pair};

/// High-level path selected for a request bridge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestBridgePath {
    ViaNormalized,
    Direct(DirectRequestBridgePair),
}

/// A request bridge plan selected from the requested source/target pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RequestBridgePlan {
    pub source: Option<BridgeTarget>,
    pub target: BridgeTarget,
    pub path: RequestBridgePath,
}

impl RequestBridgePlan {
    pub const fn uses_direct_pair_bridge(&self) -> bool {
        matches!(self.path, RequestBridgePath::Direct(_))
    }
}

/// Plan how a request bridge should be executed.
///
/// The current policy is intentionally conservative:
///
/// - select a direct bridge only for the highest-value pairwise paths
/// - otherwise fall back to the normalized bridge path
pub fn plan_chat_request_bridge(
    source: Option<BridgeTarget>,
    target: BridgeTarget,
) -> RequestBridgePlan {
    let path = match source.and_then(|source| direct_request_bridge_pair(source, target)) {
        Some(pair) => RequestBridgePath::Direct(pair),
        None => RequestBridgePath::ViaNormalized,
    };

    RequestBridgePlan {
        source,
        target,
        path,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::experimental_bridge::request::pairs::DirectRequestBridgePair;

    #[test]
    fn planner_selects_direct_pair_for_anthropic_to_openai_responses() {
        let plan = plan_chat_request_bridge(
            Some(BridgeTarget::AnthropicMessages),
            BridgeTarget::OpenAiResponses,
        );

        assert_eq!(
            plan.path,
            RequestBridgePath::Direct(DirectRequestBridgePair::AnthropicMessagesToOpenAiResponses,)
        );
    }

    #[test]
    fn planner_selects_direct_pair_for_openai_responses_to_anthropic() {
        let plan = plan_chat_request_bridge(
            Some(BridgeTarget::OpenAiResponses),
            BridgeTarget::AnthropicMessages,
        );

        assert_eq!(
            plan.path,
            RequestBridgePath::Direct(DirectRequestBridgePair::OpenAiResponsesToAnthropicMessages,)
        );
    }

    #[test]
    fn planner_uses_normalized_path_for_non_direct_target_pair() {
        let plan = plan_chat_request_bridge(
            Some(BridgeTarget::AnthropicMessages),
            BridgeTarget::OpenAiChatCompletions,
        );

        assert_eq!(plan.path, RequestBridgePath::ViaNormalized);
    }
}
