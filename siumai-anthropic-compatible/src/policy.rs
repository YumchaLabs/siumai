use std::sync::Arc;

use siumai_core::{
    ModelAdvisory, ModelLifecycle, ModelOperation, ModelPolicy, ModelPolicyContext,
    ModelPolicyDecision, ProviderProfile, SupportScope, UnsupportedReason,
};

pub(crate) struct AnthropicCompatibleModelPolicy {
    profile: Arc<ProviderProfile>,
    support_scope: SupportScope,
}

impl AnthropicCompatibleModelPolicy {
    pub(crate) fn new(profile: Arc<ProviderProfile>, support_scope: SupportScope) -> Self {
        Self {
            profile,
            support_scope,
        }
    }

    fn scope_matches(&self, context: &ModelPolicyContext) -> bool {
        context.scope().provider_id() == self.support_scope.provider()
            && context.scope().platform() == Some(self.support_scope.platform())
            && context.scope().protocol() == Some(self.support_scope.protocol())
            && context.scope().api_mode() == Some(self.support_scope.api_mode())
    }
}

impl ModelPolicy for AnthropicCompatibleModelPolicy {
    fn evaluate(&self, context: &ModelPolicyContext) -> ModelPolicyDecision {
        if !self.scope_matches(context) {
            return ModelPolicyDecision::unsupported(UnsupportedReason::ApiModeMismatch);
        }
        if !matches!(
            context.operation(),
            ModelOperation::Generate | ModelOperation::Stream
        ) {
            return ModelPolicyDecision::unsupported(UnsupportedReason::OperationNotImplemented);
        }
        let Some(model) = self
            .profile
            .catalog()
            .and_then(|catalog| catalog.get(&self.support_scope, context.model()))
        else {
            return ModelPolicyDecision::unknown_model();
        };
        if !model.operations().contains(&context.operation()) {
            return ModelPolicyDecision::unsupported(UnsupportedReason::OperationNotImplemented);
        }
        match model.lifecycle() {
            ModelLifecycle::Active => ModelPolicyDecision::supported(),
            ModelLifecycle::Deprecated { replacement } => ModelPolicyDecision::supported()
                .with_advisory(ModelAdvisory::Deprecated {
                    replacement: replacement.clone(),
                }),
            ModelLifecycle::RollingAlias => {
                ModelPolicyDecision::supported().with_advisory(ModelAdvisory::RollingAlias)
            }
            ModelLifecycle::Retired { .. } => {
                ModelPolicyDecision::unsupported(UnsupportedReason::ModelRetired)
            }
            _ => ModelPolicyDecision::unknown_model(),
        }
    }
}
