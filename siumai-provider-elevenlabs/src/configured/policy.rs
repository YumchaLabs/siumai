use std::sync::Arc;

use siumai_core::{
    ModelAdvisory, ModelFamily, ModelLifecycle, ModelOperation, ModelPolicy, ModelPolicyContext,
    ModelPolicyDecision, ProviderProfile, SupportScope, UnsupportedReason,
};

pub(crate) struct ElevenLabsModelPolicy {
    profile: Arc<ProviderProfile>,
    support_scope: SupportScope,
}

impl ElevenLabsModelPolicy {
    pub(crate) fn new(profile: Arc<ProviderProfile>, support_scope: SupportScope) -> Self {
        Self {
            profile,
            support_scope,
        }
    }
}

impl ModelPolicy for ElevenLabsModelPolicy {
    fn evaluate(&self, context: &ModelPolicyContext) -> ModelPolicyDecision {
        if context.scope.provider_id() != self.support_scope.provider()
            || context.scope.platform() != Some(self.support_scope.platform())
            || context.scope.protocol() != Some(self.support_scope.protocol())
            || context.scope.api_mode() != Some(self.support_scope.api_mode())
        {
            return ModelPolicyDecision::unsupported(UnsupportedReason::ApiModeMismatch);
        }
        if context.family != ModelFamily::Speech {
            return ModelPolicyDecision::unsupported(UnsupportedReason::FamilyNotImplemented);
        }
        if context.operation != ModelOperation::SynthesizeSpeech {
            return ModelPolicyDecision::unsupported(UnsupportedReason::OperationNotImplemented);
        }
        let Some(catalog) = self.profile.catalog() else {
            return ModelPolicyDecision::unknown_model();
        };
        let Some(model) = catalog.get(&self.support_scope, &context.model) else {
            return ModelPolicyDecision::unknown_model();
        };
        if !model.operations().contains(&context.operation) {
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
