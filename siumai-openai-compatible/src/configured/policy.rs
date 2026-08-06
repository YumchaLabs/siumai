use std::sync::Arc;

use siumai_core::{
    ModelAdvisory, ModelLifecycle, ModelOperation, ModelPolicy, ModelPolicyContext,
    ModelPolicyDecision, ProviderProfile, SupportScope, UnsupportedReason,
};

use super::mode::OpenAiCompatibleApiMode;
use super::profile::OpenAiCompatibleProfile;

pub(crate) struct OpenAiCompatibleModelPolicy {
    profile: Arc<ProviderProfile>,
    chat_scope: Option<SupportScope>,
    responses_scope: Option<SupportScope>,
}

impl OpenAiCompatibleModelPolicy {
    pub(crate) fn new(profile: &OpenAiCompatibleProfile) -> Self {
        Self {
            profile: profile.profile_arc(),
            chat_scope: profile
                .support_scope(OpenAiCompatibleApiMode::ChatCompletions)
                .cloned(),
            responses_scope: profile
                .support_scope(OpenAiCompatibleApiMode::Responses)
                .cloned(),
        }
    }

    fn matching_scope(&self, context: &ModelPolicyContext) -> Option<&SupportScope> {
        self.chat_scope
            .iter()
            .chain(self.responses_scope.iter())
            .find(|expected| {
                context.scope().provider_id() == expected.provider()
                    && context.scope().platform() == Some(expected.platform())
                    && context.scope().protocol() == Some(expected.protocol())
                    && context.scope().api_mode() == Some(expected.api_mode())
            })
    }
}

impl ModelPolicy for OpenAiCompatibleModelPolicy {
    fn evaluate(&self, context: &ModelPolicyContext) -> ModelPolicyDecision {
        let Some(expected) = self.matching_scope(context) else {
            return ModelPolicyDecision::unsupported(UnsupportedReason::ApiModeMismatch);
        };
        if !matches!(
            context.operation(),
            ModelOperation::Generate | ModelOperation::Stream
        ) {
            return ModelPolicyDecision::unsupported(UnsupportedReason::OperationNotImplemented);
        }
        let Some(model) = self
            .profile
            .catalog()
            .and_then(|catalog| catalog.get(expected, context.model()))
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
