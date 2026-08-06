use std::sync::Arc;

use siumai_core::{
    ModelAdvisory, ModelLifecycle, ModelOperation, ModelPolicy, ModelPolicyContext,
    ModelPolicyDecision, ProviderProfile, SupportScope, UnsupportedReason,
};

use super::mode::OpenAiApiMode;
use super::profile::{OpenAiProfile, mode_from_scope};

pub(crate) struct OpenAiModelPolicy {
    profile: Arc<ProviderProfile>,
    responses_scope: SupportScope,
    chat_completions_scope: SupportScope,
}

impl OpenAiModelPolicy {
    pub(crate) fn new(profile: &OpenAiProfile) -> Self {
        Self {
            profile: profile.profile_arc(),
            responses_scope: profile.support_scope(OpenAiApiMode::Responses).clone(),
            chat_completions_scope: profile
                .support_scope(OpenAiApiMode::ChatCompletions)
                .clone(),
        }
    }

    fn expected_scope(&self, mode: OpenAiApiMode) -> &SupportScope {
        match mode {
            OpenAiApiMode::Responses => &self.responses_scope,
            OpenAiApiMode::ChatCompletions => &self.chat_completions_scope,
        }
    }
}

impl ModelPolicy for OpenAiModelPolicy {
    fn evaluate(&self, context: &ModelPolicyContext) -> ModelPolicyDecision {
        let Some(mode) = mode_from_scope(context.scope()) else {
            return ModelPolicyDecision::unsupported(UnsupportedReason::ApiModeMismatch);
        };
        let expected = self.expected_scope(mode);
        if context.scope().provider_id() != expected.provider()
            || context.scope().platform() != Some(expected.platform())
            || context.scope().protocol() != Some(expected.protocol())
            || context.scope().api_mode() != Some(expected.api_mode())
        {
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
            .and_then(|catalog| catalog.get(expected, context.model()))
        else {
            return ModelPolicyDecision::unknown_model();
        };
        if !model.operations().contains(&context.operation()) {
            return ModelPolicyDecision::unsupported(UnsupportedReason::OperationNotImplemented);
        }
        match model.lifecycle() {
            ModelLifecycle::Active => ModelPolicyDecision::supported(),
            ModelLifecycle::RollingAlias => {
                ModelPolicyDecision::supported().with_advisory(ModelAdvisory::RollingAlias)
            }
            ModelLifecycle::Deprecated { replacement } => ModelPolicyDecision::supported()
                .with_advisory(ModelAdvisory::Deprecated {
                    replacement: replacement.clone(),
                }),
            ModelLifecycle::Retired { .. } => {
                ModelPolicyDecision::unsupported(UnsupportedReason::ModelRetired)
            }
            _ => ModelPolicyDecision::unknown_model(),
        }
    }
}

#[cfg(test)]
mod tests {
    use siumai_core::{ModelId, SupportState};

    use super::*;
    use crate::configured::catalog::GPT_5_6;

    #[test]
    fn current_alias_is_advisory_and_unknown_ids_stay_callable() {
        let profile = OpenAiProfile::current().unwrap();
        let policy = OpenAiModelPolicy::new(&profile);
        let scope = profile.provider_scope(OpenAiApiMode::Responses).clone();

        let alias = policy.evaluate(&ModelPolicyContext::new(
            scope.clone(),
            ModelId::new(GPT_5_6).unwrap(),
            ModelOperation::Generate,
        ));
        assert_eq!(alias.state(), &SupportState::Supported);
        assert_eq!(alias.advisories(), &[ModelAdvisory::RollingAlias]);

        let future = policy.evaluate(&ModelPolicyContext::new(
            scope,
            ModelId::new("gpt-6:future").unwrap(),
            ModelOperation::Generate,
        ));
        assert_eq!(future.state(), &SupportState::Unknown);
        assert_eq!(future.advisories(), &[ModelAdvisory::UnknownModel]);
    }
}
