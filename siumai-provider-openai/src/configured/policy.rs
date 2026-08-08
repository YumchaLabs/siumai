use std::collections::BTreeMap;
use std::sync::Arc;

use siumai_core::{
    ModelAdvisory, ModelFamily, ModelLifecycle, ModelOperation, ModelPolicy, ModelPolicyContext,
    ModelPolicyDecision, ProviderProfile, SupportScope, UnsupportedReason,
};

use super::mode::OpenAiApiMode;
use super::profile::{OpenAiProfile, mode_from_scope};

pub(crate) struct OpenAiModelPolicy {
    profile: Arc<ProviderProfile>,
    responses_scope: SupportScope,
    chat_completions_scope: SupportScope,
    family_scopes: BTreeMap<ModelFamily, SupportScope>,
}

impl OpenAiModelPolicy {
    pub(crate) fn new(profile: &OpenAiProfile) -> Self {
        Self {
            profile: profile.profile_arc(),
            responses_scope: profile.support_scope(OpenAiApiMode::Responses).clone(),
            chat_completions_scope: profile
                .support_scope(OpenAiApiMode::ChatCompletions)
                .clone(),
            family_scopes: [
                ModelFamily::Embedding,
                ModelFamily::Image,
                ModelFamily::Speech,
                ModelFamily::Transcription,
            ]
            .into_iter()
            .map(|family| {
                (
                    family,
                    profile
                        .family_support_scope(family)
                        .expect("OpenAI profile contains every exposed portable family")
                        .clone(),
                )
            })
            .collect(),
        }
    }

    fn expected_scope(&self, mode: OpenAiApiMode) -> &SupportScope {
        match mode {
            OpenAiApiMode::Responses => &self.responses_scope,
            OpenAiApiMode::ChatCompletions => &self.chat_completions_scope,
        }
    }

    fn expected_scope_for_context(&self, context: &ModelPolicyContext) -> Option<&SupportScope> {
        match context.family() {
            ModelFamily::Language => {
                mode_from_scope(context.scope()).map(|mode| self.expected_scope(mode))
            }
            family => self.family_scopes.get(&family),
        }
    }
}

impl ModelPolicy for OpenAiModelPolicy {
    fn evaluate(&self, context: &ModelPolicyContext) -> ModelPolicyDecision {
        let Some(expected) = self.expected_scope_for_context(context) else {
            return ModelPolicyDecision::unsupported(match context.family() {
                ModelFamily::Rerank => UnsupportedReason::FamilyNotImplemented,
                _ => UnsupportedReason::ApiModeMismatch,
            });
        };
        if context.scope().provider_id() != expected.provider()
            || context.scope().platform() != Some(expected.platform())
            || context.scope().protocol() != Some(expected.protocol())
            || context.scope().api_mode() != Some(expected.api_mode())
        {
            return ModelPolicyDecision::unsupported(UnsupportedReason::ApiModeMismatch);
        }
        if !operation_is_implemented(context.operation()) {
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

fn operation_is_implemented(operation: ModelOperation) -> bool {
    matches!(
        operation,
        ModelOperation::Generate
            | ModelOperation::Stream
            | ModelOperation::Embed
            | ModelOperation::GenerateImage
            | ModelOperation::SynthesizeSpeech
            | ModelOperation::Transcribe
    )
}

#[cfg(test)]
mod tests {
    use siumai_core::{ModelId, SupportState};

    use super::*;
    use crate::configured::{TEXT_EMBEDDING_3_SMALL, catalog::GPT_5_6};

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

    #[test]
    fn portable_family_scopes_use_their_own_protocol_identity() {
        let profile = OpenAiProfile::current().unwrap();
        let policy = OpenAiModelPolicy::new(&profile);
        let scope = profile
            .family_provider_scope(ModelFamily::Embedding)
            .unwrap()
            .clone();

        let supported = policy.evaluate(&ModelPolicyContext::new(
            scope.clone(),
            ModelId::new(TEXT_EMBEDDING_3_SMALL).unwrap(),
            ModelOperation::Embed,
        ));
        assert_eq!(supported.state(), &SupportState::Supported);

        let future = policy.evaluate(&ModelPolicyContext::new(
            scope,
            ModelId::new("future-embedding-model").unwrap(),
            ModelOperation::Embed,
        ));
        assert_eq!(future.state(), &SupportState::Unknown);
    }
}
