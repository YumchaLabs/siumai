use serde_json::Value;
use siumai_core::bridge::BridgeReport;
use siumai_core::types::{ProviderDefinedTool, Tool};

pub(crate) type ProviderToolArgsMapper = fn(usize, &Value, &mut BridgeReport) -> Value;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TargetToolNamePolicy {
    PreserveSourceName,
    Fixed(&'static str),
}

#[derive(Clone, Copy)]
pub(crate) struct ProviderToolTranslationRule {
    pub(crate) source_tool_types: &'static [&'static str],
    pub(crate) target_tool_id: &'static str,
    pub(crate) target_tool_name: TargetToolNamePolicy,
    pub(crate) choice_name: Option<&'static str>,
    pub(crate) aliases: &'static [&'static str],
    pub(crate) args_mapper: ProviderToolArgsMapper,
}

impl ProviderToolTranslationRule {
    pub(crate) fn matches(self, provider_tool: &ProviderDefinedTool) -> bool {
        let Some(tool_type) = provider_tool.tool_type() else {
            return false;
        };

        self.source_tool_types
            .iter()
            .any(|candidate| *candidate == tool_type)
    }

    pub(crate) fn translate_tool(
        self,
        index: usize,
        provider_tool: &ProviderDefinedTool,
        report: &mut BridgeReport,
    ) -> Tool {
        let tool_name = match self.target_tool_name {
            TargetToolNamePolicy::PreserveSourceName => provider_tool.name.clone(),
            TargetToolNamePolicy::Fixed(name) => name.to_string(),
        };

        Tool::provider_defined(self.target_tool_id, tool_name).with_args((self.args_mapper)(
            index,
            &provider_tool.args,
            report,
        ))
    }

    pub(crate) fn choice_name(self, provider_tool: &ProviderDefinedTool) -> String {
        self.choice_name
            .map(str::to_string)
            .unwrap_or_else(|| match self.target_tool_name {
                TargetToolNamePolicy::PreserveSourceName => provider_tool.name.clone(),
                TargetToolNamePolicy::Fixed(name) => name.to_string(),
            })
    }

    pub(crate) fn aliases(self, provider_tool: &ProviderDefinedTool) -> Vec<String> {
        let mut aliases = Vec::with_capacity(1 + self.aliases.len());
        aliases.push(provider_tool.name.clone());
        aliases.extend(self.aliases.iter().map(|alias| (*alias).to_string()));
        aliases
    }
}

pub(crate) fn find_provider_tool_translation_rule<'a>(
    provider_tool: &ProviderDefinedTool,
    rules: &'a [ProviderToolTranslationRule],
) -> Option<&'a ProviderToolTranslationRule> {
    rules.iter().find(|rule| rule.matches(provider_tool))
}
