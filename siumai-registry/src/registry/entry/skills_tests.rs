use super::*;
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Clone)]
struct BridgeSkillsClient;

#[async_trait::async_trait]
impl crate::traits::SkillsCapability for BridgeSkillsClient {
    async fn upload_skill(
        &self,
        request: crate::types::SkillUploadRequest,
    ) -> Result<crate::types::SkillUploadResult, LlmError> {
        Ok(crate::types::SkillUploadResult {
            provider_reference: crate::types::ProviderReference::single(
                "testprov_skills",
                format!("skill:{}", request.files[0].path),
            ),
            display_title: request.display_title,
            name: Some("bridge-skill".to_string()),
            description: Some("registry bridge".to_string()),
            latest_version: Some("1".to_string()),
            provider_metadata: Some(HashMap::from([(
                "source".to_string(),
                serde_json::json!("bridge"),
            )])),
            warnings: Vec::new(),
        })
    }
}

impl LlmClient for BridgeSkillsClient {
    fn provider_id(&self) -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed("testprov_skills")
    }

    fn supported_models(&self) -> Vec<String> {
        vec!["skill-model".into()]
    }

    fn capabilities(&self) -> crate::traits::ProviderCapabilities {
        crate::traits::ProviderCapabilities::new().with_custom_feature("skills", true)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn clone_box(&self) -> Box<dyn LlmClient> {
        Box::new(self.clone())
    }

    fn as_skills_capability(&self) -> Option<&dyn crate::traits::SkillsCapability> {
        Some(self)
    }
}

struct BridgeSkillsFactory;

#[async_trait::async_trait]
impl ProviderFactory for BridgeSkillsFactory {
    async fn compat_language_client(
        &self,
        _model_id: &str,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        Ok(Arc::new(BridgeSkillsClient))
    }

    fn provider_id(&self) -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed("testprov_skills")
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new()
            .with_chat()
            .with_custom_feature("skills", true)
    }
}

#[tokio::test]
async fn language_model_handle_delegates_skills_capability() {
    let mut providers = HashMap::new();
    providers.insert(
        "testprov_skills".to_string(),
        Arc::new(BridgeSkillsFactory) as Arc<dyn ProviderFactory>,
    );

    let reg = create_provider_registry(providers, None);
    let handle = reg.language_model("testprov_skills:skill-model").unwrap();

    let uploaded = handle
        .upload_skill(crate::types::SkillUploadRequest::new(vec![
            crate::types::SkillUploadFile::base64("index.ts", "aGVsbG8="),
        ]))
        .await
        .unwrap();

    assert_eq!(
        uploaded.provider_reference.get("testprov_skills"),
        Some("skill:index.ts")
    );
    assert_eq!(uploaded.name.as_deref(), Some("bridge-skill"));
    assert_eq!(
        uploaded
            .provider_metadata
            .as_ref()
            .and_then(|metadata| metadata.get("source")),
        Some(&serde_json::json!("bridge"))
    );
    assert!(handle.as_skills_capability().is_some());
}
