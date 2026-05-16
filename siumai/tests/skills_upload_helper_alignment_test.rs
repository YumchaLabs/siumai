use siumai::prelude::unified::{ProviderReference, Warning};
use siumai::skills::{
    self, SkillUploadProvider, UploadSkillFile, UploadSkillOptions, UploadSkillPayload,
    UploadSkillResult,
};
use siumai_core::error::LlmError;
use siumai_core::traits::SkillsCapability;
use std::borrow::Cow;
use std::sync::{Arc, Mutex};

#[derive(Clone, Default)]
struct MockSkillsClient {
    payloads: Arc<Mutex<Vec<UploadSkillPayload>>>,
    next_result: Arc<Mutex<Option<UploadSkillResult>>>,
}

impl MockSkillsClient {
    fn with_result(result: UploadSkillResult) -> Self {
        Self {
            payloads: Arc::new(Mutex::new(Vec::new())),
            next_result: Arc::new(Mutex::new(Some(result))),
        }
    }

    fn take_payload(&self) -> UploadSkillPayload {
        self.payloads
            .lock()
            .expect("payload lock")
            .pop()
            .expect("captured payload")
    }
}

#[async_trait::async_trait]
impl SkillsCapability for MockSkillsClient {
    async fn upload_skill(
        &self,
        payload: UploadSkillPayload,
    ) -> Result<UploadSkillResult, LlmError> {
        self.payloads.lock().expect("payload lock").push(payload);
        self.next_result
            .lock()
            .expect("result lock")
            .clone()
            .ok_or_else(|| LlmError::InvalidInput("missing mock result".to_string()))
    }
}

impl SkillUploadProvider for MockSkillsClient {
    fn upload_skill_provider_id(&self) -> Cow<'static, str> {
        Cow::Borrowed("mock-provider")
    }
}

fn sample_result() -> UploadSkillResult {
    UploadSkillResult {
        provider_reference: ProviderReference::single("mock-provider", "skill_123"),
        display_title: Some("My Skill".to_string()),
        name: Some("my-skill".to_string()),
        description: Some("A mocked skill".to_string()),
        latest_version: Some("1".to_string()),
        provider_metadata: Some(std::collections::HashMap::from([(
            "mock-provider".to_string(),
            serde_json::json!({ "source": "custom" }),
        )])),
        warnings: vec![Warning::unsupported("displayTitle", None::<String>)],
    }
}

#[tokio::test]
async fn upload_delegates_to_api_with_files_and_display_title() {
    let api = MockSkillsClient::with_result(sample_result());
    let files = vec![UploadSkillFile::base64("test.ts", "aGVsbG8=")];

    skills::upload(
        &api,
        files.clone(),
        UploadSkillOptions::new().with_display_title("My Skill"),
    )
    .await
    .expect("upload result");

    let payload = api.take_payload();
    assert_eq!(payload.files, files);
    assert_eq!(payload.display_title.as_deref(), Some("My Skill"));
    assert!(payload.provider_options.is_empty());
}

#[tokio::test]
async fn upload_returns_provider_reference_and_warnings_from_api() {
    let api = MockSkillsClient::with_result(sample_result());

    let result = skills::upload(
        &api,
        vec![UploadSkillFile::bytes("test.ts", b"hello".to_vec())],
        UploadSkillOptions::new(),
    )
    .await
    .expect("upload result");

    assert_eq!(
        result.provider_reference,
        ProviderReference::single("mock-provider", "skill_123")
    );
    assert_eq!(result.display_title.as_deref(), Some("My Skill"));
    assert!(
        result
            .provider_metadata
            .as_ref()
            .and_then(|metadata| metadata.get("mock-provider"))
            .and_then(|metadata| metadata.as_object())
            .is_some(),
        "expected provider metadata to keep the AI SDK provider->object root"
    );
    assert_eq!(
        result
            .provider_metadata
            .as_ref()
            .and_then(|metadata| metadata.get("mock-provider"))
            .and_then(|metadata| metadata.get("source")),
        Some(&serde_json::json!("custom"))
    );
    assert_eq!(
        result.warnings,
        vec![Warning::unsupported("displayTitle", None::<String>)]
    );
}

#[tokio::test]
async fn upload_passes_provider_options_to_api() {
    let api = MockSkillsClient::with_result(sample_result());

    skills::upload(
        &api,
        vec![UploadSkillFile::bytes("test.ts", b"hello".to_vec())],
        UploadSkillOptions::new()
            .with_provider_option("anthropic", serde_json::json!({ "custom": "value" })),
    )
    .await
    .expect("upload result");

    let payload = api.take_payload();
    assert_eq!(
        payload.provider_options.get("anthropic"),
        Some(&serde_json::json!({ "custom": "value" }))
    );
}
