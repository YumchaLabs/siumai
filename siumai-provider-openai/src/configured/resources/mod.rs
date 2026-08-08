//! Provider-owned OpenAI resource lifecycles.

mod common;
mod conversations;
mod files;
pub(crate) mod skills;
mod vector_stores;

pub use common::OpenAiBinaryContent;
pub use conversations::{OpenAiConversationItemsListOptions, OpenAiConversations};
pub use files::{OpenAiFileListOptions, OpenAiFileUpload, OpenAiFiles};
pub use vector_stores::{
    OpenAiVectorStoreFileListOptions, OpenAiVectorStoreFileStatusFilter,
    OpenAiVectorStoreListOptions, OpenAiVectorStores,
};

pub use siumai_protocol_openai::resources::{
    OpenAiChunkingStrategy, OpenAiConversation, OpenAiConversationCreateRequest,
    OpenAiConversationDeleted, OpenAiConversationInputItem, OpenAiConversationItem,
    OpenAiConversationItemsCreateRequest, OpenAiConversationRole, OpenAiConversationUpdateRequest,
    OpenAiCursorPage, OpenAiFile, OpenAiFileDeleted, OpenAiFileExpirationAnchor,
    OpenAiFileExpiresAfter, OpenAiFilePurpose, OpenAiFileUploadPurpose, OpenAiListOrder,
    OpenAiMetadata, OpenAiResourceCodecError, OpenAiStaticChunkingSettings, OpenAiVectorStore,
    OpenAiVectorStoreCreateRequest, OpenAiVectorStoreDeleted, OpenAiVectorStoreExpiration,
    OpenAiVectorStoreExpirationAnchor, OpenAiVectorStoreFile, OpenAiVectorStoreFileAttachRequest,
    OpenAiVectorStoreFileCounts, OpenAiVectorStoreFileDeleted, OpenAiVectorStoreFileError,
    OpenAiVectorStoreUpdateRequest,
};

#[cfg(test)]
mod tests {
    use siumai_core::{ReplayDomain, ReplayDomainId};
    use siumai_transport::EndpointConfig;
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use super::skills::{OpenAiSkillFile, OpenAiSkillUpload, OpenAiSkillsProviderExt};
    use super::*;
    use crate::configured::{OpenAiCredential, OpenAiProvider};

    async fn provider(server: &MockServer) -> OpenAiProvider {
        OpenAiProvider::builder(OpenAiCredential::unauthenticated())
            .with_endpoint(EndpointConfig::local_explicit(format!("{}/v1", server.uri())).unwrap())
            .with_replay_domain(ReplayDomain::custom(
                ReplayDomainId::new("openai-resource-fixture").unwrap(),
            ))
            .build()
            .unwrap()
    }

    #[tokio::test]
    async fn conversations_create_uses_the_provider_owned_lifecycle() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/conversations"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "conv_1",
                "object": "conversation",
                "created_at": 1,
                "metadata": {},
                "future": true
            })))
            .expect(1)
            .mount(&server)
            .await;

        let conversation = provider(&server)
            .await
            .conversations()
            .create(OpenAiConversationCreateRequest::new().with_item(
                OpenAiConversationInputItem::message(OpenAiConversationRole::User, "hello"),
            ))
            .await
            .unwrap();

        assert_eq!(conversation.id, "conv_1");
        assert_eq!(conversation.extra["future"], true);
    }

    #[tokio::test]
    async fn files_upload_uses_bounded_multipart_and_preserves_future_fields() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/files"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "file_1",
                "object": "file",
                "bytes": 4,
                "created_at": 1,
                "filename": "note.txt",
                "purpose": "user_data",
                "future": true
            })))
            .expect(1)
            .mount(&server)
            .await;

        let file = provider(&server)
            .await
            .files()
            .upload(OpenAiFileUpload::new(
                "note.txt",
                "text/plain",
                b"data".to_vec(),
                OpenAiFileUploadPurpose::UserData,
            ))
            .await
            .unwrap();

        assert_eq!(file.id, "file_1");
        assert_eq!(file.extra["future"], true);
    }

    #[tokio::test]
    async fn vector_store_file_attach_keeps_the_resource_specific_shape() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/vector_stores/vs_1/files"))
            .and(header("openai-beta", "assistants=v2"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "file_1",
                "object": "vector_store.file",
                "created_at": 1,
                "last_error": null,
                "status": "completed",
                "usage_bytes": 4,
                "vector_store_id": "vs_1",
                "attributes": {},
                "future": true
            })))
            .expect(1)
            .mount(&server)
            .await;

        let file = provider(&server)
            .await
            .vector_stores()
            .attach_file("vs_1", OpenAiVectorStoreFileAttachRequest::new("file_1"))
            .await
            .unwrap();

        assert_eq!(file.vector_store_id, "vs_1");
        assert_eq!(file.extra["future"], true);
    }

    #[test]
    fn vector_store_expiration_enforces_the_official_range() {
        assert!(
            super::common::validate_vector_store_expiration(
                OpenAiVectorStoreExpiration::after_last_active(1),
            )
            .is_ok()
        );
        assert!(
            super::common::validate_vector_store_expiration(
                OpenAiVectorStoreExpiration::after_last_active(365),
            )
            .is_ok()
        );
        assert!(
            super::common::validate_vector_store_expiration(
                OpenAiVectorStoreExpiration::after_last_active(366),
            )
            .is_err()
        );
    }

    #[tokio::test]
    async fn skills_create_uses_directory_multipart_without_a_universal_resource_client() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/skills"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "skill_1",
                "object": "skill",
                "created_at": 1,
                "default_version": "1",
                "latest_version": "1",
                "future": true
            })))
            .expect(1)
            .mount(&server)
            .await;

        let skill = provider(&server)
            .await
            .skills()
            .create(OpenAiSkillUpload::new([OpenAiSkillFile::new(
                "SKILL.md",
                "text/markdown",
                b"# Skill".to_vec(),
            )]))
            .await
            .unwrap();

        assert_eq!(skill.id, "skill_1");
        assert_eq!(skill.extra["future"], true);
    }

    #[test]
    fn binary_content_debug_never_exposes_the_payload() {
        let sentinel = b"openai-binary-content-sentinel".to_vec();
        let content = OpenAiBinaryContent::new(sentinel.clone());

        assert_eq!(content.as_bytes(), sentinel);
        assert!(!format!("{content:?}").contains("openai-binary-content-sentinel"));
    }
}
