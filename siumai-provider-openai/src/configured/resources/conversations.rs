use std::fmt;
use std::sync::Arc;

use http::Method;
use siumai_core::{CallOptions, Error};
use siumai_protocol_openai::resources::{
    OpenAiConversation, OpenAiConversationCreateRequest, OpenAiConversationDeleted,
    OpenAiConversationItem, OpenAiConversationItemsCreateRequest, OpenAiConversationUpdateRequest,
    OpenAiCursorPage, OpenAiListOrder,
};
use siumai_transport::{ReplaySafety, RequestBody};

use super::super::provider::OpenAiRuntime;
use super::common::{
    OpenAiNativeRuntime, invalid_input, json_body, target, target_with_segments,
    target_with_segments_and_query, validate_metadata, validate_resource_id,
};

const MAX_ITEMS_PER_MUTATION: usize = 20;

/// Typed options for listing the items stored in one conversation.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct OpenAiConversationItemsListOptions {
    pub after: Option<String>,
    pub limit: Option<u8>,
    pub order: Option<OpenAiListOrder>,
    pub include_web_search_sources: bool,
}

impl OpenAiConversationItemsListOptions {
    fn validate(&self) -> Result<(), Error> {
        if let Some(after) = &self.after {
            validate_resource_id(after)?;
        }
        if self.limit == Some(0) {
            return Err(invalid_input(
                "OpenAI conversation item limit must be between 1 and 100",
            ));
        }
        Ok(())
    }

    fn query(self) -> Vec<(&'static str, String)> {
        let mut pairs = Vec::new();
        if let Some(after) = self.after {
            pairs.push(("after", after));
        }
        if let Some(limit) = self.limit {
            pairs.push(("limit", limit.to_string()));
        }
        if let Some(order) = self.order {
            pairs.push(("order", order.as_str().to_string()));
        }
        if self.include_web_search_sources {
            pairs.push(("include", "web_search_call.action.sources".to_string()));
        }
        pairs
    }
}

/// Provider-owned OpenAI Conversations lifecycle client.
#[derive(Clone)]
pub struct OpenAiConversations {
    runtime: OpenAiNativeRuntime,
}

impl OpenAiConversations {
    pub(crate) fn new(runtime: Arc<OpenAiRuntime>) -> Self {
        Self {
            runtime: OpenAiNativeRuntime::new(runtime),
        }
    }

    pub async fn create(
        &self,
        request: OpenAiConversationCreateRequest,
    ) -> Result<OpenAiConversation, Error> {
        self.create_with_options(request, CallOptions::default())
            .await
    }

    pub async fn create_with_options(
        &self,
        request: OpenAiConversationCreateRequest,
        options: CallOptions,
    ) -> Result<OpenAiConversation, Error> {
        validate_items(request.items.len(), true)?;
        validate_metadata(&request.metadata)?;
        self.runtime
            .execute_json(
                Method::POST,
                target("conversations")?,
                json_body(&request)?,
                ReplaySafety::Never,
                options,
            )
            .await
    }

    pub async fn retrieve(&self, conversation_id: &str) -> Result<OpenAiConversation, Error> {
        self.retrieve_with_options(conversation_id, CallOptions::default())
            .await
    }

    pub async fn retrieve_with_options(
        &self,
        conversation_id: &str,
        options: CallOptions,
    ) -> Result<OpenAiConversation, Error> {
        validate_resource_id(conversation_id)?;
        self.runtime
            .execute_json(
                Method::GET,
                target_with_segments("conversations", [conversation_id])?,
                RequestBody::Empty,
                ReplaySafety::SemanticallyIdempotent,
                options,
            )
            .await
    }

    pub async fn update(
        &self,
        conversation_id: &str,
        request: OpenAiConversationUpdateRequest,
    ) -> Result<OpenAiConversation, Error> {
        self.update_with_options(conversation_id, request, CallOptions::default())
            .await
    }

    pub async fn update_with_options(
        &self,
        conversation_id: &str,
        request: OpenAiConversationUpdateRequest,
        options: CallOptions,
    ) -> Result<OpenAiConversation, Error> {
        validate_resource_id(conversation_id)?;
        validate_metadata(&request.metadata)?;
        self.runtime
            .execute_json(
                Method::POST,
                target_with_segments("conversations", [conversation_id])?,
                json_body(&request)?,
                ReplaySafety::Never,
                options,
            )
            .await
    }

    pub async fn delete(&self, conversation_id: &str) -> Result<OpenAiConversationDeleted, Error> {
        self.delete_with_options(conversation_id, CallOptions::default())
            .await
    }

    pub async fn delete_with_options(
        &self,
        conversation_id: &str,
        options: CallOptions,
    ) -> Result<OpenAiConversationDeleted, Error> {
        validate_resource_id(conversation_id)?;
        self.runtime
            .execute_json(
                Method::DELETE,
                target_with_segments("conversations", [conversation_id])?,
                RequestBody::Empty,
                ReplaySafety::Never,
                options,
            )
            .await
    }

    pub async fn create_items(
        &self,
        conversation_id: &str,
        request: OpenAiConversationItemsCreateRequest,
    ) -> Result<OpenAiCursorPage<OpenAiConversationItem>, Error> {
        self.create_items_with_options(conversation_id, request, CallOptions::default())
            .await
    }

    pub async fn create_items_with_options(
        &self,
        conversation_id: &str,
        request: OpenAiConversationItemsCreateRequest,
        options: CallOptions,
    ) -> Result<OpenAiCursorPage<OpenAiConversationItem>, Error> {
        validate_resource_id(conversation_id)?;
        validate_items(request.items.len(), false)?;
        self.runtime
            .execute_json(
                Method::POST,
                target_with_segments("conversations", [conversation_id, "items"])?,
                json_body(&request)?,
                ReplaySafety::Never,
                options,
            )
            .await
    }

    pub async fn list_items(
        &self,
        conversation_id: &str,
        list: OpenAiConversationItemsListOptions,
    ) -> Result<OpenAiCursorPage<OpenAiConversationItem>, Error> {
        self.list_items_with_options(conversation_id, list, CallOptions::default())
            .await
    }

    pub async fn list_items_with_options(
        &self,
        conversation_id: &str,
        list: OpenAiConversationItemsListOptions,
        options: CallOptions,
    ) -> Result<OpenAiCursorPage<OpenAiConversationItem>, Error> {
        validate_resource_id(conversation_id)?;
        list.validate()?;
        self.runtime
            .execute_json(
                Method::GET,
                target_with_segments_and_query(
                    "conversations",
                    [conversation_id, "items"],
                    list.query(),
                )?,
                RequestBody::Empty,
                ReplaySafety::SemanticallyIdempotent,
                options,
            )
            .await
    }

    /// Retrieve one provider-native item from an OpenAI conversation.
    pub async fn retrieve_item(
        &self,
        conversation_id: &str,
        item_id: &str,
    ) -> Result<OpenAiConversationItem, Error> {
        self.retrieve_item_with_options(conversation_id, item_id, CallOptions::default())
            .await
    }

    /// Retrieve one provider-native item with per-call transport options.
    pub async fn retrieve_item_with_options(
        &self,
        conversation_id: &str,
        item_id: &str,
        options: CallOptions,
    ) -> Result<OpenAiConversationItem, Error> {
        validate_resource_id(conversation_id)?;
        validate_resource_id(item_id)?;
        self.runtime
            .execute_json(
                Method::GET,
                target_with_segments("conversations", [conversation_id, "items", item_id])?,
                RequestBody::Empty,
                ReplaySafety::SemanticallyIdempotent,
                options,
            )
            .await
    }

    /// Delete one provider-native item and return the updated conversation.
    pub async fn delete_item(
        &self,
        conversation_id: &str,
        item_id: &str,
    ) -> Result<OpenAiConversation, Error> {
        self.delete_item_with_options(conversation_id, item_id, CallOptions::default())
            .await
    }

    /// Delete one provider-native item with per-call transport options.
    pub async fn delete_item_with_options(
        &self,
        conversation_id: &str,
        item_id: &str,
        options: CallOptions,
    ) -> Result<OpenAiConversation, Error> {
        validate_resource_id(conversation_id)?;
        validate_resource_id(item_id)?;
        self.runtime
            .execute_json(
                Method::DELETE,
                target_with_segments("conversations", [conversation_id, "items", item_id])?,
                RequestBody::Empty,
                ReplaySafety::Never,
                options,
            )
            .await
    }
}

impl fmt::Debug for OpenAiConversations {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OpenAiConversations")
            .field("runtime", &self.runtime)
            .finish()
    }
}

fn validate_items(count: usize, empty_allowed: bool) -> Result<(), Error> {
    if count > MAX_ITEMS_PER_MUTATION || (!empty_allowed && count == 0) {
        return Err(invalid_input(
            "OpenAI conversation mutations accept between 1 and 20 items",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use siumai_core::{ErrorKind, ReplayDomain, ReplayDomainId};
    use siumai_transport::EndpointConfig;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use crate::configured::{OpenAiCredential, OpenAiProvider};

    fn provider(server: &MockServer) -> OpenAiProvider {
        OpenAiProvider::builder(OpenAiCredential::unauthenticated())
            .with_endpoint(EndpointConfig::local_explicit(format!("{}/v1", server.uri())).unwrap())
            .with_replay_domain(ReplayDomain::custom(
                ReplayDomainId::new("openai-conversation-item-fixture").unwrap(),
            ))
            .build()
            .unwrap()
    }

    #[tokio::test]
    async fn retrieve_item_encodes_opaque_ids_once_and_preserves_future_payloads() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path(
                "/v1/conversations/conv%2Fpart%252F%E8%B5%84%E6%BA%90/items/item%3Fquery%23fragment%5Cchild",
            ))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "type": "future_item",
                "id": "item-private-id",
                "private_payload": "conversation-item-private-payload",
                "future": {"kept": true}
            })))
            .expect(1)
            .mount(&server)
            .await;

        let item = provider(&server)
            .conversations()
            .retrieve_item("conv/part%2F资源", "item?query#fragment\\child")
            .await
            .unwrap();

        assert_eq!(item.as_value()["future"]["kept"], true);
        let debug = format!("{item:?}");
        assert!(!debug.contains("item-private-id"));
        assert!(!debug.contains("conversation-item-private-payload"));
    }

    #[tokio::test]
    async fn delete_item_returns_the_updated_conversation() {
        let server = MockServer::start().await;
        Mock::given(method("DELETE"))
            .and(path("/v1/conversations/conv_1/items/item_1"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "conv_private_id",
                "object": "conversation",
                "created_at": 1,
                "metadata": {"state": "updated"},
                "private_payload": "conversation-private-payload"
            })))
            .expect(1)
            .mount(&server)
            .await;

        let conversation = provider(&server)
            .conversations()
            .delete_item("conv_1", "item_1")
            .await
            .unwrap();

        assert_eq!(conversation.metadata["state"], "updated");
        assert_eq!(
            conversation.extra["private_payload"],
            "conversation-private-payload"
        );
        let debug = format!("{conversation:?}");
        assert!(!debug.contains("conv_private_id"));
        assert!(!debug.contains("conversation-private-payload"));
    }

    #[tokio::test]
    async fn retrieve_item_reports_missing_resources_without_disclosing_provider_text() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/conversations/conv_missing/items/item_missing"))
            .respond_with(ResponseTemplate::new(404).set_body_json(serde_json::json!({
                "error": {
                    "code": "resource_not_found",
                    "message": "missing-conversation-item-private-payload"
                }
            })))
            .expect(1)
            .mount(&server)
            .await;

        let error = provider(&server)
            .conversations()
            .retrieve_item("conv_missing", "item_missing")
            .await
            .unwrap_err();

        assert_eq!(error.kind(), ErrorKind::Provider);
        assert_eq!(
            error
                .diagnostics()
                .and_then(|diagnostics| diagnostics.status()),
            Some(404)
        );
        assert!(!format!("{error:?}").contains("missing-conversation-item-private-payload"));
        assert!(
            !error
                .to_string()
                .contains("missing-conversation-item-private-payload")
        );
    }

    #[tokio::test]
    async fn delete_item_keeps_classified_provider_details_sensitive() {
        let server = MockServer::start().await;
        Mock::given(method("DELETE"))
            .and(path("/v1/conversations/conv_secret/items/item_secret"))
            .respond_with(ResponseTemplate::new(429).set_body_json(serde_json::json!({
                "error": {
                    "type": "rate_limit_error",
                    "code": "rate_limit_exceeded",
                    "message": "conv_secret item_secret private-provider-message"
                }
            })))
            .expect(1)
            .mount(&server)
            .await;

        let error = provider(&server)
            .conversations()
            .delete_item("conv_secret", "item_secret")
            .await
            .unwrap_err();

        assert_eq!(error.kind(), ErrorKind::RateLimited);
        assert_eq!(
            error
                .diagnostics()
                .and_then(|diagnostics| diagnostics.provider_code()),
            Some("rate_limit_exceeded")
        );
        for public in [format!("{error:?}"), error.to_string()] {
            assert!(!public.contains("conv_secret"));
            assert!(!public.contains("item_secret"));
            assert!(!public.contains("private-provider-message"));
        }
    }
}
