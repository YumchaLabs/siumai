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
    OpenAiNativeRuntime, invalid_input, json_body, target, target_with_query, validate_metadata,
    validate_resource_id,
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
                target(format!("conversations/{conversation_id}"))?,
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
                target(format!("conversations/{conversation_id}"))?,
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
                target(format!("conversations/{conversation_id}"))?,
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
                target(format!("conversations/{conversation_id}/items"))?,
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
                target_with_query(
                    &format!("conversations/{conversation_id}/items"),
                    list.query(),
                )?,
                RequestBody::Empty,
                ReplaySafety::SemanticallyIdempotent,
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
