use std::collections::BTreeSet;
use std::fmt;
use std::sync::Arc;

use http::Method;
use siumai_core::{CallOptions, Error};
use siumai_protocol_openai::resources::{
    OpenAiCursorPage, OpenAiListOrder, OpenAiVectorStore, OpenAiVectorStoreCreateRequest,
    OpenAiVectorStoreDeleted, OpenAiVectorStoreFile, OpenAiVectorStoreFileAttachRequest,
    OpenAiVectorStoreFileDeleted, OpenAiVectorStoreUpdateRequest,
};
use siumai_transport::{ReplaySafety, RequestBody};

use super::super::provider::OpenAiRuntime;
use super::common::{
    OpenAiNativeRuntime, invalid_input, json_body, target, target_with_query, validate_attributes,
    validate_bounded_text, validate_chunking, validate_metadata, validate_resource_id,
    validate_vector_store_expiration,
};

const MAX_VECTOR_STORE_NAME_BYTES: usize = 256;
const MAX_VECTOR_STORE_DESCRIPTION_BYTES: usize = 4_096;

/// Typed cursor options for listing OpenAI vector stores.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct OpenAiVectorStoreListOptions {
    pub after: Option<String>,
    pub before: Option<String>,
    pub limit: Option<u8>,
    pub order: Option<OpenAiListOrder>,
}

impl OpenAiVectorStoreListOptions {
    fn validate(&self) -> Result<(), Error> {
        if self.after.is_some() && self.before.is_some() {
            return Err(invalid_input(
                "OpenAI vector-store cursors are mutually exclusive",
            ));
        }
        if let Some(after) = &self.after {
            validate_resource_id(after)?;
        }
        if let Some(before) = &self.before {
            validate_resource_id(before)?;
        }
        if self.limit == Some(0) {
            return Err(invalid_input(
                "OpenAI vector-store limit must be between 1 and 100",
            ));
        }
        Ok(())
    }

    fn query(self) -> Vec<(&'static str, String)> {
        let mut pairs = Vec::new();
        if let Some(after) = self.after {
            pairs.push(("after", after));
        }
        if let Some(before) = self.before {
            pairs.push(("before", before));
        }
        if let Some(limit) = self.limit {
            pairs.push(("limit", limit.to_string()));
        }
        if let Some(order) = self.order {
            pairs.push(("order", order.as_str().to_string()));
        }
        pairs
    }
}

/// Server-side ingestion state filter for vector-store files.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenAiVectorStoreFileStatusFilter {
    InProgress,
    Completed,
    Failed,
    Cancelled,
}

impl OpenAiVectorStoreFileStatusFilter {
    const fn as_str(self) -> &'static str {
        match self {
            Self::InProgress => "in_progress",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
        }
    }
}

/// Typed cursor options for files attached to one vector store.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct OpenAiVectorStoreFileListOptions {
    pub after: Option<String>,
    pub before: Option<String>,
    pub limit: Option<u8>,
    pub order: Option<OpenAiListOrder>,
    pub status: Option<OpenAiVectorStoreFileStatusFilter>,
}

impl OpenAiVectorStoreFileListOptions {
    fn validate(&self) -> Result<(), Error> {
        if self.after.is_some() && self.before.is_some() {
            return Err(invalid_input(
                "OpenAI vector-store file cursors are mutually exclusive",
            ));
        }
        if let Some(after) = &self.after {
            validate_resource_id(after)?;
        }
        if let Some(before) = &self.before {
            validate_resource_id(before)?;
        }
        if self.limit == Some(0) {
            return Err(invalid_input(
                "OpenAI vector-store file limit must be between 1 and 100",
            ));
        }
        Ok(())
    }

    fn query(self) -> Vec<(&'static str, String)> {
        let mut pairs = Vec::new();
        if let Some(after) = self.after {
            pairs.push(("after", after));
        }
        if let Some(before) = self.before {
            pairs.push(("before", before));
        }
        if let Some(limit) = self.limit {
            pairs.push(("limit", limit.to_string()));
        }
        if let Some(order) = self.order {
            pairs.push(("order", order.as_str().to_string()));
        }
        if let Some(status) = self.status {
            pairs.push(("filter", status.as_str().to_string()));
        }
        pairs
    }
}

/// Provider-owned OpenAI Vector Stores lifecycle client.
#[derive(Clone)]
pub struct OpenAiVectorStores {
    runtime: OpenAiNativeRuntime,
}

impl OpenAiVectorStores {
    pub(crate) fn new(runtime: Arc<OpenAiRuntime>) -> Self {
        Self {
            runtime: OpenAiNativeRuntime::new(runtime),
        }
    }

    pub async fn create(
        &self,
        request: OpenAiVectorStoreCreateRequest,
    ) -> Result<OpenAiVectorStore, Error> {
        self.create_with_options(request, CallOptions::default())
            .await
    }

    pub async fn create_with_options(
        &self,
        request: OpenAiVectorStoreCreateRequest,
        options: CallOptions,
    ) -> Result<OpenAiVectorStore, Error> {
        validate_create(&request)?;
        self.runtime
            .execute_json(
                Method::POST,
                target("vector_stores")?,
                json_body(&request)?,
                ReplaySafety::Never,
                options,
            )
            .await
    }

    pub async fn retrieve(&self, vector_store_id: &str) -> Result<OpenAiVectorStore, Error> {
        self.retrieve_with_options(vector_store_id, CallOptions::default())
            .await
    }

    pub async fn retrieve_with_options(
        &self,
        vector_store_id: &str,
        options: CallOptions,
    ) -> Result<OpenAiVectorStore, Error> {
        validate_resource_id(vector_store_id)?;
        self.runtime
            .execute_json(
                Method::GET,
                target(format!("vector_stores/{vector_store_id}"))?,
                RequestBody::Empty,
                ReplaySafety::SemanticallyIdempotent,
                options,
            )
            .await
    }

    pub async fn update(
        &self,
        vector_store_id: &str,
        request: OpenAiVectorStoreUpdateRequest,
    ) -> Result<OpenAiVectorStore, Error> {
        self.update_with_options(vector_store_id, request, CallOptions::default())
            .await
    }

    pub async fn update_with_options(
        &self,
        vector_store_id: &str,
        request: OpenAiVectorStoreUpdateRequest,
        options: CallOptions,
    ) -> Result<OpenAiVectorStore, Error> {
        validate_resource_id(vector_store_id)?;
        validate_update(&request)?;
        self.runtime
            .execute_json(
                Method::POST,
                target(format!("vector_stores/{vector_store_id}"))?,
                json_body(&request)?,
                ReplaySafety::Never,
                options,
            )
            .await
    }

    pub async fn delete(&self, vector_store_id: &str) -> Result<OpenAiVectorStoreDeleted, Error> {
        self.delete_with_options(vector_store_id, CallOptions::default())
            .await
    }

    pub async fn delete_with_options(
        &self,
        vector_store_id: &str,
        options: CallOptions,
    ) -> Result<OpenAiVectorStoreDeleted, Error> {
        validate_resource_id(vector_store_id)?;
        self.runtime
            .execute_json(
                Method::DELETE,
                target(format!("vector_stores/{vector_store_id}"))?,
                RequestBody::Empty,
                ReplaySafety::Never,
                options,
            )
            .await
    }

    pub async fn list(
        &self,
        list: OpenAiVectorStoreListOptions,
    ) -> Result<OpenAiCursorPage<OpenAiVectorStore>, Error> {
        self.list_with_options(list, CallOptions::default()).await
    }

    pub async fn list_with_options(
        &self,
        list: OpenAiVectorStoreListOptions,
        options: CallOptions,
    ) -> Result<OpenAiCursorPage<OpenAiVectorStore>, Error> {
        list.validate()?;
        self.runtime
            .execute_json(
                Method::GET,
                target_with_query("vector_stores", list.query())?,
                RequestBody::Empty,
                ReplaySafety::SemanticallyIdempotent,
                options,
            )
            .await
    }

    pub async fn attach_file(
        &self,
        vector_store_id: &str,
        request: OpenAiVectorStoreFileAttachRequest,
    ) -> Result<OpenAiVectorStoreFile, Error> {
        self.attach_file_with_options(vector_store_id, request, CallOptions::default())
            .await
    }

    pub async fn attach_file_with_options(
        &self,
        vector_store_id: &str,
        request: OpenAiVectorStoreFileAttachRequest,
        options: CallOptions,
    ) -> Result<OpenAiVectorStoreFile, Error> {
        validate_resource_id(vector_store_id)?;
        validate_resource_id(&request.file_id)?;
        validate_attributes(&request.attributes)?;
        if let Some(strategy) = request.chunking_strategy {
            validate_chunking(strategy)?;
        }
        self.runtime
            .execute_json(
                Method::POST,
                target(format!("vector_stores/{vector_store_id}/files"))?,
                json_body(&request)?,
                ReplaySafety::Never,
                options,
            )
            .await
    }

    pub async fn list_files(
        &self,
        vector_store_id: &str,
        list: OpenAiVectorStoreFileListOptions,
    ) -> Result<OpenAiCursorPage<OpenAiVectorStoreFile>, Error> {
        self.list_files_with_options(vector_store_id, list, CallOptions::default())
            .await
    }

    pub async fn list_files_with_options(
        &self,
        vector_store_id: &str,
        list: OpenAiVectorStoreFileListOptions,
        options: CallOptions,
    ) -> Result<OpenAiCursorPage<OpenAiVectorStoreFile>, Error> {
        validate_resource_id(vector_store_id)?;
        list.validate()?;
        self.runtime
            .execute_json(
                Method::GET,
                target_with_query(
                    &format!("vector_stores/{vector_store_id}/files"),
                    list.query(),
                )?,
                RequestBody::Empty,
                ReplaySafety::SemanticallyIdempotent,
                options,
            )
            .await
    }

    pub async fn remove_file(
        &self,
        vector_store_id: &str,
        file_id: &str,
    ) -> Result<OpenAiVectorStoreFileDeleted, Error> {
        self.remove_file_with_options(vector_store_id, file_id, CallOptions::default())
            .await
    }

    pub async fn remove_file_with_options(
        &self,
        vector_store_id: &str,
        file_id: &str,
        options: CallOptions,
    ) -> Result<OpenAiVectorStoreFileDeleted, Error> {
        validate_resource_id(vector_store_id)?;
        validate_resource_id(file_id)?;
        self.runtime
            .execute_json(
                Method::DELETE,
                target(format!("vector_stores/{vector_store_id}/files/{file_id}"))?,
                RequestBody::Empty,
                ReplaySafety::Never,
                options,
            )
            .await
    }
}

impl fmt::Debug for OpenAiVectorStores {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OpenAiVectorStores")
            .field("runtime", &self.runtime)
            .finish()
    }
}

fn validate_create(request: &OpenAiVectorStoreCreateRequest) -> Result<(), Error> {
    if let Some(name) = &request.name {
        validate_bounded_text(
            name,
            MAX_VECTOR_STORE_NAME_BYTES,
            "OpenAI vector-store name is invalid",
        )?;
    }
    if let Some(description) = &request.description {
        validate_bounded_text(
            description,
            MAX_VECTOR_STORE_DESCRIPTION_BYTES,
            "OpenAI vector-store description is invalid",
        )?;
    }
    if let Some(expires) = request.expires_after {
        validate_vector_store_expiration(expires)?;
    }
    validate_metadata(&request.metadata)?;
    if let Some(strategy) = request.chunking_strategy {
        if request.file_ids.is_empty() {
            return Err(invalid_input(
                "OpenAI vector-store chunking requires at least one file ID",
            ));
        }
        validate_chunking(strategy)?;
    }
    let mut unique = BTreeSet::new();
    for file_id in &request.file_ids {
        validate_resource_id(file_id)?;
        if !unique.insert(file_id) {
            return Err(invalid_input("OpenAI vector-store file IDs must be unique"));
        }
    }
    Ok(())
}

fn validate_update(request: &OpenAiVectorStoreUpdateRequest) -> Result<(), Error> {
    if request.name.is_none() && request.expires_after.is_none() && request.metadata.is_none() {
        return Err(invalid_input("OpenAI vector-store update cannot be empty"));
    }
    if let Some(name) = &request.name {
        validate_bounded_text(
            name,
            MAX_VECTOR_STORE_NAME_BYTES,
            "OpenAI vector-store name is invalid",
        )?;
    }
    if let Some(expires) = request.expires_after {
        validate_vector_store_expiration(expires)?;
    }
    if let Some(metadata) = &request.metadata {
        validate_metadata(metadata)?;
    }
    Ok(())
}
