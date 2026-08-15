use std::sync::Arc;

use rmcp::model::Tool;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use siumai_core::ToolSpec;
use siumai_runtime::tool::{ToolArgumentError, ToolBinding, ToolSet};

use crate::{McpClient, McpError};

/// Fingerprint of one complete, sorted MCP tool catalog.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct McpCatalogFingerprint(Arc<str>);

impl McpCatalogFingerprint {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for McpCatalogFingerprint {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// One discovered MCP tool and its lossless bounded native definition.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct McpToolDefinition {
    remote_name: String,
    exposed_name: String,
    description: Option<String>,
    input_schema: Value,
    native_definition: Value,
    revision: Arc<str>,
}

impl std::fmt::Debug for McpToolDefinition {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("McpToolDefinition")
            .field("remote_name_bytes", &self.remote_name.len())
            .field("exposed_name_bytes", &self.exposed_name.len())
            .field(
                "description_bytes",
                &self.description.as_ref().map(String::len),
            )
            .field(
                "input_schema_bytes",
                &serde_json::to_vec(&self.input_schema)
                    .ok()
                    .map(|bytes| bytes.len()),
            )
            .field(
                "native_definition_bytes",
                &serde_json::to_vec(&self.native_definition)
                    .ok()
                    .map(|bytes| bytes.len()),
            )
            .field("revision_bytes", &self.revision.len())
            .finish()
    }
}

impl McpToolDefinition {
    pub fn remote_name(&self) -> &str {
        &self.remote_name
    }

    pub fn exposed_name(&self) -> &str {
        &self.exposed_name
    }

    pub fn description(&self) -> Option<&str> {
        self.description.as_deref()
    }

    pub fn input_schema(&self) -> &Value {
        &self.input_schema
    }

    /// Return the original MCP declaration, including inert annotations and metadata.
    pub fn native_definition(&self) -> &Value {
        &self.native_definition
    }

    pub fn revision(&self) -> &str {
        &self.revision
    }
}

/// A discovered catalog plus immutable runtime bindings.
#[derive(Debug, Clone)]
pub struct McpToolCatalog {
    definitions: Arc<[McpToolDefinition]>,
    bindings: ToolSet,
    fingerprint: McpCatalogFingerprint,
}

impl McpToolCatalog {
    pub fn definitions(&self) -> &[McpToolDefinition] {
        &self.definitions
    }

    pub fn tool_set(&self) -> ToolSet {
        self.bindings.clone()
    }

    pub fn fingerprint(&self) -> &McpCatalogFingerprint {
        &self.fingerprint
    }

    pub fn len(&self) -> usize {
        self.definitions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.definitions.is_empty()
    }

    pub(crate) fn from_remote(client: &McpClient, mut tools: Vec<Tool>) -> Result<Self, McpError> {
        tools.sort_by(|left, right| left.name.as_ref().cmp(right.name.as_ref()));

        let mut definitions = Vec::with_capacity(tools.len());
        for tool in tools {
            let remote_name = tool.name.to_string();
            let native_definition = serde_json::to_value(&tool)
                .map_err(|error| McpError::Encoding(error.to_string()))?;
            let encoded_size = serde_json::to_vec(&native_definition)
                .map_err(|error| McpError::Encoding(error.to_string()))?
                .len();
            if encoded_size > client.config().limits().max_schema_bytes() {
                return Err(McpError::SchemaLimitExceeded {
                    tool: remote_name,
                    maximum: client.config().limits().max_schema_bytes(),
                });
            }

            let exposed_name = client.exposed_name(&remote_name)?;
            let input_schema = Value::Object((*tool.input_schema).clone());
            let revision = format!(
                "mcp-definition-v1:{}",
                siumai_runtime::tool::canonical_arguments_digest(&native_definition)
            );
            definitions.push(McpToolDefinition {
                remote_name,
                exposed_name,
                description: tool.description.map(|value| value.to_string()),
                input_schema,
                native_definition,
                revision: revision.into(),
            });
        }

        let fingerprint = catalog_fingerprint(&definitions);
        let bindings = definitions
            .iter()
            .map(|definition| build_binding(client, definition, &fingerprint))
            .collect::<Result<Vec<_>, _>>()?;
        let bindings =
            ToolSet::from_bindings(bindings).map_err(|error| McpError::ToolNameConflict {
                name: error.to_string(),
            })?;
        Ok(Self {
            definitions: definitions.into(),
            bindings,
            fingerprint,
        })
    }
}

fn build_binding(
    client: &McpClient,
    definition: &McpToolDefinition,
    fingerprint: &McpCatalogFingerprint,
) -> Result<ToolBinding, McpError> {
    let spec = ToolSpec::new(
        definition.exposed_name.clone(),
        definition.description.clone(),
        definition.input_schema.clone(),
    )
    .map_err(|error| McpError::InvalidToolDefinition {
        tool: definition.remote_name.clone(),
        message: error.to_string(),
    })?;
    let policy = client.config().tool_policy(&definition.remote_name);
    let binding_client = client.clone();
    let binding_remote_name = definition.remote_name.clone();
    let expected_catalog = fingerprint.clone();
    ToolBinding::from_fn(
        spec,
        definition.revision.to_string(),
        |arguments| {
            if arguments.is_object() {
                Ok(())
            } else {
                Err(ToolArgumentError::new(
                    "MCP tool arguments must be a JSON object",
                ))
            }
        },
        move |request| {
            let binding_client = binding_client.clone();
            let binding_remote_name = binding_remote_name.clone();
            let expected_catalog = expected_catalog.clone();
            async move {
                binding_client
                    .execute_bound_tool(
                        &binding_remote_name,
                        request.arguments(),
                        &expected_catalog,
                        request.call_id(),
                    )
                    .await
            }
        },
    )
    .map(|binding| {
        binding
            .with_effect(policy.effect())
            .with_concurrency(policy.concurrency())
            .with_approval_policy(policy.approval())
            .with_recovery_policy(policy.recovery())
    })
    .map_err(|error| McpError::InvalidToolDefinition {
        tool: definition.remote_name.clone(),
        message: error.to_string(),
    })
}

pub(crate) fn catalog_fingerprint(definitions: &[McpToolDefinition]) -> McpCatalogFingerprint {
    let mut digest = sha2::Sha256::new();
    use sha2::Digest;
    digest.update(b"siumai.mcp.catalog.v1");
    for definition in definitions {
        digest.update(definition.remote_name.as_bytes());
        digest.update([0]);
        digest.update(definition.exposed_name.as_bytes());
        digest.update([0]);
        digest.update(definition.revision.as_bytes());
        digest.update([0]);
    }
    let bytes = digest.finalize();
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        encoded.push_str(&format!("{byte:02x}"));
    }
    McpCatalogFingerprint(encoded.into())
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn tool_definition_debug_exposes_only_structural_metadata() {
        let definition = McpToolDefinition {
            remote_name: "remote-name-canary".to_string(),
            exposed_name: "exposed-name-canary".to_string(),
            description: Some("description-canary".to_string()),
            input_schema: json!({"secret": "schema-canary"}),
            native_definition: json!({"authorization": "native-canary"}),
            revision: Arc::from("revision-canary"),
        };

        let debug = format!("{definition:?}");
        assert!(debug.contains("remote_name_bytes"));
        assert!(debug.contains("native_definition_bytes"));
        for canary in [
            "remote-name-canary",
            "exposed-name-canary",
            "description-canary",
            "schema-canary",
            "native-canary",
            "revision-canary",
        ] {
            assert!(!debug.contains(canary));
        }
    }
}
