//! OpenAI Rerank Standard (external)

use siumai_core::error::LlmError;
use siumai_core::execution::rerank::{
    RerankInput, RerankItem, RerankOutput, RerankRequestTransformer, RerankResponseTransformer,
};
use std::sync::Arc;

#[derive(Clone)]
pub struct OpenAiRerankStandard {
    adapter: Option<Arc<dyn OpenAiRerankAdapter>>,
}

impl OpenAiRerankStandard {
    pub fn new() -> Self {
        Self { adapter: None }
    }
    pub fn with_adapter(adapter: Arc<dyn OpenAiRerankAdapter>) -> Self {
        Self {
            adapter: Some(adapter),
        }
    }

    pub fn create_request_transformer(
        &self,
        provider_id: &str,
    ) -> Arc<dyn RerankRequestTransformer> {
        Arc::new(OpenAiRerankRequestTx {
            provider_id: provider_id.to_string(),
            adapter: self.adapter.clone(),
        })
    }
    pub fn create_response_transformer(
        &self,
        provider_id: &str,
    ) -> Arc<dyn RerankResponseTransformer> {
        Arc::new(OpenAiRerankResponseTx {
            provider_id: provider_id.to_string(),
            adapter: self.adapter.clone(),
        })
    }
}

impl Default for OpenAiRerankStandard {
    fn default() -> Self {
        Self::new()
    }
}

pub trait OpenAiRerankAdapter: Send + Sync {
    fn transform_request(
        &self,
        _req: &RerankInput,
        _body: &mut serde_json::Value,
    ) -> Result<(), LlmError> {
        Ok(())
    }
    fn transform_response(&self, _resp: &mut serde_json::Value) -> Result<(), LlmError> {
        Ok(())
    }
    fn rerank_endpoint(&self) -> &str {
        "/rerank"
    }
}

#[derive(Clone)]
struct OpenAiRerankRequestTx {
    provider_id: String,
    adapter: Option<Arc<dyn OpenAiRerankAdapter>>,
}

impl RerankRequestTransformer for OpenAiRerankRequestTx {
    fn provider_id(&self) -> &str {
        &self.provider_id
    }
    fn transform(&self, req: &RerankInput) -> Result<serde_json::Value, LlmError> {
        let mut body = serde_json::json!({
            "query": req.query,
            "documents": req.documents,
        });
        if let Some(m) = &req.model {
            body["model"] = serde_json::json!(m);
        }
        if let Some(n) = req.top_n {
            body["top_n"] = serde_json::json!(n);
        }
        if let Some(rd) = req.return_documents {
            body["return_documents"] = serde_json::json!(rd);
        }
        if let Some(obj) = body.as_object_mut() {
            for (k, v) in &req.extra {
                obj.insert(k.clone(), v.clone());
            }
        }
        if let Some(adapter) = &self.adapter {
            adapter.transform_request(req, &mut body)?;
        }
        Ok(body)
    }
}

#[derive(Clone)]
struct OpenAiRerankResponseTx {
    provider_id: String,
    adapter: Option<Arc<dyn OpenAiRerankAdapter>>,
}

impl RerankResponseTransformer for OpenAiRerankResponseTx {
    fn provider_id(&self) -> &str {
        &self.provider_id
    }
    fn transform_response(&self, raw: &serde_json::Value) -> Result<RerankOutput, LlmError> {
        let mut resp = raw.clone();
        if let Some(adapter) = &self.adapter {
            adapter.transform_response(&mut resp)?;
        }
        let id = resp
            .get("id")
            .and_then(|s| s.as_str())
            .map(|s| s.to_string());
        let results_v = resp
            .get("results")
            .and_then(|a| a.as_array())
            .ok_or_else(|| LlmError::ParseError("Missing 'results' field".into()))?;
        let mut results = Vec::with_capacity(results_v.len());
        for item in results_v {
            let index = item
                .get("index")
                .and_then(|n| n.as_u64())
                .ok_or_else(|| LlmError::ParseError("Missing 'index'".into()))?
                as u32;
            let relevance = item
                .get("relevance_score")
                .and_then(|n| n.as_f64())
                .ok_or_else(|| LlmError::ParseError("Missing 'relevance_score'".into()))?;
            let document = if let Some(doc) = item.get("document") {
                if doc.is_string() {
                    doc.as_str().map(|s| s.to_string())
                } else {
                    doc.get("text")
                        .and_then(|s| s.as_str())
                        .map(|s| s.to_string())
                }
            } else {
                None
            };
            results.push(RerankItem {
                index,
                relevance_score: relevance,
                document,
            });
        }
        let (input_tokens, output_tokens) = if let Some(u) = resp.get("usage") {
            let i = u.get("input_tokens").and_then(|n| n.as_u64()).unwrap_or(0) as u32;
            let o = u.get("output_tokens").and_then(|n| n.as_u64()).unwrap_or(0) as u32;
            (i, o)
        } else {
            (0, 0)
        };
        Ok(RerankOutput {
            id,
            results,
            input_tokens,
            output_tokens,
        })
    }
}
