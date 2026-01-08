#![cfg(feature = "google-vertex")]

use siumai::prelude::unified::*;

struct EnvGuard {
    key: &'static str,
    previous: Option<String>,
}

impl EnvGuard {
    fn set(key: &'static str, value: &str) -> Self {
        let previous = std::env::var(key).ok();
        unsafe {
            std::env::set_var(key, value);
        }
        Self { key, previous }
    }

    fn remove(key: &'static str) -> Self {
        let previous = std::env::var(key).ok();
        unsafe {
            std::env::remove_var(key);
        }
        Self { key, previous }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        match &self.previous {
            Some(v) => unsafe {
                std::env::set_var(self.key, v);
            },
            None => unsafe {
                std::env::remove_var(self.key);
            },
        }
    }
}

#[tokio::test]
async fn vertex_builder_uses_express_base_url_when_api_key_is_set() {
    let _g1 = EnvGuard::remove("GOOGLE_VERTEX_API_KEY");
    let _g2 = EnvGuard::remove("GOOGLE_VERTEX_PROJECT");
    let _g3 = EnvGuard::remove("GOOGLE_VERTEX_LOCATION");

    let siumai = Siumai::builder()
        .vertex()
        .api_key("test-api-key")
        .model("gemini-2.5-pro")
        .build()
        .await
        .expect("build vertex client");

    let inner = siumai
        .downcast_client::<siumai::provider_ext::google_vertex::GoogleVertexClient>()
        .expect("downcast GoogleVertexClient");

    assert_eq!(
        inner.base_url(),
        siumai::experimental::utils::vertex::GOOGLE_VERTEX_EXPRESS_BASE_URL
    );
}

#[tokio::test]
async fn vertex_builder_uses_env_project_location_when_no_base_url_and_no_api_key() {
    let _g1 = EnvGuard::remove("GOOGLE_VERTEX_API_KEY");
    let _g2 = EnvGuard::set("GOOGLE_VERTEX_PROJECT", "test-project");
    let _g3 = EnvGuard::set("GOOGLE_VERTEX_LOCATION", "us-central1");

    let siumai = Siumai::builder()
        .vertex()
        .model("gemini-2.5-pro")
        .build()
        .await
        .expect("build vertex client");

    let inner = siumai
        .downcast_client::<siumai::provider_ext::google_vertex::GoogleVertexClient>()
        .expect("downcast GoogleVertexClient");

    assert_eq!(
        inner.base_url(),
        "https://us-central1-aiplatform.googleapis.com/v1beta1/projects/test-project/locations/us-central1/publishers/google"
    );
}

#[tokio::test]
async fn vertex_builder_uses_global_host_when_location_is_global() {
    let _g1 = EnvGuard::remove("GOOGLE_VERTEX_API_KEY");
    let _g2 = EnvGuard::set("GOOGLE_VERTEX_PROJECT", "test-project");
    let _g3 = EnvGuard::set("GOOGLE_VERTEX_LOCATION", "global");

    let siumai = Siumai::builder()
        .vertex()
        .model("gemini-2.5-pro")
        .build()
        .await
        .expect("build vertex client");

    let inner = siumai
        .downcast_client::<siumai::provider_ext::google_vertex::GoogleVertexClient>()
        .expect("downcast GoogleVertexClient");

    assert_eq!(
        inner.base_url(),
        "https://aiplatform.googleapis.com/v1beta1/projects/test-project/locations/global/publishers/google"
    );
}

#[tokio::test]
async fn vertex_builder_uses_custom_base_url_when_provided() {
    let _g1 = EnvGuard::remove("GOOGLE_VERTEX_API_KEY");
    let _g2 = EnvGuard::set("GOOGLE_VERTEX_PROJECT", "test-project");
    let _g3 = EnvGuard::set("GOOGLE_VERTEX_LOCATION", "us-central1");

    let siumai = Siumai::builder()
        .vertex()
        .base_url("https://custom-endpoint.example.com")
        .model("gemini-2.5-pro")
        .build()
        .await
        .expect("build vertex client");

    let inner = siumai
        .downcast_client::<siumai::provider_ext::google_vertex::GoogleVertexClient>()
        .expect("downcast GoogleVertexClient");

    assert_eq!(inner.base_url(), "https://custom-endpoint.example.com");
}
