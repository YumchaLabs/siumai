#![cfg(feature = "google-vertex")]

use siumai::provider_ext::google_vertex;
use siumai_core::types::Tool;

#[test]
fn google_vertex_tools_surface_exports_expected_ids() {
    fn assert_provider_tool(tool: Tool, id: &str, name: &str) {
        let Tool::ProviderDefined(pd) = tool else {
            panic!("expected Tool::ProviderDefined");
        };
        assert_eq!(pd.id, id);
        assert_eq!(pd.name, name);
    }

    assert_provider_tool(
        google_vertex::tools::code_execution(),
        "google.code_execution",
        "code_execution",
    );

    assert_provider_tool(
        google_vertex::tools::url_context(),
        "google.url_context",
        "url_context",
    );

    assert_provider_tool(
        google_vertex::tools::enterprise_web_search(),
        "google.enterprise_web_search",
        "enterprise_web_search",
    );

    assert_provider_tool(
        google_vertex::tools::google_maps(),
        "google.google_maps",
        "google_maps",
    );

    let tool = google_vertex::tools::google_search().build();
    assert_provider_tool(tool, "google.google_search", "google_search");

    let tool = google_vertex::tools::file_search().build();
    assert_provider_tool(tool, "google.file_search", "file_search");

    let tool = google_vertex::tools::vertex_rag_store("ragCorpora/123").build();
    assert_provider_tool(tool, "google.vertex_rag_store", "vertex_rag_store");
}
