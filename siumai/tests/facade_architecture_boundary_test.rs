use std::fs;
use std::path::{Path, PathBuf};

fn crate_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
}

fn read_source(relative_path: &str) -> String {
    fs::read_to_string(crate_root().join(relative_path)).expect("read source file")
}

#[test]
fn facade_keeps_provider_extension_bodies_out_of_lib_rs() {
    let lib_rs = read_source("src/lib.rs");
    let provider_ext_rs = read_source("src/provider_ext.rs");

    assert!(
        lib_rs.contains("pub mod provider_ext;"),
        "siumai/src/lib.rs should declare provider_ext as an external module"
    );
    assert!(
        lib_rs.contains("pub use crate::provider_ext as providers;"),
        "siumai::providers should stay a thin alias for provider_ext"
    );
    assert!(
        !lib_rs.contains("pub mod provider_ext {"),
        "provider extension bodies must stay out of siumai/src/lib.rs"
    );

    for provider in ["openai", "anthropic", "gemini", "google_vertex", "xai"] {
        let declaration = format!("pub mod {provider};");
        assert!(
            provider_ext_rs.contains(&declaration),
            "provider_ext.rs should own the {provider} module declaration"
        );
    }
}

#[test]
fn gemini_model_catalog_stays_out_of_provider_reexport_glue() {
    let gemini_rs = read_source("src/provider_ext/gemini.rs");
    let gemini_models_rs = read_source("src/provider_ext/gemini/models.rs");

    assert!(
        gemini_rs.contains("pub mod models;"),
        "Gemini provider extension should declare the model catalog as a dedicated module"
    );
    assert!(
        !gemini_rs.contains("pub mod models {"),
        "Gemini model-id catalog should not be inlined in provider_ext/gemini.rs"
    );
    assert!(
        gemini_rs.contains("pub use models::{chat, embedding, image, model_sets, video};"),
        "Gemini model-id group paths should remain available from provider_ext::gemini"
    );

    for family_module in ["chat", "embedding", "image", "video", "model_sets"] {
        let declaration = format!("pub mod {family_module}");
        assert!(
            gemini_models_rs.contains(&declaration),
            "Gemini model catalog should expose the {family_module} model group"
        );
    }
}

#[test]
fn experimental_bridge_is_owned_by_bridge_crate_and_reexported_by_facade() {
    let lib_rs = read_source("src/lib.rs");
    let bridge_crate_lib = fs::read_to_string(crate_root().join("../siumai-bridge/src/lib.rs"))
        .expect("read siumai-bridge lib.rs");

    assert!(
        !lib_rs.contains("mod experimental_bridge;"),
        "siumai facade should not own the bridge implementation module"
    );
    assert!(
        !lib_rs.contains("pub mod experimental_bridge;"),
        "experimental_bridge should not become a top-level public facade module"
    );
    assert!(
        lib_rs.contains("pub use siumai_bridge::*;"),
        "siumai::experimental::bridge should re-export the dedicated bridge crate"
    );

    assert!(
        !crate_root().join("src/experimental_bridge.rs").exists(),
        "bridge implementation file should not live in the facade crate"
    );
    assert!(
        !crate_root().join("src/experimental_bridge").exists(),
        "bridge implementation directory should not live in the facade crate"
    );

    assert!(
        bridge_crate_lib.contains("This crate owns gateway/protocol conversion code")
            && bridge_crate_lib.contains("siumai-extras")
            && bridge_crate_lib.contains("siumai-core"),
        "siumai-bridge should document why the bridge lives outside the facade and core crates"
    );
}

#[test]
fn stable_unified_prelude_excludes_compatibility_construction_aliases() {
    let lib_rs = read_source("src/lib.rs");
    let unified_start = lib_rs
        .find("pub mod unified {")
        .expect("unified prelude module");
    let compat_start = lib_rs[unified_start..]
        .find("pub mod compat {")
        .expect("compat prelude module");
    let unified_source = &lib_rs[unified_start..unified_start + compat_start];

    for forbidden in [
        "pub use crate::Provider;",
        "pub use crate::provider::Siumai;",
        "pub use crate::compat::{Siumai, SiumaiBuilder};",
        "experimental_generate_image",
        "experimental_generate_speech",
        "experimental_transcribe",
        "experimental_generate_video",
    ] {
        assert!(
            !unified_source.contains(forbidden),
            "prelude::unified should not export compatibility-only surface `{forbidden}`"
        );
    }

    assert!(
        lib_rs.contains("pub mod compat {")
            && lib_rs.contains("pub use crate::Provider;")
            && lib_rs.contains("pub use crate::compat::{Siumai, SiumaiBuilder};"),
        "compatibility construction aliases should remain explicit under prelude::compat"
    );
}
