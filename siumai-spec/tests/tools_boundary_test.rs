use std::fs;
use std::path::{Path, PathBuf};

fn crate_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
}

fn collect_rust_files(path: &Path, files: &mut Vec<PathBuf>) {
    if path.is_file() {
        if path.extension().and_then(|ext| ext.to_str()) == Some("rs") {
            files.push(path.to_path_buf());
        }
        return;
    }

    for entry in fs::read_dir(path).expect("read source directory") {
        let entry = entry.expect("read source directory entry");
        collect_rust_files(&entry.path(), files);
    }
}

#[test]
fn spec_does_not_own_provider_defined_tool_catalogs() {
    let lib_rs =
        fs::read_to_string(crate_root().join("src/lib.rs")).expect("read siumai-spec/src/lib.rs");
    let tools_rs_path = crate_root().join("src/tools.rs");

    assert!(
        !tools_rs_path.exists(),
        "siumai-spec must not own the provider-defined tool catalog; keep concrete catalogs in protocol/provider-owned hosted_tools modules"
    );

    assert!(
        !lib_rs.contains("pub mod tools;"),
        "siumai-spec should only expose passive tool data shapes, not a provider catalog module"
    );
    assert!(lib_rs.contains("pub mod types;"));
}

#[test]
fn provider_defined_tool_data_surface_remains_passive() {
    let root = crate_root();
    let mut files = Vec::new();
    collect_rust_files(&root.join("src/types/tools"), &mut files);
    files.sort();

    for file in files {
        let source = fs::read_to_string(&file)
            .unwrap_or_else(|error| panic!("failed to read {}: {error}", file.display()));

        for forbidden in [
            "crate::tools::",
            "siumai_core::",
            "siumai_provider_",
            "siumai_protocol_",
            "tokio",
            "reqwest",
            "hyper::",
            "axum::",
            "async_trait",
            "spawn_blocking",
            "pub async fn",
            "async fn",
            ".await",
            "std::net",
            "std::process",
            "std::thread",
            "execute_tool",
            "ExecutableTool",
            "ExecutableTools",
            "ToolExecuteFunction",
            "ToolExecutionOptions",
            "ToolExecutionResult",
            "ToolExecutionStream",
            "ToolModelOutputContext",
            "ToolSet",
        ] {
            assert!(
                !source.contains(forbidden),
                "{} must keep spec tool surfaces passive and must not contain runtime/provider execution fragment `{forbidden}`",
                file.display()
            );
        }
    }
}
