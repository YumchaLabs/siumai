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
fn provider_defined_tool_factories_remain_passive_data_constructors() {
    let tools_rs = fs::read_to_string(crate_root().join("src/tools.rs"))
        .expect("read siumai-spec/src/tools.rs");
    let lib_rs =
        fs::read_to_string(crate_root().join("src/lib.rs")).expect("read siumai-spec/src/lib.rs");

    assert!(
        lib_rs.contains("pub mod tools;"),
        "siumai-spec should keep the passive provider-defined tool factory module public"
    );

    for forbidden in [
        "siumai_core::",
        "siumai_provider_",
        "siumai_protocol_",
        "tokio",
        "reqwest",
        "async_trait",
        "spawn_blocking",
        "execute_tool",
        "ToolExecutionOptions",
        "ToolExecutionResult",
        "ToolSet",
    ] {
        assert!(
            !tools_rs.contains(forbidden),
            "siumai-spec::tools must stay passive and must not contain runtime/provider execution fragment `{forbidden}`"
        );
    }

    for passive_constructor in [
        "provider_defined_tool",
        "pub mod openai",
        "pub mod anthropic",
        "pub mod google",
        "pub mod groq",
        "pub mod xai",
    ] {
        assert!(
            tools_rs.contains(passive_constructor),
            "siumai-spec::tools should keep passive provider-defined constructor `{passive_constructor}`"
        );
    }
}

#[test]
fn provider_defined_tool_data_surface_remains_passive() {
    let root = crate_root();
    let mut files = vec![root.join("src/tools.rs")];
    collect_rust_files(&root.join("src/types/tools"), &mut files);
    files.sort();

    for file in files {
        let source = fs::read_to_string(&file)
            .unwrap_or_else(|error| panic!("failed to read {}: {error}", file.display()));

        for forbidden in [
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
