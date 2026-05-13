use std::fs;
use std::path::{Path, PathBuf};

#[test]
fn openai_compatible_protocol_is_not_imported_from_core() {
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root");
    let forbidden = ["siumai_core::standards::openai", "::compat"].concat();

    let mut offenders = Vec::new();
    for root in [
        "siumai/src",
        "siumai/tests",
        "siumai-registry/src",
        "siumai-provider-openai-compatible/src",
        "siumai-provider-openai/src",
        "siumai-provider-groq/src",
        "siumai-provider-deepseek/src",
        "siumai-protocol-openai/src",
        "siumai-extras/src",
        "siumai-extras/tests",
    ] {
        collect_forbidden_imports(&workspace.join(root), &forbidden, &mut offenders);
    }

    assert!(
        offenders.is_empty(),
        "OpenAI-compatible protocol imports must go through siumai-protocol-openai:\n{}",
        offenders.join("\n")
    );
}

#[test]
fn core_no_longer_owns_openai_compatible_protocol_files() {
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root");
    let core_compat_dir = workspace.join("siumai-core/src/standards/openai/compat");
    let mut remaining_files = Vec::new();
    collect_rs_files(&core_compat_dir, &mut remaining_files);

    assert!(
        remaining_files.is_empty(),
        "siumai-core must not own OpenAI-compatible protocol files:\n{}",
        remaining_files
            .iter()
            .map(|path| path.display().to_string())
            .collect::<Vec<_>>()
            .join("\n")
    );
}

fn collect_forbidden_imports(root: &Path, forbidden: &str, offenders: &mut Vec<String>) {
    let mut files = Vec::new();
    collect_rs_files(root, &mut files);

    for file in files {
        let Ok(content) = fs::read_to_string(&file) else {
            continue;
        };
        for (index, line) in content.lines().enumerate() {
            if line.contains(forbidden) {
                offenders.push(format!("{}:{}", file.display(), index + 1));
            }
        }
    }
}

fn collect_rs_files(root: &Path, files: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(root) else {
        return;
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_rs_files(&path, files);
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            files.push(path);
        }
    }
}
