//! End-to-end Python support: discovery -> Ruff-parser extraction -> Python
//! import resolution -> graph -> cycle + dead-code detection.

use super::common::{create_config, fixture_path};

fn file_names(paths: &[std::path::PathBuf]) -> Vec<String> {
    paths
        .iter()
        .filter_map(|p| p.file_name().map(|n| n.to_string_lossy().to_string()))
        .collect()
}

fn unused_file_names(results: &fallow_core::results::AnalysisResults) -> Vec<String> {
    results
        .unused_files
        .iter()
        .filter_map(|f| {
            f.file
                .path
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
        })
        .collect()
}

#[test]
fn python_mutual_imports_detected_as_cycle() {
    let root = fixture_path("python/basic");
    let config = create_config(root);
    let results = fallow_core::analyze(&config).expect("analysis should succeed");

    let has_cycle = results.circular_dependencies.iter().any(|finding| {
        let names = file_names(&finding.cycle.files);
        names.contains(&"models.py".to_string()) && names.contains(&"service.py".to_string())
    });
    assert!(
        has_cycle,
        "expected models.py <-> service.py cycle, got: {:?}",
        results.circular_dependencies
    );
}

#[test]
fn python_sibling_imports_resolve_without_unresolved_findings() {
    let root = fixture_path("python/basic");
    let config = create_config(root);
    let results = fallow_core::analyze(&config).expect("analysis should succeed");

    assert!(
        results.unresolved_imports.is_empty(),
        "sibling Python imports should resolve, got: {:?}",
        results.unresolved_imports
    );
}

#[test]
fn python_exports_flow_into_dead_code_detection() {
    let root = fixture_path("python/basic");
    let config = create_config(root);
    let results = fallow_core::analyze(&config).expect("analysis should succeed");

    let unused: Vec<&str> = results
        .unused_exports
        .iter()
        .map(|e| e.export.export_name.as_str())
        .collect();
    assert!(
        unused.contains(&"unused_helper"),
        "unused_helper should be flagged as an unused export, got: {unused:?}"
    );
}

#[test]
fn python_main_module_is_entry_point() {
    let root = fixture_path("python/entrypoints");
    let config = create_config(root);
    let results = fallow_core::analyze(&config).expect("analysis should succeed");

    let unused = unused_file_names(&results);
    assert!(
        !unused.contains(&"runner.py".to_string()),
        "runner.py is reachable from __main__.py and must not be unused, got: {unused:?}"
    );
    assert!(
        unused.contains(&"orphan_script.py".to_string()),
        "orphan_script.py is unreachable and should be unused, got: {unused:?}"
    );
}

#[test]
fn python_package_init_is_entry_point() {
    let root = fixture_path("python/package_lib");
    let config = create_config(root);
    let results = fallow_core::analyze(&config).expect("analysis should succeed");

    let unused = unused_file_names(&results);
    assert!(
        !unused.contains(&"api.py".to_string()),
        "api.py is reachable from __init__.py and must not be unused, got: {unused:?}"
    );
    assert!(
        unused.contains(&"orphan.py".to_string()),
        "mypkg/orphan.py is unreachable and should be unused, got: {unused:?}"
    );

    // `public_fn` is re-exported by the package __init__ barrel (an entry),
    // so it must be credited rather than reported as an unused export.
    let unused_exports: Vec<&str> = results
        .unused_exports
        .iter()
        .map(|e| e.export.export_name.as_str())
        .collect();
    assert!(
        !unused_exports.contains(&"public_fn"),
        "public_fn is re-exported by the __init__ barrel and should be credited, got: {unused_exports:?}"
    );
}

#[test]
fn python_venv_directory_is_excluded() {
    let root = fixture_path("python/venv_excluded");
    let config = create_config(root);
    let results = fallow_core::analyze(&config).expect("analysis should succeed");

    let touched: Vec<String> = results
        .unused_files
        .iter()
        .map(|f| f.file.path.to_string_lossy().replace('\\', "/"))
        .collect();
    assert!(
        !touched.iter().any(|p| p.contains("installed_pkg")),
        "files under venv/ must be excluded from discovery, got: {touched:?}"
    );
}
