//! End-to-end Python support: discovery -> Ruff-parser extraction -> Python
//! import resolution -> graph -> cycle + dead-code detection.

use super::common::{create_config, fixture_path};

fn file_names(paths: &[std::path::PathBuf]) -> Vec<String> {
    paths
        .iter()
        .filter_map(|p| p.file_name().map(|n| n.to_string_lossy().to_string()))
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
