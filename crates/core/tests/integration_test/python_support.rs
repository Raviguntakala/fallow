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
fn python_type_checking_imports_do_not_create_false_cycle() {
    // alpha.py imports beta only under `if TYPE_CHECKING:` (type-only), while
    // beta.py imports alpha at runtime. The type-only edge must be excluded
    // from cycle detection, so no alpha<->beta cycle is reported.
    let root = fixture_path("python/type_checking_cycle");
    let config = create_config(root);
    let results = fallow_core::analyze(&config).expect("analysis should succeed");

    let has_cycle = results.circular_dependencies.iter().any(|finding| {
        let names = file_names(&finding.cycle.files);
        names.contains(&"alpha.py".to_string()) && names.contains(&"beta.py".to_string())
    });
    assert!(
        !has_cycle,
        "TYPE_CHECKING-only import must not create a cycle, got: {:?}",
        results.circular_dependencies
    );
    assert!(
        results.unresolved_imports.is_empty(),
        "sibling imports should resolve, got: {:?}",
        results.unresolved_imports
    );
}

#[test]
fn python_stub_only_module_resolves_and_is_reachable() {
    // shapes.pyi has no .py implementation; importing it from the __main__
    // entry must resolve to the stub (not report unresolved / unused).
    let root = fixture_path("python/stub_package");
    let config = create_config(root);
    let results = fallow_core::analyze(&config).expect("analysis should succeed");

    assert!(
        results.unresolved_imports.is_empty(),
        "stub import should resolve, got: {:?}",
        results.unresolved_imports
    );
    let unused = unused_file_names(&results);
    assert!(
        !unused.contains(&"shapes.pyi".to_string()),
        "shapes.pyi is imported from the entry and must not be unused, got: {unused:?}"
    );
}

#[test]
fn python_stub_only_package_init_is_entry_point() {
    // A package whose only marker is __init__.pyi (no __init__.py) must still
    // get a reachability root, so its re-exported stub surface is credited.
    let root = fixture_path("python/stub_pkg_init");
    let config = create_config(root);
    let results = fallow_core::analyze(&config).expect("analysis should succeed");

    let unused = unused_file_names(&results);
    assert!(
        !unused.contains(&"api.pyi".to_string()),
        "api.pyi is re-exported by the __init__.pyi entry and must not be unused, got: {unused:?}"
    );
    assert!(
        unused.contains(&"orphan.pyi".to_string()),
        "orphan.pyi is unreachable and should be unused, got: {unused:?}"
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
