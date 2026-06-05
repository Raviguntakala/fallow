//! Python import resolution.
//!
//! `oxc_resolver` is JavaScript/TypeScript-only, so Python `.py` imports take a
//! separate, filesystem-free path that maps dotted and relative module
//! specifiers onto project files. Resolution is driven entirely by a `lookup`
//! closure (`&Path -> Option<FileId>`) over the discovered project file index,
//! so it is deterministic and unit-testable without touching disk.
//!
//! ## Semantics
//! - **Relative imports** (`from . import x`, `from ..pkg import y`): the leading
//!   dot count is the package level. Level 1 is the importing file's own
//!   package directory; each extra dot climbs one parent. A bare `from . import
//!   name` resolves `name` as a submodule first, then falls back to the
//!   package's `__init__.py`.
//! - **Absolute imports** (`import a.b`, `from a.b import x`): resolved against
//!   the importing file's *import root* — the nearest ancestor that is not
//!   itself a package (no `__init__.py`). This matches CPython's `sys.path`-from
//!   -package-root behaviour for classic (``__init__.py``) and flat layouts.
//! - An unresolved **relative** import is [`ResolveResult::Unresolvable`] (a
//!   relative import must be internal). An unresolved **absolute** import is
//!   treated as an external [`ResolveResult::NpmPackage`] keyed on its first
//!   dotted segment (stdlib vs. third-party classification is a later phase).
//!
//! Known limitations (deferred): PEP 420 implicit namespace packages, multiple
//! `sys.path` roots beyond the import-root walk, and `from pkg import submodule`
//! where `submodule` is a sibling module rather than a name in `pkg/__init__`.

use std::path::{Path, PathBuf};

use fallow_types::discover::FileId;
use fallow_types::extract::{ImportInfo, ImportedName};

use super::ResolvedImport;
use super::types::ResolveResult;

/// Whether `path` is a Python source or stub file (resolution dispatch).
pub(super) fn is_python_path(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|ext| ext == "py" || ext == "pyi")
}

/// Resolve every Python import for one module against the project file index.
pub(super) fn resolve_python_imports<F: Fn(&Path) -> Option<FileId>>(
    file_path: &Path,
    imports: &[ImportInfo],
    lookup: F,
) -> Vec<ResolvedImport> {
    imports
        .iter()
        .map(|imp| ResolvedImport {
            info: imp.clone(),
            target: resolve_python_import(file_path, imp, &lookup),
        })
        .collect()
}

/// Resolve a single Python import to a [`ResolveResult`].
fn resolve_python_import<F: Fn(&Path) -> Option<FileId>>(
    importing_file: &Path,
    import: &ImportInfo,
    lookup: &F,
) -> ResolveResult {
    let dir = importing_file.parent().unwrap_or_else(|| Path::new(""));
    let source = import.source.as_str();
    let leading_dots = source.chars().take_while(|c| *c == '.').count();

    if leading_dots > 0 {
        let module = &source[leading_dots..];
        resolve_relative(dir, leading_dots, module, import, lookup)
            .unwrap_or_else(|| ResolveResult::Unresolvable(source.to_string()))
    } else {
        resolve_absolute(dir, source, lookup)
    }
}

/// Resolve a relative import (`from . import x`, `from ..pkg import y`).
fn resolve_relative<F: Fn(&Path) -> Option<FileId>>(
    importing_dir: &Path,
    level: usize,
    module: &str,
    import: &ImportInfo,
    lookup: &F,
) -> Option<ResolveResult> {
    // Level 1 is the current package directory; each extra dot climbs a parent.
    let mut base = importing_dir.to_path_buf();
    for _ in 0..level.saturating_sub(1) {
        base = base.parent()?.to_path_buf();
    }

    if module.is_empty() {
        // `from . import NAME` — NAME is most often a submodule file; fall back
        // to a name re-exported from the package's `__init__.py`.
        if let ImportedName::Named(name) = &import.imported_name
            && let Some(id) = resolve_module_file(&base, name, lookup)
        {
            return Some(ResolveResult::InternalModule(id));
        }
        return resolve_package_init(&base, lookup).map(ResolveResult::InternalModule);
    }

    resolve_module_file(&base, module, lookup).map(ResolveResult::InternalModule)
}

/// Resolve an absolute import against the importing file's import root.
fn resolve_absolute<F: Fn(&Path) -> Option<FileId>>(
    importing_dir: &Path,
    dotted: &str,
    lookup: &F,
) -> ResolveResult {
    let root = import_root(importing_dir, lookup);
    if let Some(id) = resolve_module_file(&root, dotted, lookup) {
        return ResolveResult::InternalModule(id);
    }
    // Flat-layout fallback: a sibling module imported by bare name from inside a
    // package directory (common in scripts and test helpers).
    if root != importing_dir
        && let Some(id) = resolve_module_file(importing_dir, dotted, lookup)
    {
        return ResolveResult::InternalModule(id);
    }
    let package = dotted.split('.').next().unwrap_or(dotted);
    ResolveResult::NpmPackage(package.to_string())
}

/// Walk up from `start_dir` while each directory is a package (`__init__.py`),
/// returning the first ancestor that is not — the import root.
fn import_root<F: Fn(&Path) -> Option<FileId>>(start_dir: &Path, lookup: &F) -> PathBuf {
    let mut dir = start_dir.to_path_buf();
    while lookup(&dir.join("__init__.py")).is_some() {
        match dir.parent() {
            Some(parent) => dir = parent.to_path_buf(),
            None => break,
        }
    }
    dir
}

/// Resolve a dotted module path under `base` to `base/seg/.../leaf.py` or
/// `base/seg/.../leaf/__init__.py`.
fn resolve_module_file<F: Fn(&Path) -> Option<FileId>>(
    base: &Path,
    dotted: &str,
    lookup: &F,
) -> Option<FileId> {
    let module_dir = push_dotted(base, dotted);

    let mut as_module = module_dir.clone();
    as_module.set_extension("py");
    if let Some(id) = lookup(&as_module) {
        return Some(id);
    }

    lookup(&module_dir.join("__init__.py"))
}

/// Resolve the `__init__.py` of the package directory `base`.
fn resolve_package_init<F: Fn(&Path) -> Option<FileId>>(base: &Path, lookup: &F) -> Option<FileId> {
    lookup(&base.join("__init__.py"))
}

/// Push each dot-separated segment of `dotted` onto `base` as a path component.
fn push_dotted(base: &Path, dotted: &str) -> PathBuf {
    let mut path = base.to_path_buf();
    for segment in dotted.split('.') {
        path.push(segment);
    }
    path
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxc_span::Span;
    use rustc_hash::FxHashMap;

    /// Build a `&Path -> Option<FileId>` lookup from `(path, id)` pairs.
    fn file_index(entries: &[(&str, u32)]) -> FxHashMap<PathBuf, FileId> {
        entries
            .iter()
            .map(|(p, id)| (PathBuf::from(p), FileId(*id)))
            .collect()
    }

    fn lookup_in(index: &FxHashMap<PathBuf, FileId>) -> impl Fn(&Path) -> Option<FileId> + '_ {
        move |p: &Path| index.get(p).copied()
    }

    fn import(source: &str, imported: ImportedName) -> ImportInfo {
        ImportInfo {
            source: source.to_string(),
            imported_name: imported,
            local_name: String::new(),
            is_type_only: false,
            from_style: false,
            span: Span::default(),
            source_span: Span::default(),
        }
    }

    #[test]
    fn flat_sibling_absolute_import_resolves() {
        let index = file_index(&[("basic/service.py", 0), ("basic/models.py", 1)]);
        let result = resolve_python_import(
            Path::new("basic/service.py"),
            &import("models", ImportedName::Named("User".to_string())),
            &lookup_in(&index),
        );
        assert!(matches!(result, ResolveResult::InternalModule(FileId(1))));
    }

    #[test]
    fn dotted_absolute_import_through_import_root() {
        // src/app is a package; src is the import root.
        let index = file_index(&[
            ("src/app/__init__.py", 0),
            ("src/app/main.py", 1),
            ("src/app/config.py", 2),
        ]);
        let result = resolve_python_import(
            Path::new("src/app/main.py"),
            &import("app.config", ImportedName::Named("SETTINGS".to_string())),
            &lookup_in(&index),
        );
        assert!(matches!(result, ResolveResult::InternalModule(FileId(2))));
    }

    #[test]
    fn relative_dotted_module_resolves() {
        let index = file_index(&[("pkg/sub/a.py", 0), ("pkg/sub/mod.py", 1)]);
        let result = resolve_python_import(
            Path::new("pkg/sub/a.py"),
            &import(".mod", ImportedName::Named("x".to_string())),
            &lookup_in(&index),
        );
        assert!(matches!(result, ResolveResult::InternalModule(FileId(1))));
    }

    #[test]
    fn from_dot_import_resolves_submodule_first() {
        let index = file_index(&[("pkg/a.py", 0), ("pkg/foo.py", 1), ("pkg/__init__.py", 2)]);
        let result = resolve_python_import(
            Path::new("pkg/a.py"),
            &import(".", ImportedName::Named("foo".to_string())),
            &lookup_in(&index),
        );
        assert!(matches!(result, ResolveResult::InternalModule(FileId(1))));
    }

    #[test]
    fn from_dot_import_falls_back_to_package_init() {
        // `foo` is not a submodule file; it is a name re-exported by __init__.py.
        let index = file_index(&[("pkg/a.py", 0), ("pkg/__init__.py", 2)]);
        let result = resolve_python_import(
            Path::new("pkg/a.py"),
            &import(".", ImportedName::Named("foo".to_string())),
            &lookup_in(&index),
        );
        assert!(matches!(result, ResolveResult::InternalModule(FileId(2))));
    }

    #[test]
    fn parent_relative_import_climbs_one_level() {
        // `..pkg` from a/b/c.py climbs to a/, then resolves a/pkg.py.
        let index = file_index(&[("a/b/c.py", 0), ("a/pkg.py", 1)]);
        let result = resolve_python_import(
            Path::new("a/b/c.py"),
            &import("..pkg", ImportedName::Named("bar".to_string())),
            &lookup_in(&index),
        );
        assert!(matches!(result, ResolveResult::InternalModule(FileId(1))));
    }

    #[test]
    fn package_directory_resolves_to_init() {
        let index = file_index(&[("app.py", 0), ("pkg/__init__.py", 1)]);
        let result = resolve_python_import(
            Path::new("app.py"),
            &import("pkg", ImportedName::Namespace),
            &lookup_in(&index),
        );
        assert!(matches!(result, ResolveResult::InternalModule(FileId(1))));
    }

    #[test]
    fn unresolved_absolute_import_is_external_package() {
        let index = file_index(&[("app.py", 0)]);
        let result = resolve_python_import(
            Path::new("app.py"),
            &import("requests", ImportedName::Namespace),
            &lookup_in(&index),
        );
        match result {
            ResolveResult::NpmPackage(name) => assert_eq!(name, "requests"),
            other => panic!("expected NpmPackage, got {other:?}"),
        }
    }

    #[test]
    fn unresolved_absolute_dotted_import_uses_first_segment() {
        let index = file_index(&[("app.py", 0)]);
        let result = resolve_python_import(
            Path::new("app.py"),
            &import("os.path", ImportedName::Namespace),
            &lookup_in(&index),
        );
        match result {
            ResolveResult::NpmPackage(name) => assert_eq!(name, "os"),
            other => panic!("expected NpmPackage, got {other:?}"),
        }
    }

    #[test]
    fn unresolved_relative_import_is_unresolvable() {
        let index = file_index(&[("pkg/a.py", 0)]);
        let result = resolve_python_import(
            Path::new("pkg/a.py"),
            &import(".missing", ImportedName::Named("x".to_string())),
            &lookup_in(&index),
        );
        assert!(matches!(result, ResolveResult::Unresolvable(_)));
    }

    #[test]
    fn mutual_imports_resolve_both_directions() {
        // Mirrors tests/fixtures/python/basic: service <-> models cycle.
        let index = file_index(&[("basic/service.py", 0), ("basic/models.py", 1)]);
        let service_to_models = resolve_python_import(
            Path::new("basic/service.py"),
            &import("models", ImportedName::Named("User".to_string())),
            &lookup_in(&index),
        );
        let models_to_service = resolve_python_import(
            Path::new("basic/models.py"),
            &import("service", ImportedName::Named("build_user".to_string())),
            &lookup_in(&index),
        );
        assert!(matches!(
            service_to_models,
            ResolveResult::InternalModule(FileId(1))
        ));
        assert!(matches!(
            models_to_service,
            ResolveResult::InternalModule(FileId(0))
        ));
    }
}
