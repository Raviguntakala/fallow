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
//!   package directory; each extra dot climbs one parent.
//! - **`from X import Y`**: `Y` is preferred as the submodule `X.Y` when that
//!   file exists (Python loads the submodule), otherwise `Y` is treated as a
//!   name defined in / re-exported by `X` and the source `X` is resolved.
//! - **Absolute imports** (`import a.b`, `from a.b import x`): resolved against
//!   the importing file's *import root* — the nearest ancestor that is not
//!   itself a package (no `__init__.py` / `__init__.pyi`). This matches CPython's
//!   `sys.path`-from-package-root behaviour for classic and flat layouts.
//! - Module files and package markers resolve `.py` before `.pyi`, so stub-only
//!   modules and packages are reachable.
//! - An unresolved **relative** import is [`ResolveResult::Unresolvable`] (a
//!   relative import must be internal). An unresolved **absolute** import is a
//!   Python stdlib or pip module, modelled as [`ResolveResult::ExternalFile`]
//!   so it never feeds the npm package.json dependency analysis. Real Python
//!   dependency tracking (pyproject.toml) is a later phase.
//!
//! Known limitations (deferred): PEP 420 implicit namespace packages and
//! multiple `sys.path` roots beyond the import-root walk.

use std::path::{Path, PathBuf};

use fallow_types::discover::FileId;
use fallow_types::extract::{ImportInfo, ImportedName, ReExportInfo};

use super::types::ResolveResult;
use super::{ResolvedImport, ResolvedReExport};

/// Whether `path` is a Python source or stub file (resolution dispatch).
pub(super) fn is_python_path(path: &Path) -> bool {
    fallow_types::path_util::is_python_file(path)
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

/// Resolve every Python `__init__.py` re-export source against the project
/// file index (e.g. `from .api import X` -> the `api.py` module file).
pub(super) fn resolve_python_re_exports<F: Fn(&Path) -> Option<FileId>>(
    file_path: &Path,
    re_exports: &[ReExportInfo],
    lookup: F,
) -> Vec<ResolvedReExport> {
    re_exports
        .iter()
        .map(|re| ResolvedReExport {
            info: re.clone(),
            target: resolve_python_specifier(file_path, &re.source, &lookup),
        })
        .collect()
}

/// Resolve a bare module specifier (no imported-name submodule fallback), used
/// for re-export sources which always carry a named relative module.
fn resolve_python_specifier<F: Fn(&Path) -> Option<FileId>>(
    importing_file: &Path,
    source: &str,
    lookup: &F,
) -> ResolveResult {
    let dir = importing_file.parent().unwrap_or_else(|| Path::new(""));
    let leading_dots = source.chars().take_while(|c| *c == '.').count();
    if leading_dots > 0 {
        let module = &source[leading_dots..];
        climb_base(dir, leading_dots)
            .and_then(|base| {
                if module.is_empty() {
                    resolve_package_init(&base, lookup)
                } else {
                    resolve_module_file(&base, module, lookup)
                }
            })
            .map_or_else(
                || ResolveResult::Unresolvable(source.to_string()),
                ResolveResult::InternalModule,
            )
    } else {
        resolve_absolute(dir, source, lookup)
    }
}

/// Climb `level - 1` parent directories from `dir` (level 1 = current package).
fn climb_base(dir: &Path, level: usize) -> Option<PathBuf> {
    let mut base = dir.to_path_buf();
    for _ in 0..level.saturating_sub(1) {
        base = base.parent()?.to_path_buf();
    }
    Some(base)
}

/// Resolve a single Python import to a [`ResolveResult`].
///
/// For `from X import Y`, `Y` is commonly the submodule `X.Y` (Python loads the
/// submodule when the file exists), so that edge is preferred when it resolves
/// to a project file. Otherwise `Y` is a name defined in `X` (or re-exported by
/// `X`'s `__init__`) and the source `X` is resolved. Plain `import a.b` carries
/// its full path in the source and resolves directly. This subsumes the
/// `from . import NAME` case (source `"."`, submodule `".NAME"`).
fn resolve_python_import<F: Fn(&Path) -> Option<FileId>>(
    importing_file: &Path,
    import: &ImportInfo,
    lookup: &F,
) -> ResolveResult {
    let source = import.source.as_str();

    if let ImportedName::Named(name) = &import.imported_name {
        // `.`/`..` already end in a separator dot; a named module needs one.
        let submodule = if source.ends_with('.') {
            format!("{source}{name}")
        } else {
            format!("{source}.{name}")
        };
        if let ResolveResult::InternalModule(id) =
            resolve_python_specifier(importing_file, &submodule, lookup)
        {
            return ResolveResult::InternalModule(id);
        }
    }

    resolve_python_specifier(importing_file, source, lookup)
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
    // External: a Python stdlib or third-party (pip) module. Modelled as an
    // external file rather than `NpmPackage`, so it never feeds the package.json
    // dependency analysis (no false unlisted/unused-npm-dependency findings in
    // mixed JS+Python repos). Python dependency manifests (pyproject.toml) are a
    // later phase; until then external Python imports are silent.
    ResolveResult::ExternalFile(PathBuf::from(dotted))
}

/// Module file extensions, in resolution priority: a real implementation
/// (`.py`) is preferred over a type stub (`.pyi`).
const MODULE_EXTENSIONS: &[&str] = &["py", "pyi"];

/// Package marker files, same `.py`-before-`.pyi` priority.
const INIT_FILES: &[&str] = &["__init__.py", "__init__.pyi"];

/// Walk up from `start_dir` while each directory is a package (has an
/// `__init__.py` or `__init__.pyi`), returning the first ancestor that is not —
/// the import root.
fn import_root<F: Fn(&Path) -> Option<FileId>>(start_dir: &Path, lookup: &F) -> PathBuf {
    let mut dir = start_dir.to_path_buf();
    while resolve_package_init(&dir, lookup).is_some() {
        match dir.parent() {
            Some(parent) => dir = parent.to_path_buf(),
            None => break,
        }
    }
    dir
}

/// Resolve a dotted module path under `base` to `base/seg/.../leaf.{py,pyi}` or
/// `base/seg/.../leaf/__init__.{py,pyi}`.
fn resolve_module_file<F: Fn(&Path) -> Option<FileId>>(
    base: &Path,
    dotted: &str,
    lookup: &F,
) -> Option<FileId> {
    let module_dir = push_dotted(base, dotted);

    // Reuse one buffer: `set_extension` overwrites the previous extension in
    // place, so `.py` then `.pyi` is probed without re-allocating per attempt.
    let mut candidate = module_dir.clone();
    for ext in MODULE_EXTENSIONS {
        candidate.set_extension(ext);
        if let Some(id) = lookup(&candidate) {
            return Some(id);
        }
    }

    resolve_package_init(&module_dir, lookup)
}

/// Resolve the package marker (`__init__.py` / `__init__.pyi`) of directory
/// `base`.
fn resolve_package_init<F: Fn(&Path) -> Option<FileId>>(base: &Path, lookup: &F) -> Option<FileId> {
    INIT_FILES.iter().find_map(|init| lookup(&base.join(init)))
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
    fn from_package_import_submodule_resolves_to_submodule_file() {
        // `from pkg import api` where pkg/api.py exists loads the submodule, so
        // the edge must point at api.py, not pkg/__init__.py.
        let index = file_index(&[("app.py", 0), ("pkg/__init__.py", 1), ("pkg/api.py", 2)]);
        let result = resolve_python_import(
            Path::new("app.py"),
            &import("pkg", ImportedName::Named("api".to_string())),
            &lookup_in(&index),
        );
        assert!(matches!(result, ResolveResult::InternalModule(FileId(2))));
    }

    #[test]
    fn from_package_import_name_falls_back_to_package_init() {
        // `from pkg import helper` where helper is a name in __init__ (no
        // pkg/helper.py) resolves to the package __init__.
        let index = file_index(&[("app.py", 0), ("pkg/__init__.py", 1)]);
        let result = resolve_python_import(
            Path::new("app.py"),
            &import("pkg", ImportedName::Named("helper".to_string())),
            &lookup_in(&index),
        );
        assert!(matches!(result, ResolveResult::InternalModule(FileId(1))));
    }

    #[test]
    fn from_relative_package_import_submodule_resolves() {
        // `from .pkg import sub` where app/pkg/sub.py exists.
        let index = file_index(&[
            ("app/main.py", 0),
            ("app/pkg/__init__.py", 1),
            ("app/pkg/sub.py", 2),
        ]);
        let result = resolve_python_import(
            Path::new("app/main.py"),
            &import(".pkg", ImportedName::Named("sub".to_string())),
            &lookup_in(&index),
        );
        assert!(matches!(result, ResolveResult::InternalModule(FileId(2))));
    }

    #[test]
    fn unresolved_absolute_import_is_external_not_npm() {
        // Python stdlib / pip packages must NOT be classified as npm packages,
        // or they would surface as unlisted-npm-dependency findings in mixed
        // repos that also contain a package.json.
        let index = file_index(&[("app.py", 0)]);
        let result = resolve_python_import(
            Path::new("app.py"),
            &import("requests", ImportedName::Namespace),
            &lookup_in(&index),
        );
        assert!(
            matches!(result, ResolveResult::ExternalFile(_)),
            "expected ExternalFile, got {result:?}"
        );
        assert!(result.package_usage_name().is_none());
    }

    #[test]
    fn unresolved_absolute_dotted_import_is_external() {
        let index = file_index(&[("app.py", 0)]);
        let result = resolve_python_import(
            Path::new("app.py"),
            &import("os.path", ImportedName::Namespace),
            &lookup_in(&index),
        );
        assert!(matches!(result, ResolveResult::ExternalFile(_)));
    }

    #[test]
    fn stub_only_module_resolves() {
        // `.pyi` stub with no `.py` implementation is discovered, so imports
        // to it must resolve rather than fall through to an external package.
        let index = file_index(&[("app.py", 0), ("api.pyi", 1)]);
        let result = resolve_python_import(
            Path::new("app.py"),
            &import("api", ImportedName::Namespace),
            &lookup_in(&index),
        );
        assert!(matches!(result, ResolveResult::InternalModule(FileId(1))));
    }

    #[test]
    fn stub_only_package_init_resolves() {
        let index = file_index(&[("app.py", 0), ("pkg/__init__.pyi", 1)]);
        let result = resolve_python_import(
            Path::new("app.py"),
            &import("pkg", ImportedName::Namespace),
            &lookup_in(&index),
        );
        assert!(matches!(result, ResolveResult::InternalModule(FileId(1))));
    }

    #[test]
    fn import_root_walks_through_stub_only_package() {
        // `src/app` is a package declared only by `__init__.pyi`; `src` is the
        // import root, so `app.config` must resolve against it.
        let index = file_index(&[
            ("src/app/__init__.pyi", 0),
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
    fn relative_stub_module_resolves() {
        let index = file_index(&[("pkg/sub/a.py", 0), ("pkg/sub/mod.pyi", 1)]);
        let result = resolve_python_import(
            Path::new("pkg/sub/a.py"),
            &import(".mod", ImportedName::Named("x".to_string())),
            &lookup_in(&index),
        );
        assert!(matches!(result, ResolveResult::InternalModule(FileId(1))));
    }

    #[test]
    fn implementation_preferred_over_stub() {
        // When both `.py` and `.pyi` exist, the implementation wins.
        let index = file_index(&[("app.py", 0), ("api.py", 1), ("api.pyi", 2)]);
        let result = resolve_python_import(
            Path::new("app.py"),
            &import("api", ImportedName::Namespace),
            &lookup_in(&index),
        );
        assert!(matches!(result, ResolveResult::InternalModule(FileId(1))));
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

    fn re_export(source: &str, name: &str) -> ReExportInfo {
        ReExportInfo {
            source: source.to_string(),
            imported_name: name.to_string(),
            exported_name: name.to_string(),
            is_type_only: false,
            span: Span::default(),
        }
    }

    #[test]
    fn re_export_relative_module_resolves_to_file() {
        let index = file_index(&[("pkg/__init__.py", 0), ("pkg/api.py", 1)]);
        let resolved = resolve_python_re_exports(
            Path::new("pkg/__init__.py"),
            &[re_export(".api", "public_fn")],
            lookup_in(&index),
        );
        assert_eq!(resolved.len(), 1);
        assert!(matches!(
            resolved[0].target,
            ResolveResult::InternalModule(FileId(1))
        ));
    }

    #[test]
    fn re_export_parent_relative_resolves() {
        let index = file_index(&[("a/sub/__init__.py", 0), ("a/shared.py", 1)]);
        let resolved = resolve_python_re_exports(
            Path::new("a/sub/__init__.py"),
            &[re_export("..shared", "Thing")],
            lookup_in(&index),
        );
        assert!(matches!(
            resolved[0].target,
            ResolveResult::InternalModule(FileId(1))
        ));
    }

    #[test]
    fn re_export_unresolved_relative_is_unresolvable() {
        let index = file_index(&[("pkg/__init__.py", 0)]);
        let resolved = resolve_python_re_exports(
            Path::new("pkg/__init__.py"),
            &[re_export(".missing", "X")],
            lookup_in(&index),
        );
        assert!(matches!(resolved[0].target, ResolveResult::Unresolvable(_)));
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
