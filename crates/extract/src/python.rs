//! Python source extraction into the language-agnostic [`ModuleInfo`].
//!
//! Uses Ruff's Python parser (published on crates.io via the RustPython fork as
//! `rustpython-ruff_python_parser`, aliased back to `ruff_python_parser`) to
//! produce a typed AST, then maps imports/exports onto [`ModuleInfo`] so the
//! existing graph, dead-code, and cycle-detection pipeline works for `.py`
//! files with no downstream changes.
//!
//! Syntactic only (matches ADR-001 "no type resolution"): unresolvable dynamic
//! imports (`importlib`, `__import__`) are simply not captured, never errors.
//!
//! ## Import modelling
//! - `import os` / `import os.path` -> [`ImportedName::Namespace`]; `source` is
//!   the full dotted path, `local_name` is the bound top-level name.
//! - `from os import path` -> [`ImportedName::Named`].
//! - Relative imports keep their leading dots in `source` (`.`, `..pkg`); the
//!   graph resolver interprets dot-depth as package levels.
//! - `from x import *` -> [`ImportedName::Namespace`] with `local_name` `"*"`.
//!
//! ## Export modelling
//! Python has no `export` keyword, so exports are heuristic (conservative
//! default, see `.claude/rules/workflow.md`): if `__all__` is present, exactly
//! those names; otherwise all module-level `def`/`class`/simple assignments
//! whose name does not start with `_`.

use std::path::Path;

use oxc_span::Span;
use ruff_python_ast::{ExceptHandler, Expr, Stmt, StmtImport, StmtImportFrom};
use ruff_text_size::TextRange;
use rustc_hash::FxHashMap;

use crate::{
    ExportInfo, ExportName, ImportInfo, ImportedName, ModuleInfo, ReExportInfo, VisibilityTag,
};
use fallow_types::discover::FileId;

/// Whether `path` is a Python source or stub file.
pub fn is_python_file(path: &Path) -> bool {
    fallow_types::path_util::is_python_file(path)
}

/// Convert a Ruff `TextRange` (u32-backed byte offsets) into an oxc [`Span`].
fn span_of(range: TextRange) -> Span {
    Span::new(range.start().to_u32(), range.end().to_u32())
}

/// Build an [`ExportInfo`] for a public module-level name.
fn export_info(name: String, span: Span) -> ExportInfo {
    ExportInfo {
        name: ExportName::Named(name),
        local_name: None,
        is_type_only: false,
        is_side_effect_used: false,
        visibility: VisibilityTag::None,
        span,
        members: Vec::new(),
        super_class: None,
    }
}

/// Build an [`ImportInfo`] with the Python-irrelevant fields defaulted.
fn import_info(
    source: String,
    imported_name: ImportedName,
    local_name: String,
    type_only: bool,
    span: Span,
) -> ImportInfo {
    ImportInfo {
        source,
        imported_name,
        local_name,
        is_type_only: type_only,
        from_style: false,
        span,
        source_span: span,
    }
}

/// Map an `import a, b.c as d` statement onto namespace imports.
fn imports_from_import(stmt: &StmtImport, type_only: bool, out: &mut Vec<ImportInfo>) {
    for alias in &stmt.names {
        let dotted = alias.name.as_str().to_string();
        let local = match &alias.asname {
            Some(a) => a.as_str().to_string(),
            // `import a.b.c` binds the top-level name `a`.
            None => dotted
                .split('.')
                .next()
                .unwrap_or(dotted.as_str())
                .to_string(),
        };
        out.push(import_info(
            dotted,
            ImportedName::Namespace,
            local,
            type_only,
            span_of(alias.range),
        ));
    }
}

/// Map a `from <module> import <names>` statement, preserving relative dots.
fn imports_from_import_from(stmt: &StmtImportFrom, type_only: bool, out: &mut Vec<ImportInfo>) {
    let dots = ".".repeat(usize::try_from(stmt.level).unwrap_or(0));
    let module = stmt.module.as_ref().map_or("", |m| m.as_str());
    let source = format!("{dots}{module}");
    for alias in &stmt.names {
        let name = alias.name.as_str();
        let span = span_of(alias.range);
        if name == "*" {
            out.push(import_info(
                source.clone(),
                ImportedName::Namespace,
                "*".to_string(),
                type_only,
                span,
            ));
        } else {
            let local = alias
                .asname
                .as_ref()
                .map_or(name, |a| a.as_str())
                .to_string();
            out.push(import_info(
                source.clone(),
                ImportedName::Named(name.to_string()),
                local,
                type_only,
                span,
            ));
        }
    }
}

/// Collect the string elements of an `__all__ = [...]` / `(...)` value.
fn collect_string_list(value: &Expr) -> Vec<String> {
    let elts = match value {
        Expr::List(list) => &list.elts,
        Expr::Tuple(tuple) => &tuple.elts,
        _ => return Vec::new(),
    };
    elts.iter()
        .filter_map(|elt| match elt {
            Expr::StringLiteral(s) => Some(s.value.to_str().to_string()),
            _ => None,
        })
        .collect()
}

/// Extract module-level exports using the `__all__`-or-public-names heuristic.
fn extract_exports(body: &[Stmt]) -> Vec<ExportInfo> {
    let mut explicit_all: Option<Vec<String>> = None;
    // Candidate public definitions in source order, with their declaration span.
    let mut candidates: Vec<(String, Span)> = Vec::new();

    for stmt in body {
        match stmt {
            Stmt::FunctionDef(f) => {
                candidates.push((f.name.as_str().to_string(), span_of(f.range)));
            }
            Stmt::ClassDef(c) => candidates.push((c.name.as_str().to_string(), span_of(c.range))),
            Stmt::Assign(assign) => {
                for target in &assign.targets {
                    if let Expr::Name(name) = target {
                        let id = name.id.as_str();
                        if id == "__all__" {
                            explicit_all = Some(collect_string_list(&assign.value));
                        } else {
                            candidates.push((id.to_string(), span_of(assign.range)));
                        }
                    }
                }
            }
            // Annotated module variables: `FOO: int = 1` and the value-less
            // `FOO: int` (common in `.pyi` stubs). Ruff emits these as AnnAssign.
            Stmt::AnnAssign(ann) => {
                if let Expr::Name(name) = ann.target.as_ref() {
                    let id = name.id.as_str();
                    if id == "__all__" {
                        if let Some(value) = &ann.value {
                            explicit_all = Some(collect_string_list(value));
                        }
                    } else {
                        candidates.push((id.to_string(), span_of(ann.range)));
                    }
                }
            }
            _ => {}
        }
    }

    match explicit_all {
        Some(names) => {
            let span_by_name: FxHashMap<&str, Span> =
                candidates.iter().map(|(n, s)| (n.as_str(), *s)).collect();
            names
                .into_iter()
                .map(|name| {
                    let span = span_by_name.get(name.as_str()).copied().unwrap_or_default();
                    export_info(name, span)
                })
                .collect()
        }
        None => candidates
            .into_iter()
            .filter(|(name, _)| !name.starts_with('_'))
            .map(|(name, span)| export_info(name, span))
            .collect(),
    }
}

/// Whether `path` is a package `__init__.py` / `__init__.pyi`.
fn is_init_module(path: &Path) -> bool {
    path.file_name()
        .and_then(|n| n.to_str())
        .is_some_and(|name| name == "__init__.py" || name == "__init__.pyi")
}

/// Whether a `from ... import ...` is relative with a named module (`from .api
/// import X`) — the form an `__init__.py` re-exports as package API. The bare
/// `from . import submodule` form (no module) stays an ordinary import because
/// resolving it needs the imported name as the submodule.
fn is_relative_with_module(stmt: &StmtImportFrom) -> bool {
    stmt.level > 0 && stmt.module.is_some()
}

/// Whether an `if` test is a typing guard (`TYPE_CHECKING` / `t.TYPE_CHECKING`).
fn is_type_checking_test(test: &Expr) -> bool {
    match test {
        Expr::Name(name) => name.id.as_str() == "TYPE_CHECKING",
        Expr::Attribute(attr) => attr.attr.as_str() == "TYPE_CHECKING",
        _ => false,
    }
}

/// Extract imports from the module body.
///
/// Recurses into module-load-time compound blocks (`if`/`try`/`with`/`for`/
/// `while`) so conditional and `try/except ImportError` imports are captured.
/// Function and class bodies are deliberately NOT descended: those imports are
/// deferred/lazy and are commonly used to break import cycles, so capturing
/// them would reintroduce the very cycle the author avoided (a false positive).
///
/// Imports inside an `if TYPE_CHECKING:` block are marked type-only so they are
/// excluded from circular-dependency detection (the standard pattern for
/// breaking runtime cycles while keeping type annotations).
///
/// In an `__init__.py`, a top-level `from .module import name` is modelled as a
/// re-export (see [`extract_re_exports`]) and skipped here to avoid a duplicate
/// edge.
fn extract_imports(body: &[Stmt], is_init: bool) -> Vec<ImportInfo> {
    let mut imports = Vec::new();
    collect_imports(body, is_init, false, true, &mut imports);
    imports
}

/// Recursive worker for [`extract_imports`]. `type_only` carries an enclosing
/// `TYPE_CHECKING` context; `top_level` gates the `__init__.py` re-export skip
/// to module top level (nested re-exports are not modelled, so they stay
/// ordinary imports rather than being dropped).
fn collect_imports(
    stmts: &[Stmt],
    is_init: bool,
    type_only: bool,
    top_level: bool,
    out: &mut Vec<ImportInfo>,
) {
    for stmt in stmts {
        match stmt {
            Stmt::Import(import) => imports_from_import(import, type_only, out),
            Stmt::ImportFrom(import_from) => {
                if is_init && top_level && is_relative_with_module(import_from) {
                    continue;
                }
                imports_from_import_from(import_from, type_only, out);
            }
            Stmt::If(if_stmt) => {
                let body_type_only = type_only || is_type_checking_test(&if_stmt.test);
                collect_imports(&if_stmt.body, is_init, body_type_only, false, out);
                for clause in &if_stmt.elif_else_clauses {
                    // An `else:` of a TYPE_CHECKING block is runtime; only a
                    // clause whose own test is a typing guard is type-only.
                    let clause_type_only =
                        type_only || clause.test.as_ref().is_some_and(is_type_checking_test);
                    collect_imports(&clause.body, is_init, clause_type_only, false, out);
                }
            }
            Stmt::Try(try_stmt) => {
                collect_imports(&try_stmt.body, is_init, type_only, false, out);
                for handler in &try_stmt.handlers {
                    let ExceptHandler::ExceptHandler(h) = handler;
                    collect_imports(&h.body, is_init, type_only, false, out);
                }
                collect_imports(&try_stmt.orelse, is_init, type_only, false, out);
                collect_imports(&try_stmt.finalbody, is_init, type_only, false, out);
            }
            Stmt::With(with_stmt) => {
                collect_imports(&with_stmt.body, is_init, type_only, false, out);
            }
            Stmt::For(for_stmt) => {
                collect_imports(&for_stmt.body, is_init, type_only, false, out);
                collect_imports(&for_stmt.orelse, is_init, type_only, false, out);
            }
            Stmt::While(while_stmt) => {
                collect_imports(&while_stmt.body, is_init, type_only, false, out);
                collect_imports(&while_stmt.orelse, is_init, type_only, false, out);
            }
            _ => {}
        }
    }
}

/// Extract package re-exports from an `__init__.py`: each `from .module import
/// name` re-exposes `name` as part of the package's public API, so it is
/// modelled as a [`ReExportInfo`] feeding the re-export chain resolver.
fn extract_re_exports(body: &[Stmt], is_init: bool) -> Vec<ReExportInfo> {
    if !is_init {
        return Vec::new();
    }
    let mut re_exports = Vec::new();
    for stmt in body {
        let Stmt::ImportFrom(import_from) = stmt else {
            continue;
        };
        if !is_relative_with_module(import_from) {
            continue;
        }
        let dots = ".".repeat(usize::try_from(import_from.level).unwrap_or(0));
        let module = import_from.module.as_ref().map_or("", |m| m.as_str());
        let source = format!("{dots}{module}");
        let span = span_of(import_from.range);
        for alias in &import_from.names {
            let name = alias.name.as_str();
            let exported = alias.asname.as_ref().map_or(name, |a| a.as_str());
            re_exports.push(ReExportInfo {
                source: source.clone(),
                imported_name: name.to_string(),
                exported_name: exported.to_string(),
                is_type_only: false,
                span,
            });
        }
    }
    re_exports
}

/// Parse Python `source` into a [`ModuleInfo`].
///
/// On a fatal parse error this returns an empty-but-valid module (with line
/// offsets and suppressions populated) rather than panicking, matching the
/// best-effort posture of the other language front-ends.
pub fn parse_python_to_module(
    file_id: FileId,
    path: &Path,
    source: &str,
    content_hash: u64,
) -> ModuleInfo {
    let is_init = is_init_module(path);
    let parsed_suppressions = crate::suppress::parse_suppressions_from_source(source);
    let (imports, exports, re_exports) = match ruff_python_parser::parse_module(source) {
        Ok(parsed) => {
            let body = &parsed.syntax().body;
            (
                extract_imports(body, is_init),
                extract_exports(body),
                extract_re_exports(body, is_init),
            )
        }
        Err(_) => (Vec::new(), Vec::new(), Vec::new()),
    };

    ModuleInfo {
        file_id,
        exports,
        imports,
        re_exports,
        dynamic_imports: Vec::new(),
        dynamic_import_patterns: Vec::new(),
        require_calls: Vec::new(),
        member_accesses: Vec::new(),
        whole_object_uses: Vec::new(),
        has_cjs_exports: false,
        has_angular_component_template_url: false,
        content_hash,
        suppressions: parsed_suppressions.suppressions,
        unknown_suppression_kinds: parsed_suppressions.unknown_kinds,
        unused_import_bindings: Vec::new(),
        type_referenced_import_bindings: Vec::new(),
        value_referenced_import_bindings: Vec::new(),
        line_offsets: fallow_types::extract::compute_line_offsets(source),
        complexity: Vec::new(),
        flag_uses: Vec::new(),
        class_heritage: Vec::new(),
        injection_tokens: Vec::new(),
        local_type_declarations: Vec::new(),
        public_signature_type_references: Vec::new(),
        namespace_object_aliases: Vec::new(),
        iconify_prefixes: Vec::new(),
        auto_import_candidates: Vec::new(),
        directives: Vec::new(),
        security_sinks: Vec::new(),
        security_sinks_skipped: 0,
        tainted_bindings: Vec::new(),
        sanitized_sink_args: Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(src: &str) -> ModuleInfo {
        parse_python_to_module(FileId(0), Path::new("m.py"), src, 0)
    }

    fn parse_at(path: &str, src: &str) -> ModuleInfo {
        parse_python_to_module(FileId(0), Path::new(path), src, 0)
    }

    fn export_names(info: &ModuleInfo) -> Vec<String> {
        info.exports.iter().map(|e| e.name.to_string()).collect()
    }

    #[test]
    fn detects_python_extensions() {
        assert!(is_python_file(Path::new("a.py")));
        assert!(is_python_file(Path::new("pkg/stub.pyi")));
        assert!(!is_python_file(Path::new("a.ts")));
        assert!(!is_python_file(Path::new("a")));
    }

    #[test]
    fn plain_import_is_namespace() {
        let info = parse("import os\n");
        assert_eq!(info.imports.len(), 1);
        assert_eq!(info.imports[0].source, "os");
        assert_eq!(info.imports[0].local_name, "os");
        assert_eq!(info.imports[0].imported_name, ImportedName::Namespace);
    }

    #[test]
    fn dotted_import_binds_top_level_name() {
        let info = parse("import os.path\n");
        assert_eq!(info.imports[0].source, "os.path");
        assert_eq!(info.imports[0].local_name, "os");
    }

    #[test]
    fn dotted_import_as_binds_alias() {
        let info = parse("import os.path as p\n");
        assert_eq!(info.imports[0].source, "os.path");
        assert_eq!(info.imports[0].local_name, "p");
    }

    #[test]
    fn multiple_names_in_one_import() {
        let info = parse("import os, sys\n");
        let sources: Vec<&str> = info.imports.iter().map(|i| i.source.as_str()).collect();
        assert_eq!(sources, vec!["os", "sys"]);
    }

    #[test]
    fn from_import_is_named() {
        let info = parse("from os import path\n");
        assert_eq!(info.imports[0].source, "os");
        assert_eq!(
            info.imports[0].imported_name,
            ImportedName::Named("path".to_string())
        );
        assert_eq!(info.imports[0].local_name, "path");
    }

    #[test]
    fn from_import_as_binds_alias() {
        let info = parse("from os import path as p\n");
        assert_eq!(
            info.imports[0].imported_name,
            ImportedName::Named("path".to_string())
        );
        assert_eq!(info.imports[0].local_name, "p");
    }

    #[test]
    fn relative_import_current_package() {
        let info = parse("from . import foo\n");
        assert_eq!(info.imports[0].source, ".");
        assert_eq!(
            info.imports[0].imported_name,
            ImportedName::Named("foo".to_string())
        );
    }

    #[test]
    fn relative_import_parent_package_with_module() {
        let info = parse("from ..pkg import bar\n");
        assert_eq!(info.imports[0].source, "..pkg");
        assert_eq!(
            info.imports[0].imported_name,
            ImportedName::Named("bar".to_string())
        );
    }

    #[test]
    fn relative_import_dotted_module() {
        let info = parse("from .mod import x\n");
        assert_eq!(info.imports[0].source, ".mod");
    }

    #[test]
    fn star_import_is_namespace() {
        let info = parse("from os import *\n");
        assert_eq!(info.imports[0].source, "os");
        assert_eq!(info.imports[0].imported_name, ImportedName::Namespace);
        assert_eq!(info.imports[0].local_name, "*");
    }

    #[test]
    fn exports_def_and_class() {
        let info = parse("def foo():\n    pass\n\nclass Bar:\n    pass\n");
        assert_eq!(export_names(&info), vec!["foo", "Bar"]);
    }

    #[test]
    fn underscore_definitions_are_private() {
        let info = parse("def _private():\n    pass\n\ndef public():\n    pass\n");
        assert_eq!(export_names(&info), vec!["public"]);
    }

    #[test]
    fn module_level_assignment_is_exported() {
        let info = parse("VERSION = \"1.0\"\n_internal = 1\n");
        assert_eq!(export_names(&info), vec!["VERSION"]);
    }

    #[test]
    fn annotated_assignment_is_exported() {
        let info = parse("VERSION: str = \"1.0\"\n");
        assert_eq!(export_names(&info), vec!["VERSION"]);
    }

    #[test]
    fn annotated_declaration_without_value_is_exported() {
        // Common in .pyi stubs: `NAME: type` with no assignment.
        let info = parse("DEBUG: bool\n");
        assert_eq!(export_names(&info), vec!["DEBUG"]);
    }

    #[test]
    fn underscore_annotated_assignment_is_private() {
        let info = parse("_secret: int = 1\npublic: int = 2\n");
        assert_eq!(export_names(&info), vec!["public"]);
    }

    #[test]
    fn dunder_all_as_annotated_assignment_restricts_exports() {
        let info = parse(
            "__all__: list[str] = [\"foo\"]\n\ndef foo():\n    pass\n\ndef bar():\n    pass\n",
        );
        assert_eq!(export_names(&info), vec!["foo"]);
    }

    #[test]
    fn dunder_all_restricts_exports() {
        let info = parse("__all__ = [\"foo\"]\n\ndef foo():\n    pass\n\ndef bar():\n    pass\n");
        assert_eq!(export_names(&info), vec!["foo"]);
    }

    #[test]
    fn dunder_all_as_tuple() {
        let info = parse("__all__ = (\"a\", \"b\")\n\ndef a():\n    pass\n");
        assert_eq!(export_names(&info), vec!["a", "b"]);
    }

    #[test]
    fn dunder_all_does_not_leak_as_export() {
        let info = parse("__all__ = [\"foo\"]\n\ndef foo():\n    pass\n");
        assert!(!export_names(&info).contains(&"__all__".to_string()));
    }

    #[test]
    fn malformed_source_returns_empty_module_without_panic() {
        let info = parse("def (:\n  ???\n");
        // Line offsets are still populated so downstream stays consistent.
        assert!(!info.line_offsets.is_empty());
    }

    #[test]
    fn line_offsets_match_source() {
        let info = parse("import os\nimport sys\n");
        assert_eq!(info.line_offsets, vec![0, 10, 21]);
    }

    #[test]
    fn import_span_is_non_empty() {
        let info = parse("import os\n");
        let span = info.imports[0].span;
        assert!(span.end > span.start);
    }

    #[test]
    fn init_relative_from_import_becomes_re_export() {
        let info = parse_at("pkg/__init__.py", "from .api import public_fn\n");
        assert_eq!(info.re_exports.len(), 1);
        assert_eq!(info.re_exports[0].source, ".api");
        assert_eq!(info.re_exports[0].imported_name, "public_fn");
        assert_eq!(info.re_exports[0].exported_name, "public_fn");
        // Not double-counted as an import.
        assert!(info.imports.is_empty());
    }

    #[test]
    fn init_relative_from_import_alias_re_export_uses_alias_name() {
        let info = parse_at("pkg/__init__.py", "from .api import public_fn as pf\n");
        assert_eq!(info.re_exports[0].imported_name, "public_fn");
        assert_eq!(info.re_exports[0].exported_name, "pf");
    }

    #[test]
    fn init_star_re_export() {
        let info = parse_at("pkg/__init__.py", "from .api import *\n");
        assert_eq!(info.re_exports.len(), 1);
        assert_eq!(info.re_exports[0].imported_name, "*");
        assert_eq!(info.re_exports[0].exported_name, "*");
    }

    #[test]
    fn init_from_dot_import_submodule_stays_import() {
        // `from . import views` (no module) needs the imported name as the
        // submodule to resolve, so it remains an ordinary import.
        let info = parse_at("pkg/__init__.py", "from . import views\n");
        assert!(info.re_exports.is_empty());
        assert_eq!(info.imports.len(), 1);
        assert_eq!(info.imports[0].source, ".");
    }

    #[test]
    fn init_absolute_from_import_stays_import() {
        let info = parse_at("pkg/__init__.py", "from os import path\n");
        assert!(info.re_exports.is_empty());
        assert_eq!(info.imports.len(), 1);
    }

    #[test]
    fn non_init_relative_from_import_stays_import() {
        let info = parse_at("pkg/module.py", "from .api import public_fn\n");
        assert!(info.re_exports.is_empty());
        assert_eq!(info.imports.len(), 1);
        assert_eq!(info.imports[0].source, ".api");
    }

    #[test]
    fn type_checking_block_import_is_type_only() {
        let info = parse(
            "from typing import TYPE_CHECKING\n\nif TYPE_CHECKING:\n    from models import User\n",
        );
        let user = info
            .imports
            .iter()
            .find(|i| i.source == "models")
            .expect("models import captured");
        assert!(
            user.is_type_only,
            "imports under `if TYPE_CHECKING:` must be type-only"
        );
    }

    #[test]
    fn typing_dotted_type_checking_is_type_only() {
        let info =
            parse("import typing\n\nif typing.TYPE_CHECKING:\n    from models import User\n");
        let user = info.imports.iter().find(|i| i.source == "models");
        assert!(user.is_some_and(|i| i.is_type_only));
    }

    #[test]
    fn else_of_type_checking_block_is_runtime() {
        let info = parse("if TYPE_CHECKING:\n    from a import X\nelse:\n    from b import Y\n");
        let x = info
            .imports
            .iter()
            .find(|i| i.source == "a")
            .expect("a captured");
        let y = info
            .imports
            .iter()
            .find(|i| i.source == "b")
            .expect("b captured");
        assert!(x.is_type_only, "TYPE_CHECKING branch is type-only");
        assert!(!y.is_type_only, "else branch is runtime");
    }

    #[test]
    fn try_except_import_fallbacks_are_captured() {
        let info = parse(
            "try:\n    import fast_json as json\nexcept ImportError:\n    import slow_json as json\n",
        );
        let sources: Vec<&str> = info.imports.iter().map(|i| i.source.as_str()).collect();
        assert!(sources.contains(&"fast_json"));
        assert!(sources.contains(&"slow_json"));
        assert!(info.imports.iter().all(|i| !i.is_type_only));
    }

    #[test]
    fn conditional_top_level_import_is_runtime_not_type_only() {
        let info = parse("import sys\n\nif sys.version_info >= (3, 11):\n    import tomllib\n");
        let toml = info.imports.iter().find(|i| i.source == "tomllib");
        assert!(toml.is_some_and(|i| !i.is_type_only));
    }

    #[test]
    fn function_local_import_is_not_captured() {
        // Deferred imports (often used to break cycles) are intentionally skipped.
        let info = parse("def handler():\n    import heavy_module\n    return heavy_module\n");
        assert!(
            !info.imports.iter().any(|i| i.source == "heavy_module"),
            "function-local imports must not be captured, got: {:?}",
            info.imports
        );
    }

    #[test]
    fn class_body_import_is_not_captured() {
        let info = parse("class C:\n    import inside_class\n");
        assert!(!info.imports.iter().any(|i| i.source == "inside_class"));
    }
}
