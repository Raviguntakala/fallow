//! Per-function cyclomatic and cognitive complexity over the Ruff AST.
//!
//! Mirrors the Oxc-side [`crate::complexity`] visitor so Python functions feed
//! the same `fallow health` ranking as JS/TS. One traversal, a frame per
//! function/lambda, both metrics computed together.
//!
//! ## Cyclomatic (McCabe): 1 + decision points
//! Follows Radon's table (the de-facto Python standard): `if`/`elif` (+1 each,
//! `else` +0), `for`/`while` (+1, plus +1 for a present loop-`else`), each
//! `except` handler (+1, plus +1 for a present `try`-`else`), ternary `if`
//! expression (+1), each boolean operator (`a and b and c` is +2: one per
//! operand link), comprehension generators (+1 each) and their `if` filters
//! (+1 each), `assert` (+1), and `match` arms (+1 per refutable pattern, +1 per
//! guard; an irrefutable `case _`/`case x` adds 0).
//!
//! Deliberate deviation from Radon: `with` is **not** counted. Entering a
//! `with` block is unconditional, so it is not a decision point in the McCabe
//! sense; this keeps cyclomatic meaning "branches" consistently across
//! languages (the JS side has no `with`-equivalent increment either).
//!
//! ## Cognitive (SonarSource): structural increments with nesting penalty
//! `if`/`for`/`while`/`except`/`match`/ternary add `1 + nesting`; `elif`/`else`
//! add a flat +1; each boolean-operator node and each comprehension clause add
//! a flat +1. Nested functions and lambdas start a fresh frame (nesting 0),
//! matching the JS visitor.

use ruff_python_ast::visitor::{self, Visitor};
use ruff_python_ast::{
    ExceptHandler, Expr, Parameters, Stmt, StmtAssert, StmtFor, StmtFunctionDef, StmtIf, StmtMatch,
    StmtTry, StmtWhile,
};

use fallow_types::extract::{FunctionComplexity, byte_offset_to_line_col};

use super::span_of;

/// Per-function state on the scope stack.
struct Frame {
    name: String,
    span: oxc_span::Span,
    cyclomatic: u16,
    cognitive: u16,
    nesting: u16,
    param_count: u8,
}

/// Visitor accumulating per-function complexity. Counting helpers no-op when
/// the stack is empty, so module-level decisions outside any function (which
/// fallow does not report) contribute nothing.
struct ComplexityVisitor<'src> {
    stack: Vec<Frame>,
    results: Vec<FunctionComplexity>,
    source: &'src str,
    line_offsets: &'src [u32],
}

impl<'src> ComplexityVisitor<'src> {
    const fn new(source: &'src str, line_offsets: &'src [u32]) -> Self {
        Self {
            stack: Vec::new(),
            results: Vec::new(),
            source,
            line_offsets,
        }
    }

    fn add_cyclo(&mut self, n: u16) {
        if let Some(frame) = self.stack.last_mut() {
            frame.cyclomatic = frame.cyclomatic.saturating_add(n);
        }
    }

    /// Flat cognitive increment (no nesting penalty): `elif`/`else`, boolean
    /// operators, comprehension clauses.
    fn cog_flat(&mut self, n: u16) {
        if let Some(frame) = self.stack.last_mut() {
            frame.cognitive = frame.cognitive.saturating_add(n);
        }
    }

    /// Structural cognitive increment: `1 + nesting`.
    fn cog_nested(&mut self) {
        if let Some(frame) = self.stack.last_mut() {
            frame.cognitive = frame.cognitive.saturating_add(1 + frame.nesting);
        }
    }

    fn inc_nesting(&mut self) {
        if let Some(frame) = self.stack.last_mut() {
            frame.nesting = frame.nesting.saturating_add(1);
        }
    }

    fn dec_nesting(&mut self) {
        if let Some(frame) = self.stack.last_mut() {
            frame.nesting = frame.nesting.saturating_sub(1);
        }
    }

    fn push_frame(&mut self, name: String, span: oxc_span::Span, param_count: u8) {
        self.stack.push(Frame {
            name,
            span,
            cyclomatic: 1,
            cognitive: 0,
            nesting: 0,
            param_count,
        });
    }

    fn pop_frame(&mut self) {
        let Some(frame) = self.stack.pop() else {
            return;
        };
        let (line, col) = byte_offset_to_line_col(self.line_offsets, frame.span.start);
        let end_line = byte_offset_to_line_col(self.line_offsets, frame.span.end).0;
        let source_hash = self
            .source
            .get(frame.span.start as usize..frame.span.end as usize)
            .map(|slice| fallow_cov_protocol::source_hash_for(slice.as_bytes()));
        self.results.push(FunctionComplexity {
            name: frame.name,
            line,
            col,
            cyclomatic: frame.cyclomatic,
            cognitive: frame.cognitive,
            line_count: end_line.saturating_sub(line) + 1,
            param_count: frame.param_count,
            source_hash,
            contributions: Vec::new(),
        });
    }
}

/// Count declared parameters, excluding a leading `self`/`cls` receiver (the
/// Python analogue of skipping TypeScript's `this`).
fn count_params(params: &Parameters) -> u8 {
    let mut count = params.posonlyargs.len() + params.args.len() + params.kwonlyargs.len();
    if params.vararg.is_some() {
        count += 1;
    }
    if params.kwarg.is_some() {
        count += 1;
    }
    let first = params.posonlyargs.first().or_else(|| params.args.first());
    if first.is_some_and(|p| matches!(p.parameter.name.as_str(), "self" | "cls")) {
        count = count.saturating_sub(1);
    }
    u8::try_from(count).unwrap_or(u8::MAX)
}

#[expect(clippy::cast_possible_truncation, reason = "operand count never huge")]
fn link_count(values_len: usize) -> u16 {
    values_len.saturating_sub(1) as u16
}

impl ComplexityVisitor<'_> {
    fn visit_body<'a>(&mut self, stmts: &'a [Stmt])
    where
        Self: Visitor<'a>,
    {
        for stmt in stmts {
            self.visit_stmt(stmt);
        }
    }

    /// Visit a nested block: bump nesting for its duration.
    fn walk_block<'a>(&mut self, stmts: &'a [Stmt])
    where
        Self: Visitor<'a>,
    {
        self.inc_nesting();
        self.visit_body(stmts);
        self.dec_nesting();
    }

    fn handle_func<'a>(&mut self, node: &'a StmtFunctionDef)
    where
        Self: Visitor<'a>,
    {
        self.push_frame(
            node.name.as_str().to_string(),
            span_of(node.range),
            count_params(&node.parameters),
        );
        self.visit_body(&node.body);
        self.pop_frame();
    }

    fn handle_if<'a>(&mut self, node: &'a StmtIf)
    where
        Self: Visitor<'a>,
    {
        self.add_cyclo(1);
        self.cog_nested();
        self.visit_expr(&node.test);
        self.walk_block(&node.body);
        for clause in &node.elif_else_clauses {
            if let Some(test) = &clause.test {
                self.add_cyclo(1);
                self.cog_flat(1);
                self.visit_expr(test);
            } else {
                self.cog_flat(1);
            }
            self.walk_block(&clause.body);
        }
    }

    fn handle_for<'a>(&mut self, node: &'a StmtFor)
    where
        Self: Visitor<'a>,
    {
        self.add_cyclo(1 + u16::from(!node.orelse.is_empty()));
        self.cog_nested();
        self.visit_expr(&node.iter);
        self.walk_block(&node.body);
        self.visit_body(&node.orelse);
    }

    fn handle_while<'a>(&mut self, node: &'a StmtWhile)
    where
        Self: Visitor<'a>,
    {
        self.add_cyclo(1 + u16::from(!node.orelse.is_empty()));
        self.cog_nested();
        self.visit_expr(&node.test);
        self.walk_block(&node.body);
        self.visit_body(&node.orelse);
    }

    fn handle_try<'a>(&mut self, node: &'a StmtTry)
    where
        Self: Visitor<'a>,
    {
        self.visit_body(&node.body);
        for handler in &node.handlers {
            let ExceptHandler::ExceptHandler(h) = handler;
            self.add_cyclo(1);
            self.cog_nested();
            self.walk_block(&h.body);
        }
        self.add_cyclo(u16::from(!node.orelse.is_empty()));
        self.visit_body(&node.orelse);
        self.visit_body(&node.finalbody);
    }

    fn handle_match<'a>(&mut self, node: &'a StmtMatch)
    where
        Self: Visitor<'a>,
    {
        self.cog_nested();
        self.visit_expr(&node.subject);
        for case in &node.cases {
            if !case.pattern.is_irrefutable() {
                self.add_cyclo(1);
            }
            if let Some(guard) = &case.guard {
                self.add_cyclo(1);
                self.visit_expr(guard);
            }
            self.walk_block(&case.body);
        }
    }

    fn handle_assert<'a>(&mut self, node: &'a StmtAssert)
    where
        Self: Visitor<'a>,
    {
        self.add_cyclo(1);
        self.visit_expr(&node.test);
        if let Some(msg) = &node.msg {
            self.visit_expr(msg);
        }
    }

    /// Cyclomatic/cognitive contribution of comprehension generators (`for`
    /// clauses) and their `if` filters, shared by all comprehension kinds.
    fn handle_generators<'a>(&mut self, generators: &'a [ruff_python_ast::Comprehension])
    where
        Self: Visitor<'a>,
    {
        for comp in generators {
            let filters = u16::try_from(comp.ifs.len()).unwrap_or(u16::MAX);
            self.add_cyclo(1 + filters);
            self.cog_flat(1);
            self.visit_expr(&comp.iter);
            for cond in &comp.ifs {
                self.cog_flat(1);
                self.visit_expr(cond);
            }
        }
    }
}

impl<'a> Visitor<'a> for ComplexityVisitor<'_> {
    fn visit_stmt(&mut self, stmt: &'a Stmt) {
        match stmt {
            Stmt::FunctionDef(node) => self.handle_func(node),
            Stmt::If(node) => self.handle_if(node),
            Stmt::For(node) => self.handle_for(node),
            Stmt::While(node) => self.handle_while(node),
            Stmt::Try(node) => self.handle_try(node),
            Stmt::Match(node) => self.handle_match(node),
            Stmt::Assert(node) => self.handle_assert(node),
            // ClassDef, With, and everything else: default recursion. The class
            // body / `with` body is walked so contained functions still get a
            // frame, but neither construct is itself a decision point.
            _ => visitor::walk_stmt(self, stmt),
        }
    }

    fn visit_expr(&mut self, expr: &'a Expr) {
        match expr {
            Expr::BoolOp(node) => {
                self.add_cyclo(link_count(node.values.len()));
                self.cog_flat(1);
                for value in &node.values {
                    self.visit_expr(value);
                }
            }
            Expr::If(node) => {
                self.add_cyclo(1);
                self.cog_nested();
                self.visit_expr(&node.test);
                self.inc_nesting();
                self.visit_expr(&node.body);
                self.visit_expr(&node.orelse);
                self.dec_nesting();
            }
            Expr::Lambda(node) => {
                let params = node.parameters.as_ref().map_or(0, |p| count_params(p));
                self.push_frame("<lambda>".to_string(), span_of(node.range), params);
                self.visit_expr(&node.body);
                self.pop_frame();
            }
            Expr::ListComp(node) => {
                self.handle_generators(&node.generators);
                self.visit_expr(&node.elt);
            }
            Expr::SetComp(node) => {
                self.handle_generators(&node.generators);
                self.visit_expr(&node.elt);
            }
            Expr::Generator(node) => {
                self.handle_generators(&node.generators);
                self.visit_expr(&node.elt);
            }
            Expr::DictComp(node) => {
                self.handle_generators(&node.generators);
                self.visit_expr(&node.key);
                self.visit_expr(&node.value);
            }
            _ => visitor::walk_expr(self, expr),
        }
    }
}

/// Compute per-function complexity for a parsed Python module body.
pub(super) fn compute_complexity(
    body: &[Stmt],
    source: &str,
    line_offsets: &[u32],
) -> Vec<FunctionComplexity> {
    let mut visitor = ComplexityVisitor::new(source, line_offsets);
    visitor.visit_body(body);
    visitor.results
}

#[cfg(test)]
mod tests {
    use super::*;
    use fallow_types::extract::compute_line_offsets;

    fn analyze(src: &str) -> Vec<FunctionComplexity> {
        let parsed = ruff_python_parser::parse_module(src).expect("valid python");
        let offsets = compute_line_offsets(src);
        compute_complexity(&parsed.syntax().body, src, &offsets)
    }

    fn find<'a>(results: &'a [FunctionComplexity], name: &str) -> &'a FunctionComplexity {
        results
            .iter()
            .find(|r| r.name == name)
            .unwrap_or_else(|| panic!("function '{name}' not in {results:?}"))
    }

    // ---- cyclomatic ----

    #[test]
    fn empty_func_is_one() {
        assert_eq!(find(&analyze("def f():\n    pass\n"), "f").cyclomatic, 1);
    }

    #[test]
    fn if_adds_one() {
        let r = analyze("def f(x):\n    if x:\n        return 1\n");
        assert_eq!(find(&r, "f").cyclomatic, 2);
    }

    #[test]
    fn if_elif_else_adds_two() {
        let r = analyze(
            "def f(x):\n    if x > 0:\n        return 1\n    elif x < 0:\n        return -1\n    else:\n        return 0\n",
        );
        assert_eq!(find(&r, "f").cyclomatic, 3);
    }

    #[test]
    fn for_adds_one() {
        let r = analyze("def f(xs):\n    for x in xs:\n        pass\n");
        assert_eq!(find(&r, "f").cyclomatic, 2);
    }

    #[test]
    fn for_else_adds_two() {
        let r = analyze("def f(xs):\n    for x in xs:\n        pass\n    else:\n        pass\n");
        assert_eq!(find(&r, "f").cyclomatic, 3);
    }

    #[test]
    fn while_adds_one() {
        let r = analyze("def f():\n    while True:\n        break\n");
        assert_eq!(find(&r, "f").cyclomatic, 2);
    }

    #[test]
    fn single_except_adds_one() {
        let r = analyze("def f():\n    try:\n        pass\n    except ValueError:\n        pass\n");
        assert_eq!(find(&r, "f").cyclomatic, 2);
    }

    #[test]
    fn two_excepts_add_two() {
        let r = analyze(
            "def f():\n    try:\n        pass\n    except ValueError:\n        pass\n    except KeyError:\n        pass\n",
        );
        assert_eq!(find(&r, "f").cyclomatic, 3);
    }

    #[test]
    fn try_else_adds_one() {
        let r = analyze(
            "def f():\n    try:\n        pass\n    except ValueError:\n        pass\n    else:\n        pass\n",
        );
        // except (+1) + try-else (+1) = base 1 + 2 = 3
        assert_eq!(find(&r, "f").cyclomatic, 3);
    }

    #[test]
    fn ternary_adds_one() {
        let r = analyze("def f(x):\n    return 1 if x else 0\n");
        assert_eq!(find(&r, "f").cyclomatic, 2);
    }

    #[test]
    fn single_boolean_op_adds_one() {
        let r = analyze("def f(a, b):\n    return a and b\n");
        assert_eq!(find(&r, "f").cyclomatic, 2);
    }

    #[test]
    fn boolean_chain_adds_per_link() {
        // `a and b and c` is one BoolOp node with three operands -> +2.
        let r = analyze("def f(a, b, c):\n    return a and b and c\n");
        assert_eq!(find(&r, "f").cyclomatic, 3);
    }

    #[test]
    fn mixed_boolean_ops_add_two() {
        // `a and b or c` -> or-node(+1) over and-node(+1) = +2.
        let r = analyze("def f(a, b, c):\n    return a and b or c\n");
        assert_eq!(find(&r, "f").cyclomatic, 3);
    }

    #[test]
    fn assert_adds_one() {
        let r = analyze("def f(x):\n    assert x\n");
        assert_eq!(find(&r, "f").cyclomatic, 2);
    }

    #[test]
    fn comprehension_adds_one() {
        let r = analyze("def f(xs):\n    return [x for x in xs]\n");
        assert_eq!(find(&r, "f").cyclomatic, 2);
    }

    #[test]
    fn comprehension_filter_adds_one_more() {
        let r = analyze("def f(xs):\n    return [x for x in xs if x]\n");
        assert_eq!(find(&r, "f").cyclomatic, 3);
    }

    #[test]
    fn nested_generators_add_per_clause() {
        let r = analyze("def f(xs, ys):\n    return [x for x in xs for y in ys]\n");
        assert_eq!(find(&r, "f").cyclomatic, 3);
    }

    #[test]
    fn match_counts_refutable_arms() {
        // two value patterns (+1 each); wildcard `case _` adds 0.
        let r = analyze(
            "def f(x):\n    match x:\n        case 1:\n            pass\n        case 2:\n            pass\n        case _:\n            pass\n",
        );
        assert_eq!(find(&r, "f").cyclomatic, 3);
    }

    #[test]
    fn match_guard_adds_one() {
        let r = analyze(
            "def f(x):\n    match x:\n        case int() if x > 0:\n            pass\n        case _:\n            pass\n",
        );
        // one refutable pattern (+1) + guard (+1) = base 1 + 2 = 3
        assert_eq!(find(&r, "f").cyclomatic, 3);
    }

    #[test]
    fn with_is_not_a_decision() {
        let r = analyze("def f(cm):\n    with cm:\n        pass\n");
        assert_eq!(find(&r, "f").cyclomatic, 1);
    }

    #[test]
    fn with_body_branches_still_count() {
        let r = analyze("def f(cm, x):\n    with cm:\n        if x:\n            pass\n");
        assert_eq!(find(&r, "f").cyclomatic, 2);
    }

    // ---- cognitive ----

    #[test]
    fn empty_func_cognitive_zero() {
        assert_eq!(find(&analyze("def f():\n    pass\n"), "f").cognitive, 0);
    }

    #[test]
    fn simple_if_cognitive_one() {
        let r = analyze("def f(x):\n    if x:\n        return 1\n");
        assert_eq!(find(&r, "f").cognitive, 1);
    }

    #[test]
    fn nested_if_cognitive_penalty() {
        let r = analyze("def f(x, y):\n    if x:\n        if y:\n            return 1\n");
        assert_eq!(find(&r, "f").cognitive, 3);
    }

    #[test]
    fn if_else_cognitive_two() {
        let r = analyze("def f(x):\n    if x:\n        return 1\n    else:\n        return 0\n");
        assert_eq!(find(&r, "f").cognitive, 2);
    }

    #[test]
    fn if_elif_else_cognitive_three() {
        let r = analyze(
            "def f(x):\n    if x > 0:\n        return 1\n    elif x < 0:\n        return -1\n    else:\n        return 0\n",
        );
        assert_eq!(find(&r, "f").cognitive, 3);
    }

    #[test]
    fn for_then_nested_if_cognitive() {
        let r = analyze("def f(xs):\n    for x in xs:\n        if x:\n            return x\n");
        assert_eq!(find(&r, "f").cognitive, 3);
    }

    #[test]
    fn boolean_chain_cognitive_one() {
        let r = analyze("def f(a, b, c):\n    return a and b and c\n");
        assert_eq!(find(&r, "f").cognitive, 1);
    }

    #[test]
    fn mixed_boolean_cognitive_two() {
        let r = analyze("def f(a, b, c):\n    return a and b or c\n");
        assert_eq!(find(&r, "f").cognitive, 2);
    }

    #[test]
    fn match_cognitive_one() {
        let r = analyze(
            "def f(x):\n    match x:\n        case 1:\n            pass\n        case _:\n            pass\n",
        );
        assert_eq!(find(&r, "f").cognitive, 1);
    }

    #[test]
    fn ternary_cognitive_one() {
        let r = analyze("def f(x):\n    return 1 if x else 0\n");
        assert_eq!(find(&r, "f").cognitive, 1);
    }

    // ---- structure ----

    #[test]
    fn nested_function_independent_frame() {
        let r = analyze(
            "def outer(x):\n    if x:\n        def inner(y):\n            if y:\n                return 1\n",
        );
        assert_eq!(find(&r, "outer").cognitive, 1);
        assert_eq!(find(&r, "inner").cognitive, 1);
    }

    #[test]
    fn method_excludes_self() {
        let r = analyze("class C:\n    def m(self, a, b):\n        if a:\n            pass\n");
        let m = find(&r, "m");
        assert_eq!(m.param_count, 2);
        assert_eq!(m.cyclomatic, 2);
    }

    #[test]
    fn classmethod_excludes_cls() {
        let r = analyze("class C:\n    def m(cls, a):\n        pass\n");
        assert_eq!(find(&r, "m").param_count, 1);
    }

    #[test]
    fn plain_function_counts_all_params() {
        let r = analyze("def f(a, b, c):\n    pass\n");
        assert_eq!(find(&r, "f").param_count, 3);
    }

    #[test]
    fn vararg_and_kwarg_counted() {
        let r = analyze("def f(a, *args, **kwargs):\n    pass\n");
        assert_eq!(find(&r, "f").param_count, 3);
    }

    #[test]
    fn lambda_tracked_as_function() {
        let r = analyze("g = lambda a, b: a and b\n");
        let lam = find(&r, "<lambda>");
        assert_eq!(lam.cyclomatic, 2);
        assert_eq!(lam.param_count, 2);
    }

    #[test]
    fn async_function_tracked() {
        let r = analyze("async def f(x):\n    if x:\n        return 1\n");
        assert_eq!(find(&r, "f").cyclomatic, 2);
    }

    #[test]
    fn async_for_counts_as_loop() {
        let r = analyze("async def f(xs):\n    async for x in xs:\n        pass\n");
        assert_eq!(find(&r, "f").cyclomatic, 2);
    }

    #[test]
    fn module_level_code_not_reported() {
        assert!(analyze("if True:\n    x = 1\n").is_empty());
    }

    #[test]
    fn line_and_count_populated() {
        let r = analyze("\n\ndef f():\n    return 1\n");
        let f = find(&r, "f");
        assert_eq!(f.line, 3);
        assert!(f.line_count >= 2);
        assert!(f.source_hash.is_some());
    }

    #[test]
    fn complex_function_combined() {
        let r = analyze(
            "def g(x, y):\n    if x > 0:\n        for i in range(x):\n            if y and i > 5:\n                return True\n    elif x < 0:\n        while y:\n            y -= 1\n    return True if x else False\n",
        );
        let g = find(&r, "g");
        // if(+1) for(+1) if(+1) and(+1) elif(+1) while(+1) ternary(+1) = base 1 + 7
        assert_eq!(g.cyclomatic, 8);
    }
}
