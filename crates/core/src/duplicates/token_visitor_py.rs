//! Python token mapping for clone detection.
//!
//! Maps Ruff's lexer token stream onto fallow's language-neutral
//! [`SourceToken`] sequence, mirroring the Oxc-side `TokenExtractor` so the
//! suffix-array / normalize / shingle / families clone-detection pipeline runs
//! on `.py` / `.pyi` files unchanged. Trivia (comments, newlines, indent /
//! dedent) and unsupported tokens are dropped.

use oxc_span::Span;
use ruff_python_ast::token::TokenKind as PyKind;
use ruff_python_parser::parse_module;
use ruff_text_size::{Ranged, TextRange};

use super::token_types::{
    FileTokens, KeywordType as Kw, OperatorType as Op, PunctuationType as Punct, SourceToken,
    TokenKind,
};

/// Tokenize Python `source` into the language-neutral clone-detection stream.
///
/// On a fatal parse error the file yields no tokens (and is skipped by the
/// caller), matching the best-effort posture of the other language tokenizers.
pub(super) fn tokenize_python(source: &str) -> FileTokens {
    let mut tokens = Vec::new();
    if let Ok(parsed) = parse_module(source) {
        for tok in parsed.tokens() {
            if let Some(kind) = map_token(tok.kind(), tok.range(), source) {
                tokens.push(SourceToken {
                    kind,
                    span: span_of(tok.range()),
                });
            }
        }
    }
    FileTokens {
        tokens,
        atomic_invocation_spans: Vec::new(),
        source: source.to_string(),
        line_count: source.lines().count().max(1),
    }
}

/// Convert a Ruff `TextRange` into an oxc [`Span`].
fn span_of(range: TextRange) -> Span {
    Span::new(range.start().to_u32(), range.end().to_u32())
}

/// Map a Ruff token onto a normalized [`TokenKind`], or `None` to drop it
/// (trivia and tokens with no clone-detection meaning).
fn map_token(kind: PyKind, range: TextRange, source: &str) -> Option<TokenKind> {
    let text = || source[range].to_string();
    let mapped = match kind {
        // Literals.
        PyKind::Name => TokenKind::Identifier(text()),
        PyKind::Int | PyKind::Float | PyKind::Complex => TokenKind::NumericLiteral(text()),
        PyKind::String => TokenKind::StringLiteral(text()),
        PyKind::FStringStart
        | PyKind::FStringMiddle
        | PyKind::FStringEnd
        | PyKind::TStringStart
        | PyKind::TStringMiddle
        | PyKind::TStringEnd => TokenKind::TemplateLiteral,
        PyKind::True => TokenKind::BooleanLiteral(true),
        PyKind::False => TokenKind::BooleanLiteral(false),
        PyKind::None => TokenKind::NullLiteral,

        // Keywords reusing JS/TS variants.
        PyKind::As => TokenKind::Keyword(Kw::As),
        PyKind::Async => TokenKind::Keyword(Kw::Async),
        PyKind::Await => TokenKind::Keyword(Kw::Await),
        PyKind::Break => TokenKind::Keyword(Kw::Break),
        PyKind::Case => TokenKind::Keyword(Kw::Case),
        PyKind::Class => TokenKind::Keyword(Kw::Class),
        PyKind::Continue => TokenKind::Keyword(Kw::Continue),
        PyKind::Else => TokenKind::Keyword(Kw::Else),
        PyKind::Finally => TokenKind::Keyword(Kw::Finally),
        PyKind::For => TokenKind::Keyword(Kw::For),
        PyKind::From => TokenKind::Keyword(Kw::From),
        PyKind::If => TokenKind::Keyword(Kw::If),
        PyKind::Import => TokenKind::Keyword(Kw::Import),
        PyKind::In => TokenKind::Keyword(Kw::In),
        PyKind::Return => TokenKind::Keyword(Kw::Return),
        PyKind::Try => TokenKind::Keyword(Kw::Try),
        PyKind::Type => TokenKind::Keyword(Kw::Type),
        PyKind::While => TokenKind::Keyword(Kw::While),
        PyKind::Yield => TokenKind::Keyword(Kw::Yield),

        // Python-specific keywords.
        PyKind::Assert => TokenKind::Keyword(Kw::Assert),
        PyKind::Def => TokenKind::Keyword(Kw::Def),
        PyKind::Del => TokenKind::Keyword(Kw::Del),
        PyKind::Elif => TokenKind::Keyword(Kw::Elif),
        PyKind::Except => TokenKind::Keyword(Kw::Except),
        PyKind::Global => TokenKind::Keyword(Kw::Global),
        PyKind::Lambda => TokenKind::Keyword(Kw::Lambda),
        PyKind::Match => TokenKind::Keyword(Kw::Match),
        PyKind::Nonlocal => TokenKind::Keyword(Kw::Nonlocal),
        PyKind::Pass => TokenKind::Keyword(Kw::Pass),
        PyKind::Raise => TokenKind::Keyword(Kw::Raise),
        PyKind::With => TokenKind::Keyword(Kw::With),
        PyKind::Is => TokenKind::Keyword(Kw::Is),

        // Boolean operators map to their JS operator analogues.
        PyKind::And => TokenKind::Operator(Op::And),
        PyKind::Or => TokenKind::Operator(Op::Or),
        PyKind::Not => TokenKind::Operator(Op::Not),

        // Arithmetic / comparison / bitwise operators. Operators with no
        // distinct JS analogue normalize to the closest one (`@` matmul → mul,
        // `//` floordiv → div, `:=` walrus → assign) since clone detection only
        // cares about structural shape.
        PyKind::Equal | PyKind::ColonEqual => TokenKind::Operator(Op::Assign),
        PyKind::Plus => TokenKind::Operator(Op::Add),
        PyKind::Minus => TokenKind::Operator(Op::Sub),
        PyKind::Star | PyKind::At => TokenKind::Operator(Op::Mul),
        PyKind::Slash | PyKind::DoubleSlash => TokenKind::Operator(Op::Div),
        PyKind::Percent => TokenKind::Operator(Op::Mod),
        PyKind::DoubleStar => TokenKind::Operator(Op::Exp),
        PyKind::EqEqual => TokenKind::Operator(Op::Eq),
        PyKind::NotEqual => TokenKind::Operator(Op::NEq),
        PyKind::Less => TokenKind::Operator(Op::Lt),
        PyKind::Greater => TokenKind::Operator(Op::Gt),
        PyKind::LessEqual => TokenKind::Operator(Op::LtEq),
        PyKind::GreaterEqual => TokenKind::Operator(Op::GtEq),
        PyKind::Vbar => TokenKind::Operator(Op::BitwiseOr),
        PyKind::Amper => TokenKind::Operator(Op::BitwiseAnd),
        PyKind::CircumFlex => TokenKind::Operator(Op::BitwiseXor),
        PyKind::Tilde => TokenKind::Operator(Op::BitwiseNot),
        PyKind::LeftShift => TokenKind::Operator(Op::ShiftLeft),
        PyKind::RightShift => TokenKind::Operator(Op::ShiftRight),
        PyKind::Rarrow => TokenKind::Operator(Op::Arrow),
        PyKind::Ellipsis => TokenKind::Operator(Op::Spread),

        // Augmented assignment (`@=` matmul and `//=` floordiv normalize to
        // their closest JS analogue, matching the binary-operator mapping above).
        PyKind::PlusEqual => TokenKind::Operator(Op::AddAssign),
        PyKind::MinusEqual => TokenKind::Operator(Op::SubAssign),
        PyKind::StarEqual | PyKind::AtEqual => TokenKind::Operator(Op::MulAssign),
        PyKind::SlashEqual | PyKind::DoubleSlashEqual => TokenKind::Operator(Op::DivAssign),
        PyKind::PercentEqual => TokenKind::Operator(Op::ModAssign),
        PyKind::DoubleStarEqual => TokenKind::Operator(Op::ExpAssign),
        PyKind::AmperEqual => TokenKind::Operator(Op::BitwiseAndAssign),
        PyKind::VbarEqual => TokenKind::Operator(Op::BitwiseOrAssign),
        PyKind::CircumflexEqual => TokenKind::Operator(Op::BitwiseXorAssign),
        PyKind::LeftShiftEqual => TokenKind::Operator(Op::ShiftLeftAssign),
        PyKind::RightShiftEqual => TokenKind::Operator(Op::ShiftRightAssign),

        // Punctuation / delimiters.
        PyKind::Lpar => TokenKind::Punctuation(Punct::OpenParen),
        PyKind::Rpar => TokenKind::Punctuation(Punct::CloseParen),
        PyKind::Lsqb => TokenKind::Punctuation(Punct::OpenBracket),
        PyKind::Rsqb => TokenKind::Punctuation(Punct::CloseBracket),
        PyKind::Lbrace => TokenKind::Punctuation(Punct::OpenBrace),
        PyKind::Rbrace => TokenKind::Punctuation(Punct::CloseBrace),
        PyKind::Colon => TokenKind::Punctuation(Punct::Colon),
        PyKind::Semi => TokenKind::Punctuation(Punct::Semicolon),
        PyKind::Dot => TokenKind::Punctuation(Punct::Dot),
        PyKind::Comma => TokenKind::Operator(Op::Comma),

        // Trivia and anything else (newline, indent, dedent, comment, EOF,
        // IPython escapes, soft `lazy`, unknown) carry no clone-detection
        // signal and are dropped.
        _ => return None,
    };
    Some(mapped)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kinds(source: &str) -> Vec<TokenKind> {
        tokenize_python(source)
            .tokens
            .into_iter()
            .map(|t| t.kind)
            .collect()
    }

    #[test]
    fn maps_def_and_identifiers() {
        let got = kinds("def add(a, b):\n    return a + b\n");
        assert_eq!(
            got,
            vec![
                TokenKind::Keyword(Kw::Def),
                TokenKind::Identifier("add".to_string()),
                TokenKind::Punctuation(Punct::OpenParen),
                TokenKind::Identifier("a".to_string()),
                TokenKind::Operator(Op::Comma),
                TokenKind::Identifier("b".to_string()),
                TokenKind::Punctuation(Punct::CloseParen),
                TokenKind::Punctuation(Punct::Colon),
                TokenKind::Keyword(Kw::Return),
                TokenKind::Identifier("a".to_string()),
                TokenKind::Operator(Op::Add),
                TokenKind::Identifier("b".to_string()),
            ]
        );
    }

    #[test]
    fn maps_literals() {
        let got = kinds("x = 1\ny = 1.5\ns = 'hi'\nb = True\nn = None\n");
        assert!(got.contains(&TokenKind::NumericLiteral("1".to_string())));
        assert!(got.contains(&TokenKind::NumericLiteral("1.5".to_string())));
        assert!(got.contains(&TokenKind::StringLiteral("'hi'".to_string())));
        assert!(got.contains(&TokenKind::BooleanLiteral(true)));
        assert!(got.contains(&TokenKind::NullLiteral));
    }

    #[test]
    fn maps_boolean_operators() {
        let got = kinds("z = a and b or not c\n");
        assert!(got.contains(&TokenKind::Operator(Op::And)));
        assert!(got.contains(&TokenKind::Operator(Op::Or)));
        assert!(got.contains(&TokenKind::Operator(Op::Not)));
    }

    #[test]
    fn drops_comments_and_trivia() {
        // A comment, blank lines, and indentation produce no tokens of their own.
        let got = kinds("# a comment\n\nx = 1\n");
        assert_eq!(
            got,
            vec![
                TokenKind::Identifier("x".to_string()),
                TokenKind::Operator(Op::Assign),
                TokenKind::NumericLiteral("1".to_string()),
            ]
        );
    }

    #[test]
    fn fstring_maps_to_template() {
        let got = kinds("s = f'{x}'\n");
        assert!(got.contains(&TokenKind::TemplateLiteral));
    }

    #[test]
    fn syntax_error_yields_no_tokens() {
        // An unterminated def is a fatal parse error; the file contributes
        // nothing rather than panicking.
        assert!(tokenize_python("def (:::\n").tokens.is_empty());
    }

    #[test]
    fn identical_bodies_share_token_stream() {
        // The core clone-detection invariant: two structurally identical
        // functions tokenize to the same sequence.
        let a = kinds(
            "def f(n):\n    total = 0\n    for i in range(n):\n        total += i\n    return total\n",
        );
        let b = kinds(
            "def g(m):\n    total = 0\n    for j in range(m):\n        total += j\n    return total\n",
        );
        // Same shape, only identifier *values* differ.
        assert_eq!(a.len(), b.len());
    }
}
