use std::collections::HashSet;
use std::sync::Arc;

use tower_lsp::lsp_types::{
    CompletionItem, CompletionItemKind, CompletionItemLabelDetails, Documentation, InsertTextFormat,
    MarkupContent, MarkupKind,
};

use crate::analysis::{self, LocalKind, SymRef, Val};
use crate::state::{FileData, State};
use crate::symbols::{base_type_name, Kind};
use crate::syntax::{self, KEYWORDS};

const SNIPPETS: &[(&str, &str, &str)] = &[
    ("method", "method declaration", "method ${1:Name}(${2})${3: returns (${4})}\n  requires ${5:true}\n  ensures ${6:true}\n{\n  $0\n}"),
    ("lemma", "lemma declaration", "lemma ${1:Name}(${2})\n  requires ${3:true}\n  ensures ${4:true}\n{\n  $0\n}"),
    ("function", "function declaration", "function ${1:Name}(${2}): ${3:int}\n{\n  $0\n}"),
    ("predicate", "predicate declaration", "predicate ${1:Name}(${2})\n{\n  $0\n}"),
    ("ghost predicate", "ghost predicate declaration", "ghost predicate ${1:Name}(${2})\n{\n  $0\n}"),
    ("datatype", "datatype declaration", "datatype ${1:Name} = ${2:Ctor}(${3})"),
    ("class", "class declaration", "class ${1:Name} {\n  constructor(${2})\n  {\n    $0\n  }\n}"),
    ("module", "module declaration", "module ${1:Name} {\n  $0\n}"),
    ("while", "while loop with invariant", "while ${1:condition}\n  invariant ${2:true}\n  decreases ${3}\n{\n  $0\n}"),
    ("for", "for loop with invariant", "for ${1:i} := ${2:0} to ${3:n}\n  invariant ${4:true}\n{\n  $0\n}"),
    ("forall", "forall statement", "forall ${1:x} | ${2:true}\n  ensures ${3:true}\n{\n  $0\n}"),
    ("if", "if statement", "if ${1:condition} {\n  $0\n}"),
    ("ifelse", "if/else statement", "if ${1:condition} {\n  $2\n} else {\n  $0\n}"),
    ("match", "match statement", "match ${1:e}\n  case ${2:Ctor} => $0"),
    ("calc", "calc statement", "calc {\n  ${1:a};\n==\n  ${2:b};\n}"),
    ("assert by", "assert with proof", "assert ${1:P} by {\n  $0\n}"),
];

const ATTRIBUTES: &[&str] = &[
    "abstemious", "assumption", "autocontracts", "auto_generated", "axiom", "compile", "extern", "focus",
    "fuel", "induction", "isolate_assertions", "nativeType", "only", "opaque", "options", "print",
    "resource_limit", "split_here", "tailrecursion", "test", "timeLimit", "timeLimitMultiplier", "trigger",
    "vcs_split_on_every_assert", "verify", "warnShadowing",
];

/// Longest stretch of text around the cursor assumed to belong to one datatype update.
const UPDATE_WINDOW: usize = 1 << 15;

/// Added to the word at the cursor so that a half-typed expression parses. Never offered.
const CURSOR_NAME: &str = "dafny_lsp_cursor";

pub fn item_kind(kind: Kind) -> CompletionItemKind {
    match kind {
        Kind::Module | Kind::Import => CompletionItemKind::MODULE,
        Kind::Class | Kind::Iterator => CompletionItemKind::CLASS,
        Kind::Trait => CompletionItemKind::INTERFACE,
        Kind::Datatype => CompletionItemKind::ENUM,
        Kind::Ctor => CompletionItemKind::ENUM_MEMBER,
        Kind::Destructor | Kind::Field => CompletionItemKind::FIELD,
        Kind::Newtype | Kind::TypeSynonym => CompletionItemKind::STRUCT,
        Kind::Method | Kind::Lemma => CompletionItemKind::METHOD,
        Kind::Constructor => CompletionItemKind::CONSTRUCTOR,
        Kind::Function | Kind::Predicate => CompletionItemKind::FUNCTION,
        Kind::Const => CompletionItemKind::CONSTANT,
    }
}

fn first_line(s: &str) -> String {
    s.lines().next().unwrap_or("").to_string()
}

fn symbol_item(s: &SymRef, rank: u8) -> CompletionItem {
    let sym = s.sym();
    CompletionItem {
        label: sym.display_name().to_string(),
        kind: Some(item_kind(sym.kind)),
        detail: Some(first_line(&sym.signature)),
        documentation: sym.doc.clone().map(|d| Documentation::MarkupContent(MarkupContent { kind: MarkupKind::Markdown, value: d })),
        sort_text: Some(format!("{rank}{}", sym.name)),
        ..Default::default()
    }
}

fn plain_item(label: &str, kind: CompletionItemKind, detail: Option<String>, rank: u8) -> CompletionItem {
    CompletionItem {
        label: label.to_string(),
        kind: Some(kind),
        detail,
        sort_text: Some(format!("{rank}{label}")),
        ..Default::default()
    }
}

/// True when the cursor is inside a comment or a string literal.
fn in_comment_or_string(file: &FileData, offset: usize) -> bool {
    if offset == 0 {
        return false;
    }
    let node = syntax::node_at(file.root(), offset - 1);
    syntax::ancestors(node).any(|n| match n.kind() {
        "line_comment" => n.start_byte() < offset && offset <= n.end_byte(),
        "block_comment" | "string_literal" | "char_literal" => n.start_byte() < offset && offset < n.end_byte(),
        _ => false,
    })
}

fn is_word_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || matches!(b, b'_' | b'\'' | b'?') || b >= 0x80
}

/// Start of the word that ends at `offset`.
fn word_start(src: &str, offset: usize) -> usize {
    let b = src.as_bytes();
    let mut start = offset;
    while start > 0 && is_word_byte(b[start - 1]) {
        start -= 1;
    }
    while !src.is_char_boundary(start) {
        start += 1;
    }
    start
}

/// End of the word that continues past `offset`.
fn word_end(src: &str, offset: usize) -> usize {
    offset + src.as_bytes()[offset..].iter().take_while(|b| is_word_byte(**b)).count()
}

/// The file text with comments and string and char literals blanked, keeping
/// byte offsets, so that brackets and commas inside them are not counted. Only
/// looks within `UPDATE_WINDOW` bytes of `around`, since this runs on most requests.
fn code_only(file: &FileData, around: usize) -> String {
    let window = around.saturating_sub(UPDATE_WINDOW)..around + UPDATE_WINDOW;
    let mut bytes = file.text.clone().into_bytes();
    let mut stack = vec![file.raw_tree.root_node()];
    while let Some(node) = stack.pop() {
        if node.end_byte() < window.start || node.start_byte() > window.end {
            continue;
        }
        if matches!(node.kind(), "line_comment" | "block_comment" | "string_literal" | "char_literal") {
            bytes[node.byte_range()].fill(b' ');
        } else {
            stack.extend(syntax::children(node));
        }
    }
    String::from_utf8(bytes).expect("only whole tokens were blanked")
}

/// The `(` of the datatype update `recv.(a := 1, ` when a field name goes at `at`.
fn update_open(code: &str, at: usize) -> Option<usize> {
    let head = code[..at].trim_end();
    if !head.ends_with(['(', ',']) {
        return None;
    }
    let mut depth = 0;
    for (i, c) in head.bytes().enumerate().rev() {
        match c {
            b')' | b']' | b'}' => depth += 1,
            b'(' if depth == 0 => {
                let recv = code[..i].trim_end();
                return (recv.ends_with('.') && !recv.ends_with("..")).then_some(i);
            }
            b'[' | b'{' if depth == 0 => return None,
            b'(' | b'[' | b'{' => depth -= 1,
            _ => {}
        }
    }
    None
}

/// True when the cursor is where a field name goes in a datatype update.
pub fn at_update_field(file: &FileData, offset: usize) -> bool {
    update_open(&code_only(file, offset), word_start(&file.text, offset)).is_some()
}

/// Fields set by the update that opens at `open`, except the entry at `cursor`.
fn update_fields_set(code: &str, open: usize, cursor: usize) -> Vec<String> {
    let mut set = Vec::new();
    let mut depth = 0;
    let mut start = open + 1;
    for (i, c) in code.bytes().enumerate().skip(start) {
        match c {
            b'(' | b'[' | b'{' => depth += 1,
            b')' | b']' | b'}' if depth > 0 => depth -= 1,
            b',' | b')' | b']' | b'}' | b';' if depth == 0 => {
                if !(start..=i).contains(&cursor) {
                    if let Some((name, _)) = code[start..i].split_once(":=") {
                        set.push(name.trim().to_string());
                    }
                }
                if c != b',' {
                    break;
                }
                start = i + 1;
            }
            _ => {}
        }
    }
    set
}

/// Completes a field name in the update that opens at `open`. Dafny allows only
/// destructors, each at most once, and all from one constructor.
fn update_items(st: &State, file: &Arc<FileData>, code: &str, open: usize, cursor: usize) -> Vec<CompletionItem> {
    let dot = code[..open].trim_end().len() - 1;
    let recv_end = code[..dot].trim_end().len();
    let segs = analysis::chain_before(code, recv_end);
    // Looked up inside the receiver: a let expression can end right where the receiver does.
    let candidates = match analysis::eval_chain(st, file, recv_end.saturating_sub(1), &segs) {
        Val::Unknown => scope_members(st, file),
        val => analysis::member_candidates(st, &val),
    };
    let set = update_fields_set(code, open, cursor);
    let fits = |m: &SymRef| {
        m.sym().kind == Kind::Destructor
            && !set.contains(&m.sym().name)
            && m.parent().is_some_and(|ctor| set.iter().all(|f| ctor.sym().children.iter().any(|c| c.name == *f)))
    };
    let mut seen = HashSet::new();
    candidates.iter().filter(|m| fits(m) && seen.insert(m.sym().name.clone())).map(|m| symbol_item(m, 0)).collect()
}

/// End of the `)` and `]` that directly follow `at`, and how many there are.
fn closers_after(code: &str, at: usize) -> (usize, usize) {
    let (mut end, mut count) = (at, 0);
    for (i, c) in code[at..].bytes().enumerate() {
        match c {
            b')' | b']' => {
                end = at + i + 1;
                count += 1;
            }
            c if c.is_ascii_whitespace() => {}
            _ => break,
        }
    }
    (end, count)
}

/// The `)` and `]` that close the brackets open at `at`, back to the enclosing `{`.
fn closers(code: &str, at: usize) -> String {
    let mut out = String::new();
    let mut depth = 0;
    for c in code[..at].bytes().rev() {
        match c {
            b')' | b']' | b'}' => depth += 1,
            b'(' | b'[' | b'{' if depth > 0 => depth -= 1,
            b'(' => out.push(')'),
            b'[' => out.push(']'),
            b'{' => break,
            _ => {}
        }
    }
    out
}

/// The file to analyse for a request at `offset`. A half-typed line usually has a
/// syntax error, and tree-sitter's recovery can lose the declaration around it.
/// Finishing the expression at the cursor with a placeholder name keeps everything
/// else in the tree. If that does not parse, falls back to blanking the line.
pub fn typing_view(doc: Arc<FileData>, offset: usize) -> Arc<FileData> {
    if doc.has_syntax_errors() {
        let src = &doc.text;
        let (start, end) = (word_start(src, offset), word_end(src, offset));
        let code = code_only(&doc, offset);
        // A number is already an expression, and a name cannot be added to it.
        let name = if src[start..end].starts_with(|c: char| c.is_ascii_digit()) { "" } else { CURSOR_NAME };
        let is_field = update_open(&code, start).is_some() && !code[end..].trim_start().starts_with(":=");
        let word = if is_field { format!("{name} := {CURSOR_NAME}") } else { name.to_string() };
        // Close the open brackets, keeping the closing brackets already after the cursor.
        let close = closers(&code, start);
        let (tail, closed) = closers_after(&code, end);
        let finished = format!("{word}{}{}", &src[end..tail], close.get(closed..).unwrap_or(""));
        // The statement may need its `;`, and a let expression also needs a body.
        let view = ["".to_string(), ";".to_string(), format!("; {CURSOR_NAME}")]
            .iter()
            .find_map(|ending| doc.with_replacement(end..tail, &format!("{finished}{ending}")));
        if let Some(view) = view {
            return Arc::new(view);
        }
    }
    match doc.without_line(doc.index.line_of(offset)) {
        Some(view) => Arc::new(view),
        None => doc,
    }
}

pub fn complete(st: &State, file: &Arc<FileData>, offset: usize, snippets: bool) -> Vec<CompletionItem> {
    if in_comment_or_string(file, offset) {
        return Vec::new();
    }
    let src = &file.text;
    let word_start = word_start(src, offset);
    let before = &src[..word_start];

    if before.ends_with("{:") || before.ends_with('@') {
        return ATTRIBUTES.iter().map(|a| plain_item(a, CompletionItemKind::PROPERTY, Some("attribute".into()), 0)).collect();
    }
    let code = code_only(file, offset);
    if before.ends_with('.') && !before.ends_with("..") {
        let recv_end = code[..word_start - 1].trim_end().len();
        let segs = analysis::chain_before(&code, recv_end);
        if segs.is_empty() {
            return Vec::new();
        }
        // Looked up inside the receiver: a let expression can end right where the receiver does.
        let val = analysis::eval_chain(st, file, recv_end - 1, &segs);
        return member_items(st, file, &val);
    }
    if let Some(open) = update_open(&code, word_start) {
        return update_items(st, file, &code, open, word_start);
    }

    let line_start = file.index.line_start(file.index.line_of(offset));
    let line_prefix = src[line_start..word_start].trim_start();
    general_items(st, file, word_start, snippets, line_prefix)
}

fn builtin_members(ty: &str) -> &'static [&'static str] {
    let base = base_type_name(ty);
    match base {
        "map" | "imap" => &["Keys", "Values", "Items"],
        "real" => &["Floor"],
        "ORDINAL" => &["IsLimit", "IsSucc", "Offset", "IsNat"],
        "array" => &["Length"],
        b if b.starts_with("array") => &["Length0", "Length1", "Length2"],
        _ => &[],
    }
}

fn member_items(st: &State, file: &Arc<FileData>, val: &Val) -> Vec<CompletionItem> {
    let mut items = Vec::new();
    let mut seen = HashSet::new();
    let mut push = |item: CompletionItem, items: &mut Vec<CompletionItem>| {
        if seen.insert(item.label.clone()) {
            items.push(item);
        }
    };
    match val {
        Val::Unknown => {
            // Without a known receiver type, offer the members of every type in scope.
            for m in scope_members(st, file) {
                push(symbol_item(&m, 1), &mut items);
            }
            for name in ["Keys", "Values", "Items", "Length", "Floor"] {
                push(plain_item(name, CompletionItemKind::PROPERTY, Some("built-in member".into()), 2), &mut items);
            }
        }
        Val::Typed(tv) if tv.decl.is_none() => {
            for name in builtin_members(&tv.text) {
                push(plain_item(name, CompletionItemKind::PROPERTY, Some(tv.text.clone()), 0), &mut items);
            }
        }
        _ => {
            let candidates = analysis::member_candidates(st, val);
            // On a value, constructors are only reachable as discriminators (`d.Cons?`).
            let on_value = matches!(val, Val::Typed(..) | Val::Callable(_));
            for m in &candidates {
                let kind = m.sym().kind;
                if kind == Kind::Import {
                    continue;
                }
                if kind == Kind::Ctor && on_value {
                    let label = format!("{}?", m.sym().name);
                    push(plain_item(&label, CompletionItemKind::PROPERTY, Some(format!("is {}", m.sym().name)), 1), &mut items);
                } else {
                    push(symbol_item(m, 0), &mut items);
                }
            }
        }
    }
    items
}

/// Members of every type in scope, used when the receiver type is unknown.
fn scope_members(st: &State, file: &Arc<FileData>) -> Vec<SymRef> {
    let mut files = vec![file.clone()];
    files.extend(st.scope_files(file));
    files.iter().flat_map(|f| all_types(&analysis::top_level(f))).flat_map(|t| analysis::members_of(st, &t, 3)).collect()
}

fn all_types(syms: &[SymRef]) -> Vec<SymRef> {
    let mut out = Vec::new();
    for s in syms {
        let kind = s.sym().kind;
        if kind.is_type() {
            out.push(s.clone());
        } else if kind == Kind::Module {
            out.extend(all_types(&s.children()));
        }
    }
    out
}

fn general_items(st: &State, file: &Arc<FileData>, offset: usize, snippets: bool, line_prefix: &str) -> Vec<CompletionItem> {
    let mut items = Vec::new();
    let mut seen = HashSet::new();

    for l in analysis::locals_at(file, offset) {
        if seen.insert(l.name.clone()) {
            let kind = if l.kind == LocalKind::TypeParam { CompletionItemKind::TYPE_PARAMETER } else { CompletionItemKind::VARIABLE };
            let detail = match &l.ty {
                Some(ty) => format!("{}: {}", l.name, ty),
                None => l.kind.label().to_string(),
            };
            items.push(plain_item(&l.name, kind, Some(detail), 0));
        }
    }

    for (depth, level) in analysis::scope_levels(st, file, offset).into_iter().enumerate() {
        let rank = if depth == 0 { 1 } else { 2 };
        for s in level {
            let sym = s.sym();
            if sym.kind == Kind::Constructor && sym.name == "_ctor" {
                continue;
            }
            if seen.insert(sym.name.clone()) {
                items.push(symbol_item(&s, rank));
            }
        }
    }

    // Import paths can name any module in the workspace.
    if line_prefix.starts_with("import") {
        for f in st.all_files() {
            for m in all_modules(&analysis::top_level(&f)) {
                if seen.insert(m.sym().name.clone()) {
                    items.push(symbol_item(&m, 1));
                }
            }
        }
    }

    for kw in KEYWORDS {
        if seen.insert(kw.to_string()) {
            items.push(plain_item(kw, CompletionItemKind::KEYWORD, None, 3));
        }
    }

    if snippets {
        for (label, detail, body) in SNIPPETS {
            items.push(CompletionItem {
                label: label.to_string(),
                kind: Some(CompletionItemKind::SNIPPET),
                label_details: Some(CompletionItemLabelDetails { detail: None, description: Some("snippet".into()) }),
                detail: Some(detail.to_string()),
                insert_text: Some(body.to_string()),
                insert_text_format: Some(InsertTextFormat::SNIPPET),
                sort_text: Some(format!("4{label}")),
                ..Default::default()
            });
        }
    }
    // In a position that declares a name, the placeholder looks like a declaration.
    items.retain(|i| !i.label.ends_with(CURSOR_NAME));
    items
}

fn all_modules(syms: &[SymRef]) -> Vec<SymRef> {
    let mut out = Vec::new();
    for s in syms {
        if s.sym().kind == Kind::Module {
            out.push(s.clone());
            out.extend(all_modules(&s.children()));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use tower_lsp::lsp_types::Url;

    fn labels(src: &str, cursor_marker: &str) -> Vec<String> {
        let st = State::default();
        let offset = src.find(cursor_marker).unwrap();
        let src = src.replacen(cursor_marker, "", 1);
        let uri = Url::parse("file:///ws/c.dfy").unwrap();
        st.put_doc(FileData::new(uri.clone(), Some(1), src, None, None));
        let f = typing_view(st.doc(&uri).unwrap(), offset);
        complete(&st, &f, offset, true).into_iter().map(|i| i.label).collect()
    }

    /// Sorted labels at the `<|>` marker.
    fn labels_as_server(src: &str) -> Vec<String> {
        let mut got = labels(src, "<|>");
        got.sort();
        got
    }

    #[test]
    fn member_completion_uses_declared_types() {
        let src = "datatype P = P(x: int, y: int) | Q { function Norm(): int { 0 } }\nmethod M(p: P) { var a := p.<|> }";
        let got = labels(src, "<|>");
        for want in ["x", "y", "Norm", "P?", "Q?"] {
            assert!(got.contains(&want.to_string()), "missing {want} in {got:?}");
        }
        assert!(!got.contains(&"method".to_string()));
    }

    #[test]
    fn module_member_completion() {
        let src = "module A { function F(): int { 1 } datatype D = Mk }\nmethod M() { var a := A.<|> }";
        let got = labels(src, "<|>");
        assert!(got.contains(&"F".to_string()) && got.contains(&"D".to_string()) && got.contains(&"Mk".to_string()), "{got:?}");
    }

    #[test]
    fn general_completion_has_locals_and_keywords() {
        let src = "method M(count: int) {\n  var total := 0;\n  <|>\n}";
        let got = labels(src, "<|>");
        for want in ["count", "total", "M", "assert", "while"] {
            assert!(got.contains(&want.to_string()), "missing {want}");
        }
    }

    #[test]
    fn completion_while_the_line_is_incomplete() {
        let st = State::default();
        let uri = Url::parse("file:///ws/t.dfy").unwrap();
        let good = "module App {\n  datatype S = A(x: int)\n  method Use(s: S) {\n    var a := 1;\n  }\n}\n";
        st.put_doc(FileData::new(uri.clone(), Some(1), good.to_string(), None, None));
        let typed = good.replace("var a := 1;\n", "var a := 1;\n    var q := s.\n");
        let doc = st.put_doc(FileData::new(uri.clone(), Some(2), typed.clone(), None, None));
        assert!(doc.has_syntax_errors());
        // the outline survives the syntax error
        assert_eq!(doc.syms.symbols[0].children.iter().map(|s| s.name.as_str()).collect::<Vec<_>>(), vec!["S", "Use"]);
        let offset = typed.find("s.\n").unwrap() + 2;
        let got: Vec<String> = complete(&st, &doc, offset, false).into_iter().map(|i| i.label).collect();
        assert!(got.contains(&"x".to_string()) && got.contains(&"A?".to_string()), "{got:?}");
        let line_start = typed.find("    var q").unwrap();
        let got: Vec<String> = complete(&st, &doc, line_start + 4, false).into_iter().map(|i| i.label).collect();
        assert!(got.contains(&"a".to_string()) && got.contains(&"s".to_string()), "{got:?}");
    }

    #[test]
    fn no_completion_in_comments() {
        assert!(labels("// some <|> text\nmethod M() {}", "<|>").is_empty());
    }

    const UPDATE_DECLS: &str = "datatype D = A(x: int, y: int) | B(x: int, z: int) { const c := 1 function F(): int { 0 } }\n\
        datatype Option<T> = None | Some(value: T)\n\
        type Syn = D\n\
        class K { var f: int constructor() { f := 0; } }\n\
        function G(): D { A(0, 0) }\n\
        function H(a: int, b: int): int { 0 }\n";

    /// Sorted labels offered at the marker inside a method with these parameters.
    fn in_method(body: &str) -> Vec<String> {
        labels_as_server(&format!("{UPDATE_DECLS}method M(d: D, n: int, s: seq<int>, ds: seq<D>, o: Option<D>, syn: Syn, k: K) {{\n{body}\n}}"))
    }

    #[test]
    fn update_when_the_declaration_is_on_one_line() {
        for decl in [
            "function T(d: D): D { d.(<|>) }",
            "function T(d: D): D { d.(<|> }",
            "function T(d: D): D\n{ d.(<|>) }",
            "method T(d: D) { var a := d.(<|>); }",
            "method T(d: D) { var a := d.(<|>) }",
            "method T(d: D) { var a := d.(<|> }",
        ] {
            assert_eq!(labels_as_server(&format!("{UPDATE_DECLS}{decl}")), ["x", "y", "z"], "{decl}");
        }
        assert_eq!(labels_as_server(&format!("{UPDATE_DECLS}function T(d: D): D {{ d.(x := 1, <|>) }}")), ["y", "z"]);
        assert_eq!(labels_as_server(&format!("{UPDATE_DECLS}function T(d: D): D {{ d.(y := 1, <|>) }}")), ["x"]);
    }

    #[test]
    fn update_offers_only_destructors() {
        for body in ["var a := d.(<|>", "var a := d.(<|>);", "var a := d.(x<|>);", "var a := d.( <|> );", "var a := d.(\n  <|>\n);"] {
            assert_eq!(in_method(body), ["x", "y", "z"], "{body}");
        }
    }

    #[test]
    fn update_skips_fields_already_set() {
        assert_eq!(in_method("var a := d.(x := 1, <|>"), ["y", "z"]);
        assert_eq!(in_method("var a := d.(<|>, x := 1);"), ["y", "z"]);
        assert_eq!(in_method("var a := d.(\n  x := 1,\n  <|>\n);"), ["y", "z"]);
        // the entry being edited still offers its own name
        assert_eq!(in_method("var a := d.(x<|> := 1);"), ["x", "y", "z"]);
    }

    #[test]
    fn update_stays_in_one_constructor() {
        assert_eq!(in_method("var a := d.(y := 1, <|>"), ["x"]);
        assert_eq!(in_method("var a := d.(z := 1, <|>"), ["x"]);
        assert!(in_method("var a := d.(x := 1, y := 2, <|>").is_empty());
    }

    #[test]
    fn update_receivers() {
        for body in [
            "var a := G().(<|>",
            "var a := ds[0].(<|>",
            "var a := o.value.(<|>",
            "var a := syn.(<|>",
            "var a := d. (<|>",
            "var a := d\n  .(<|>",
            "var a := d.(x := 1).(<|>",
            "var a := d.(x := 1)\n  .(<|>",
            "var a := d.(x := H(1, 2)).(<|>",
            "var w := d;\nvar a := w.(<|>",
            "var w := d.(x := 1);\nvar a := w.(<|>",
            "var w := d;\nvar u := w;\nvar a := u.(<|>",
        ] {
            assert_eq!(in_method(body), ["x", "y", "z"], "{body}");
        }
        assert_eq!(labels_as_server(&format!("{UPDATE_DECLS}function T(d: D): D {{ var w := d; w.(<|>) }}")), ["x", "y", "z"]);
        assert!(in_method("var a := k.(<|>").is_empty());
        assert!(in_method("var a := s.(<|>").is_empty());
    }

    #[test]
    fn update_with_unknown_receiver_offers_destructors_in_scope() {
        assert_eq!(in_method("var a := (d).(<|>"), ["value", "x", "y", "z"]);
        assert_eq!(in_method("var a := missing.(<|>"), ["value", "x", "y", "z"]);
    }

    #[test]
    fn update_ignores_brackets_in_comments_and_literals() {
        for body in [
            "var a := d.(x := H(1, 2), <|>",
            "var a := d.(x := 1, /* ( */ <|>",
            "var a := d.(x := 1, // note (\n  <|>\n);",
            "var a := d.( // note, more (\n  x := 1,\n  <|>\n);",
            "var a := d.(x := |\"(\"|, <|>",
            "var a := d.(x := if '(' == ')' then 1 else 2, <|>",
        ] {
            assert_eq!(in_method(body), ["y", "z"], "{body}");
        }
    }

    #[test]
    fn update_value_position_keeps_normal_completion() {
        for body in [
            "var a := d.(x := <|>);",
            "var a := d.(x := n + <|>);",
            "var a := d.(x := H(<|>));",
            "var a := d.(x := H(1, <|>));",
            "var a := d.(x := [1, <|>]);",
        ] {
            let got = in_method(body);
            assert!(got.contains(&"n".to_string()) && got.contains(&"assert".to_string()), "{body}: {got:?}");
        }
        let got = in_method("var a := d.(x := G().<|>);");
        assert!(got.contains(&"F".to_string()) && !got.contains(&"n".to_string()), "{got:?}");
    }

    #[test]
    fn member_access_after_update() {
        for body in ["var a := d.(x := 1).<|>", "var w := d;\nvar a := w.<|>"] {
            let got = in_method(body);
            for want in ["x", "y", "z", "F", "c", "A?", "B?"] {
                assert!(got.contains(&want.to_string()), "{body}: missing {want} in {got:?}");
            }
            assert!(!got.contains(&"value".to_string()), "{body}: {got:?}");
        }
    }

    #[test]
    fn member_access_when_the_declaration_is_on_one_line() {
        for decl in [
            "function T(d: D): int { d.<|> }",
            "function T(d: D): int { d.x<|> }",
            // a partial word that is a keyword does not parse
            "function T(d: D): int { d.in<|> }",
            "function T(d: D): int { H(d.<|>) }",
            "function T(d: D): int { H(1, d.<|>) }",
            "function T(d: D): int { if d.<|> then 1 else 2 }",
            "function T(d: D): int requires d.<|> { 0 }",
            "function T(d: D): int { var w := d; w.<|> }",
            "function T(ds: seq<D>): int { ds[0].<|> }",
            "function T(o: Option<D>): int { o.value.<|> }",
            "function T(d: D): int { d.(x := 1).<|> }",
            "function T(): int { G().<|> }",
            "method T(d: D) { var a := d.<|> }",
            "method T(d: D) { var a := d.<|>; }",
            "method T(d: D) { d.<|> }",
            "lemma T(d: D) ensures d.<|> { }",
            "class C2 { var d: D method T() { d.<|> } }",
            "class C2 { var d: D function T(): int { this.d.<|> } }",
        ] {
            assert_eq!(labels_as_server(&format!("{UPDATE_DECLS}{decl}")), ["A?", "B?", "F", "c", "x", "y", "z"], "{decl}");
        }
        assert_eq!(labels_as_server(&format!("{UPDATE_DECLS}function T(m: map<int, int>): int {{ m.<|> }}")), ["Items", "Keys", "Values"]);
        // a type that really is unknown still falls back to every type's members
        let got = labels_as_server(&format!("{UPDATE_DECLS}function T(): int {{ missing.<|> }}"));
        assert!(got.contains(&"value".to_string()) && got.contains(&"x".to_string()), "{got:?}");
    }

    #[test]
    fn member_access_when_the_dot_is_on_a_new_line() {
        for body in [
            "var a := d\n  .<|>",
            "var a := d\n  .<|>;",
            "var a := d\n  .x<|>",
            "var a := ds[0]\n  .<|>",
            "var a := o\n  .value\n  .<|>",
            "var a := d\n  .(x := 1)\n  .<|>",
            "var a := d // note\n  .<|>",
            "var a := d /* note */ .<|>",
            "var a := d /* caf\u{e9} ( */ .<|>",
            "var a := d .<|>",
        ] {
            assert_eq!(in_method(body), ["A?", "B?", "F", "c", "x", "y", "z"], "{body}");
        }
        assert_eq!(labels_as_server(&format!("{UPDATE_DECLS}function T(d: D): int {{ d\n  .<|> }}")), ["A?", "B?", "F", "c", "x", "y", "z"]);
        // nothing to the left of the dot
        assert!(in_method("var a := 1;\n.<|>").is_empty());
        assert!(in_method(".<|>").is_empty());
    }

    #[test]
    fn general_list_when_the_declaration_is_on_one_line() {
        for decl in [
            "function T(d: D, n: int): int { n + <|> }",
            "function T(d: D, n: int): int { <|> }",
            "function T(d: D, n: int): int { n + in<|> }",
            "function T(d: D, n: int): int { H(1, <|>) }",
            "function T(d: D, n: int): int { H(1, <|> }",
            "function T(d: D, n: int): int { if <|> then 1 else 2 }",
            "function T(d: D, n: int): int requires n > <|> { 0 }",
            "function T(d: D, n: int): int { var w := <|> }",
            "function T(d: D, n: int): D { d.(x := <|>) }",
            "method T(d: D, n: int) { var a := <|> }",
            "method T(d: D, n: int) { var a := n + <|>; }",
            "method T(d: D, n: int) { assert n > <|>; }",
            "method T(d: D, n: int) { var a := H(1, <|> }",
            "method T(d: D, n: int) { while n > <|> { } }",
            "lemma T(d: D, n: int) ensures n > <|> { }",
            "class C2 { function T(d: D, n: int): int { n + <|> } }",
        ] {
            let got = labels_as_server(&format!("{UPDATE_DECLS}{decl}"));
            for want in ["d", "n", "T", "H", "assert"] {
                assert!(got.contains(&want.to_string()), "{decl}: missing {want} in {got:?}");
            }
        }
        let got = labels_as_server(&format!("{UPDATE_DECLS}function T(n: int): int {{ var w := n; w + <|> }}"));
        assert!(got.contains(&"w".to_string()) && got.contains(&"n".to_string()), "{got:?}");
    }

    #[test]
    fn placeholder_is_never_offered() {
        for src in [
            "function T(n: int): int { n + <|> }",
            "method T() { var <|> }",
            "method T() { var fo<|> }",
            "datatype E = <|>",
            "datatype E = fo<|>",
            "const <|>",
            "import <|>",
        ] {
            let got = labels_as_server(src);
            assert!(!got.is_empty() && !got.iter().any(|l| l.contains(CURSOR_NAME)), "{src}: {got:?}");
        }
    }

    #[test]
    fn typing_view_keeps_a_clean_file_and_gives_up_on_other_errors() {
        let uri = Url::parse("file:///ws/v.dfy").unwrap();
        let clean = Arc::new(FileData::new(uri.clone(), Some(1), "function F(n: int): int { n }\n".into(), None, None));
        assert!(Arc::ptr_eq(&clean, &typing_view(clean.clone(), 27)));
        // A second error stops the placeholder from helping, and the text is left as it was.
        let src = "method Broken( {\n}\nfunction F(n: int): int { n + }\n";
        let doc = Arc::new(FileData::new(uri, Some(1), src.into(), None, None));
        let offset = src.find("+ }").unwrap() + 2;
        assert_eq!(typing_view(doc, offset).text, src);
    }

    #[test]
    fn other_parens_are_not_updates() {
        for body in ["var a := H(<|>", "var a := H(1, <|>", "var a := (<|>", "var a := s[1..(<|>", "var a := [1, <|>"] {
            let got = in_method(body);
            assert!(got.contains(&"n".to_string()), "{body}: {got:?}");
        }
    }

    #[test]
    fn at_update_field_decides_the_trigger_characters() {
        let at = |src: &str| {
            let offset = src.find("<|>").unwrap();
            let src = src.replacen("<|>", "", 1);
            let uri = Url::parse("file:///ws/t.dfy").unwrap();
            at_update_field(&FileData::new(uri, Some(1), src, None, None), offset)
        };
        assert!(at("method M(d: D) { var a := d.(<|>); }"));
        assert!(at("method M(d: D) { var a := d.(x := 1,<|>); }"));
        assert!(!at("method M() { var a := H(<|>); }"));
        assert!(!at("method M() { var a := H(1,<|>); }"));
        assert!(!at("method M(d: D) { var a := d.(x := H(1,<|>)); }"));
        assert!(!at("// d.(<|>\nmethod M() {}"));
    }

    #[test]
    fn update_while_a_multi_line_update_is_edited() {
        let st = State::default();
        let uri = Url::parse("file:///ws/u.dfy").unwrap();
        // E also has `x`, so an unresolved `w` would offer `other` as well.
        let good = "module App {\n  datatype E = E(x: int, other: int)\n  datatype D = A(x: int, y: int)\n  method M() {\n    var w := A(0, 0);\n    var a := w.(\n      x := 1\n    );\n  }\n}\n";
        st.put_doc(FileData::new(uri.clone(), Some(1), good.to_string(), None, None));
        let typed = good.replace("x := 1\n", "x := 1,\n      \n");
        let doc = st.put_doc(FileData::new(uri.clone(), Some(2), typed.clone(), None, None));
        assert!(doc.has_syntax_errors());
        let offset = typed.find("x := 1,\n      ").unwrap() + "x := 1,\n      ".len();
        let got: Vec<String> = complete(&st, &typing_view(doc, offset), offset, false).into_iter().map(|i| i.label).collect();
        assert_eq!(got, ["y"]);

        // The same edit before the statement has its `;`.
        let unfinished = typed.replace("    );\n", "    )\n");
        let doc = st.put_doc(FileData::new(uri.clone(), Some(3), unfinished, None, None));
        let got: Vec<String> = complete(&st, &typing_view(doc, offset), offset, false).into_iter().map(|i| i.label).collect();
        assert_eq!(got, ["y"]);
    }
}
