//! Dictionary-free, multilingual tokenizer.
//!
//! The pipeline (see crate docs for the rationale):
//!
//! 1. NFKC normalize + lowercase (unify full/half width, variant forms, case).
//! 2. Split the normalized text into runs of a single *script class*
//!    (Latin-ish / CJK-ish / digit / other), so a Japanese sentence written
//!    with embedded English identifiers is handled per-segment.
//! 3. **Space-delimited scripts** (Latin, Cyrillic, Greek, …) → UAX#29 word
//!    boundaries. `snake_case` / `kebab-case` / `camelCase` identifiers also emit
//!    their sub-tokens so partial matches survive.
//! 4. **Non-spacing scripts** (Han, Hiragana, Katakana, Hangul) → segmented via
//!    [`crate::segmenter::push_segmented_ja`]: Japanese runs (hiragana /
//!    katakana / kanji) are split into words by a trained boundary model;
//!    non-Japanese non-spacing runs (e.g. Hangul) and single-character runs
//!    fall back to character bi-grams, the dictionary-free scheme Apache
//!    Lucene's CJK analyzer uses.
//! 5. Across the whole (normalized) text we additionally emit character
//!    **3-grams** (CL-CnG: Cross-Language Character N-Gram), prefixed so they
//!    never collide with word tokens. These pick up identifiers, proper nouns
//!    and spelling variants in a language-independent way.

use unicode_normalization::UnicodeNormalization;
use unicode_segmentation::UnicodeSegmentation;

/// Marker prepended to character 3-grams so they live in a distinct token
/// namespace from word tokens (a 3-gram `"abc"` must not match a word `"abc"`).
const NGRAM_PREFIX: char = '\u{1}';

/// Length of the cross-language character n-gram.
const CL_NGRAM: usize = 3;

/// Tokenize `text` into a flat list of tokens (words / segmented Japanese
/// words / CL-CnG trigrams). Order is deterministic; duplicates are kept
/// (BM25 needs term frequencies). For set-based metrics (Jaccard) deduplicate
/// downstream.
///
/// All emitted tokens are lowercased. NFKC normalization happens first, but
/// case folding is deferred to token emission so that `camelCase` boundaries
/// survive long enough to be split (lowercasing the whole string up front would
/// erase them).
pub fn tokenize(text: &str) -> Vec<String> {
    // NFKC only (case preserved) so camelCase splitting can see boundaries.
    let cased: String = text.nfkc().collect();
    let mut out = Vec::new();

    for segment in script_segments(&cased) {
        match segment.class {
            ScriptClass::Spacing => {
                push_word_tokens(segment.text, &mut out);
            }
            ScriptClass::NonSpacing => {
                crate::segmenter::push_segmented_ja(segment.text, &mut out);
            }
            ScriptClass::Other => {}
        }
    }

    // CL-CnG runs over the fully normalized (lowercased) text — character
    // n-grams don't benefit from case and must match case-insensitively.
    let lowered = lowercase(&cased);
    push_char_ngrams(&lowered, &mut out);
    out
}

/// Returns `true` if `token` is a cross-language character n-gram (CL-CnG)
/// rather than a word token. CL-CnG tokens are prefixed with an internal
/// marker and are useful for matching but not for human-facing output.
pub fn is_cl_ngram(token: &str) -> bool {
    token.starts_with(NGRAM_PREFIX)
}

/// Boost for a topic/subject-marked content word (`Xは` / `Xが`). The particle
/// marks X as what the sentence is *about* — the strongest relevance signal a
/// query term can carry.
pub const TOPIC_BOOST: f64 = 2.0;
/// Boost for a direct object (`Xを`).
pub const OBJECT_BOOST: f64 = 1.8;
/// Boost for case-marked complements (`Xで` / `Xに` / `Xから` / `Xへ` /
/// `Xまで` / `Xより`): means, target, origin — relevant but less central than
/// topic/object.
pub const CASE_BOOST: f64 = 1.5;

/// A token with a particle-context weight for weighted BM25 scoring
/// ([`crate::Corpus::bm25_scores_weighted`]).
#[derive(Debug, Clone, PartialEq)]
pub struct WeightedToken {
    pub token: String,
    /// `1.0` = default content word, `>1.0` = boosted (topic/object/case-
    /// marked), `0.0` = stopword or CL-CnG trigram (excluded from scoring).
    pub weight: f64,
}

/// True for characters that end a sentence or line: a boost particle after
/// one of these must not boost a word from before it. ASCII `.` is *not*
/// listed — `classify` routes it into Spacing segments as an identifier
/// connector (`0.13.0`), so it never reaches an Other segment.
fn is_boost_barrier(c: char) -> bool {
    matches!(c, '。' | '！' | '？' | '!' | '?' | '\n' | '\r')
}

/// Boost factor carried by a Japanese case particle onto the content word
/// immediately before it; `None` for anything that is not a boost particle.
fn particle_boost(token: &str) -> Option<f64> {
    match token {
        "は" | "が" => Some(TOPIC_BOOST),
        "を" => Some(OBJECT_BOOST),
        "で" | "に" | "から" | "へ" | "まで" | "より" => Some(CASE_BOOST),
        _ => None,
    }
}

/// Like [`tokenize`], but each token carries a weight inferred from the
/// Japanese particle following it — a dictionary-free stand-in for
/// part-of-speech tagging (`Xは` marks X as topic, `Xを` as object, ...).
///
/// Emits the same token multiset as [`tokenize`]:
/// - stopwords and CL-CnG trigrams are kept, with `weight = 0.0`;
/// - a content word followed by a boost particle gets that particle's boost
///   ([`TOPIC_BOOST`] / [`OBJECT_BOOST`] / [`CASE_BOOST`]); for an identifier
///   this applies to the whole identifier *and* its sub-tokens
///   (`atomic_write は` boosts `atomic_write`, `atomic`, `write`);
/// - every other content word has `weight = 1.0`.
///
/// "Followed by" is judged over the content-token stream: whitespace,
/// brackets and quotes between the word and the particle are transparent
/// (`「メモリ」は` still boosts `メモリ`), but sentence-final punctuation
/// (`。` `！` `？` `!` `?`) and newlines block the boost — a particle never
/// boosts a word from the previous sentence or line.
///
/// ```
/// use lexsim::{tokenize_weighted, TOPIC_BOOST};
///
/// let weighted = tokenize_weighted("メモリは重要");
/// let memory = weighted.iter().find(|wt| wt.token == "メモリ").unwrap();
/// assert_eq!(memory.weight, TOPIC_BOOST);
/// ```
pub fn tokenize_weighted(text: &str) -> Vec<WeightedToken> {
    let cased: String = text.nfkc().collect();

    // Emission groups: all tokens that originate from one source word (an
    // identifier plus its sub-tokens, or one segmented Japanese word). A
    // boost particle boosts the whole *group* before it, so `atomic_write は`
    // boosts the identifier and its sub-tokens alike. `barrier_before[i]`
    // records a sentence/line boundary between group i-1 and group i, which
    // blocks the boost from crossing it.
    let mut groups: Vec<Vec<String>> = Vec::new();
    let mut barrier_before: Vec<bool> = Vec::new();
    let mut pending_barrier = false;
    for segment in script_segments(&cased) {
        match segment.class {
            ScriptClass::Spacing => {
                // A Spacing segment can never contain whitespace (classify()
                // sends whitespace to Other), so this loop yields exactly one
                // chunk; split_whitespace is belt-and-braces against that
                // invariant changing. Tokens per chunk are the same multiset
                // push_word_tokens emits (word + identifier sub-tokens).
                for raw in segment.text.split_whitespace() {
                    let mut group = Vec::new();
                    for word in raw.unicode_words() {
                        group.push(lowercase(word));
                        push_identifier_subtokens(word, &mut group);
                    }
                    push_identifier_subtokens(raw, &mut group);
                    if !group.is_empty() {
                        groups.push(group);
                        barrier_before.push(std::mem::take(&mut pending_barrier));
                    }
                }
            }
            ScriptClass::NonSpacing => {
                let mut toks = Vec::new();
                crate::segmenter::push_segmented_ja(segment.text, &mut toks);
                for t in toks {
                    groups.push(vec![t]);
                    barrier_before.push(std::mem::take(&mut pending_barrier));
                }
            }
            ScriptClass::Other => {
                // Dropped from tokenization, but sentence-final punctuation
                // and newlines must still block a boost from crossing them.
                if segment.text.chars().any(is_boost_barrier) {
                    pending_barrier = true;
                }
            }
        }
    }

    // A group followed by a lone boost-particle group gets that particle's
    // boost. Boost particles only ever come out of the Japanese segmenter as
    // single-token groups; adjacency is over the content-token stream, with
    // sentence/line boundaries recorded as barriers.
    let mut out = Vec::new();
    for (i, group) in groups.iter().enumerate() {
        let boost = groups
            .get(i + 1)
            .filter(|next| next.len() == 1 && !barrier_before[i + 1])
            .and_then(|next| particle_boost(&next[0]))
            .unwrap_or(1.0);
        for token in group {
            let weight = if crate::stopwords::is_stopword(token) {
                0.0
            } else {
                boost
            };
            out.push(WeightedToken {
                token: token.clone(),
                weight,
            });
        }
    }

    // CL-CnG trigrams: same emission as `tokenize`, weight 0.0 (matching is
    // done by the filtered corpus, so they carry no score).
    let lowered = lowercase(&cased);
    let mut ngrams = Vec::new();
    push_char_ngrams(&lowered, &mut ngrams);
    out.extend(
        ngrams
            .into_iter()
            .map(|token| WeightedToken { token, weight: 0.0 }),
    );
    out
}

/// Like [`tokenize`] but generates word-level n-grams of the given size.
///
/// `n = 1` is equivalent to [`tokenize`]. For `n = 2`, adjacent word tokens
/// (excluding CL-CnG trigrams) are joined with a space to form bigrams; the
/// base unigrams are also included. For `n = 3`, both bigrams and trigrams are
/// emitted alongside unigrams.
pub fn tokenize_ngrams(text: &str, n: usize) -> Vec<String> {
    let mut out = tokenize(text);
    if n <= 1 {
        return out;
    }

    let word_tokens: Vec<String> = out
        .iter()
        .filter(|t| !t.starts_with(NGRAM_PREFIX))
        .cloned()
        .collect();

    for window_size in 2..=n {
        if word_tokens.len() >= window_size {
            for window in word_tokens.windows(window_size) {
                out.push(window.join(" "));
            }
        }
    }
    out
}

/// NFKC normalize and lowercase. Public so callers can compute a stable
/// canonical form (e.g. for content hashing) with the same normalization the
/// tokenizer uses.
pub fn normalize(text: &str) -> String {
    lowercase(&text.nfkc().collect::<String>())
}

fn lowercase(s: &str) -> String {
    s.chars().flat_map(|c| c.to_lowercase()).collect()
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ScriptClass {
    /// Space-delimited writing systems (Latin, Cyrillic, Greek, Arabic, …),
    /// plus digits and identifier connectors (`_`, `-`, `.`).
    Spacing,
    /// Non-spacing scripts that need n-gram segmentation (CJK, Hangul, Kana).
    NonSpacing,
    /// Whitespace, punctuation, symbols — dropped from word tokenization.
    Other,
}

struct Segment<'a> {
    class: ScriptClass,
    text: &'a str,
}

fn classify(c: char) -> ScriptClass {
    if is_non_spacing_script(c) {
        return ScriptClass::NonSpacing;
    }
    if c.is_alphabetic() || c.is_numeric() {
        // Letters and digits both belong to the word-token path. Keeping them in
        // one class means `p1`, `v0`, `utf8` stay as single identifiers.
        return ScriptClass::Spacing;
    }
    // Identifier connectors keep a run together (`atomic_write`, `feat-memory`,
    // `0.13.0`) without becoming standalone tokens; `push_identifier_subtokens`
    // splits on them. Everything else (whitespace, symbols) ends the run.
    if matches!(c, '_' | '-' | '.') {
        return ScriptClass::Spacing;
    }
    ScriptClass::Other
}

/// True for scripts written without spaces between words, where dictionary-free
/// recall is best served by character n-grams. Ranges per the Unicode blocks.
pub(crate) fn is_non_spacing_script(c: char) -> bool {
    matches!(c as u32,
        0x3040..=0x309F   // Hiragana
        | 0x30A0..=0x30FF // Katakana
        | 0x31F0..=0x31FF // Katakana Phonetic Extensions
        | 0x3400..=0x4DBF // CJK Unified Ideographs Extension A
        | 0x4E00..=0x9FFF // CJK Unified Ideographs
        | 0xF900..=0xFAFF // CJK Compatibility Ideographs
        | 0xAC00..=0xD7AF // Hangul Syllables
        | 0x1100..=0x11FF // Hangul Jamo
        | 0x20000..=0x2A6DF // CJK Extension B
        | 0x2A700..=0x2EBEF // CJK Extension C–F
    )
}

/// Split `text` into maximal runs of one script class.
fn script_segments(text: &str) -> Vec<Segment<'_>> {
    let mut segments = Vec::new();
    let mut start = 0usize;
    let mut current: Option<ScriptClass> = None;

    for (idx, c) in text.char_indices() {
        let class = classify(c);
        match current {
            Some(cur) if cur == class => {}
            Some(cur) => {
                segments.push(Segment {
                    class: cur,
                    text: &text[start..idx],
                });
                start = idx;
                current = Some(class);
            }
            None => {
                start = idx;
                current = Some(class);
            }
        }
    }
    if let Some(cur) = current {
        segments.push(Segment {
            class: cur,
            text: &text[start..],
        });
    }
    segments
}

/// UAX#29 word tokens for spacing scripts, plus identifier sub-tokens.
fn push_word_tokens(text: &str, out: &mut Vec<String>) {
    for word in text.unicode_words() {
        out.push(lowercase(word));
        push_identifier_subtokens(word, out);
    }
    // `unicode_words` drops pure-digit / underscore-joined runs in some cases;
    // also emit identifier sub-tokens split on `_`, `-`, and case transitions
    // for the raw segment so snake/kebab/camel identifiers are always covered.
    for raw in text.split(|c: char| c.is_whitespace()) {
        if raw.is_empty() {
            continue;
        }
        push_identifier_subtokens(raw, out);
    }
}

/// Emit sub-tokens of an identifier: split on `_`/`-`/`.` separators and on
/// lowercase→uppercase camelCase transitions. Only emits when the split yields
/// more than one piece (avoids duplicating plain words).
fn push_identifier_subtokens(ident: &str, out: &mut Vec<String>) {
    let mut pieces: Vec<String> = Vec::new();
    for part in ident.split(['_', '-', '.']) {
        if part.is_empty() {
            continue;
        }
        for camel in split_camel(part) {
            pieces.push(camel);
        }
    }
    if pieces.len() > 1 {
        for p in pieces {
            out.push(p);
        }
    }
}

/// Split a `camelCase` / `PascalCase` run into lowercase pieces. Returns the
/// input unchanged (as one piece) when there is no internal case transition.
fn split_camel(s: &str) -> Vec<String> {
    let chars: Vec<char> = s.chars().collect();
    let mut pieces = Vec::new();
    let mut buf = String::new();
    for (i, &c) in chars.iter().enumerate() {
        if i > 0 && c.is_uppercase() && chars[i - 1].is_lowercase() && !buf.is_empty() {
            pieces.push(std::mem::take(&mut buf));
        }
        buf.extend(c.to_lowercase());
    }
    if !buf.is_empty() {
        pieces.push(buf);
    }
    pieces
}

/// Cross-language character 3-grams over the whole normalized text. Whitespace
/// is collapsed to a single marker so n-grams don't span large gaps but word
/// boundaries still influence the grams. Each gram is namespaced by a prefix
/// char so it can't collide with a word token.
fn push_char_ngrams(normalized: &str, out: &mut Vec<String>) {
    let mut squashed = String::with_capacity(normalized.len());
    let mut prev_space = false;
    for c in normalized.chars() {
        if c.is_whitespace() {
            if !prev_space {
                squashed.push(' ');
                prev_space = true;
            }
            continue;
        }
        squashed.push(c);
        prev_space = false;
    }
    let chars: Vec<char> = squashed.trim().chars().collect();
    if chars.len() < CL_NGRAM {
        return;
    }
    for window in chars.windows(CL_NGRAM) {
        // Skip grams that are entirely separator/space — no signal.
        if window.iter().all(|c| *c == ' ') {
            continue;
        }
        let mut gram = String::with_capacity(CL_NGRAM + 1);
        gram.push(NGRAM_PREFIX);
        gram.extend(window.iter());
        out.push(gram);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn word_tokens(text: &str) -> Vec<String> {
        tokenize(text)
            .into_iter()
            .filter(|t| !t.starts_with(NGRAM_PREFIX))
            .collect()
    }

    #[test]
    fn english_words_lowercased() {
        let toks = word_tokens("Hello World FOO");
        assert!(toks.contains(&"hello".to_string()));
        assert!(toks.contains(&"world".to_string()));
        assert!(toks.contains(&"foo".to_string()));
    }

    #[test]
    fn snake_case_split_into_subtokens() {
        let toks = word_tokens("use atomic_write always");
        assert!(toks.contains(&"atomic".to_string()));
        assert!(toks.contains(&"write".to_string()));
        // and the joined form is preserved too (unicode_words keeps it together
        // since `_` is part of a word in UAX#29).
        assert!(toks.iter().any(|t| t.contains("atomic")));
    }

    #[test]
    fn kebab_case_split() {
        let toks = word_tokens("feat-memory-p1");
        assert!(toks.contains(&"feat".to_string()));
        assert!(toks.contains(&"memory".to_string()));
        assert!(toks.contains(&"p1".to_string()));
    }

    #[test]
    fn camel_case_split() {
        let toks = word_tokens("getMemoryQuery");
        assert!(toks.contains(&"get".to_string()));
        assert!(toks.contains(&"memory".to_string()));
        assert!(toks.contains(&"query".to_string()));
    }

    #[test]
    fn japanese_word_segmentation() {
        // "メモリ機能" is segmented into words ("メモリ", "機能") by the
        // learned boundary model, not fixed-length bigrams (spec §7.4).
        let toks = word_tokens("メモリ機能");
        assert!(toks.contains(&"メモリ".to_string()));
        assert!(toks.contains(&"機能".to_string()));
    }

    #[test]
    fn japanese_word_segmentation_memory_function() {
        // Spec §7.4 recommended test, asserted directly against tokenize():
        // "メモリ機能" → ["メモリ", "機能"] (order-preserving, no bigrams).
        let toks = word_tokens("メモリ機能");
        assert_eq!(toks, vec!["メモリ".to_string(), "機能".to_string()]);
    }

    #[test]
    fn single_cjk_char_is_unigram() {
        let toks = word_tokens("猫");
        assert!(toks.contains(&"猫".to_string()));
    }

    #[test]
    fn mixed_japanese_english() {
        // Japanese sentence with an embedded English identifier.
        let toks = word_tokens("atomic_write を使う");
        assert!(toks.contains(&"atomic".to_string()));
        assert!(toks.contains(&"write".to_string()));
        assert!(toks.contains(&"使う".to_string()) || toks.contains(&"を使う".to_string()));
    }

    #[test]
    fn nfkc_fullwidth_unified() {
        // Fullwidth ＡＢＣ should normalize to abc.
        let toks = word_tokens("ＡＢＣ");
        assert!(toks.contains(&"abc".to_string()));
    }

    #[test]
    fn cl_ngram_emitted_and_namespaced() {
        let all = tokenize("hello");
        let ngrams: Vec<_> = all.iter().filter(|t| t.starts_with(NGRAM_PREFIX)).collect();
        assert!(!ngrams.is_empty(), "expected CL-CnG trigrams");
        // "hello" → hel ell llo
        assert!(ngrams.iter().any(|g| g.ends_with("hel")));
        assert!(ngrams.iter().any(|g| g.ends_with("llo")));
    }

    #[test]
    fn ngram_does_not_collide_with_word() {
        // A 3-letter word "abc" and the trigram of "abc" must be distinct tokens.
        let toks = tokenize("abc");
        let word = toks.iter().filter(|t| *t == "abc").count();
        let gram = toks
            .iter()
            .filter(|t| t.starts_with(NGRAM_PREFIX) && t.ends_with("abc"))
            .count();
        assert_eq!(word, 1);
        assert_eq!(gram, 1);
    }

    #[test]
    fn empty_input_no_panic() {
        assert!(tokenize("").is_empty());
        assert!(tokenize("   ").is_empty());
    }

    #[test]
    fn digits_kept() {
        let toks = word_tokens("version 0.13.0");
        assert!(toks.iter().any(|t| t.contains("13") || t == "0"));
    }

    #[test]
    fn tokenize_ngrams_unigram_equals_tokenize() {
        let text = "hello world foo";
        assert_eq!(tokenize_ngrams(text, 1), tokenize(text));
    }

    #[test]
    fn tokenize_ngrams_bigrams() {
        let toks = tokenize_ngrams("hello world foo", 2);
        assert!(toks.contains(&"hello world".to_string()));
        assert!(toks.contains(&"world foo".to_string()));
        // unigrams still present
        assert!(toks.contains(&"hello".to_string()));
    }

    #[test]
    fn tokenize_ngrams_trigrams() {
        let toks = tokenize_ngrams("hello world foo bar", 3);
        assert!(toks.contains(&"hello world foo".to_string()));
        assert!(toks.contains(&"world foo bar".to_string()));
        // bigrams also present
        assert!(toks.contains(&"hello world".to_string()));
    }

    #[test]
    fn tokenize_ngrams_empty() {
        assert!(tokenize_ngrams("", 2).is_empty());
    }

    #[test]
    fn tokenize_ngrams_single_word() {
        let toks = tokenize_ngrams("hello", 2);
        assert!(toks.contains(&"hello".to_string()));
        // no bigrams possible from a single word token
        assert!(!toks.iter().any(|t| t.contains(' ')));
    }

    /// Weight of `token` in the weighted tokenization of `text`. Panics when
    /// the token is absent; when a token appears more than once the weights
    /// must all agree (a token's role is per-occurrence, but these fixtures
    /// only use unambiguous sentences).
    fn weight_of(text: &str, token: &str) -> f64 {
        let weighted = tokenize_weighted(text);
        let hits: Vec<f64> = weighted
            .iter()
            .filter(|wt| wt.token == token)
            .map(|wt| wt.weight)
            .collect();
        assert!(!hits.is_empty(), "token {token:?} not found in {text:?}");
        for w in &hits {
            assert_eq!(*w, hits[0], "inconsistent weights for {token:?}: {hits:?}");
        }
        hits[0]
    }

    #[test]
    fn weighted_topic_particle_boosts_preceding_word() {
        // `atomic_write は` — the topic particle marks atomic_write as the
        // subject; the full identifier AND its sub-tokens get the boost so a
        // sub-token query match benefits too.
        assert_eq!(
            weight_of("atomic_write は重要", "atomic_write"),
            TOPIC_BOOST
        );
        assert_eq!(weight_of("atomic_write は重要", "atomic"), TOPIC_BOOST);
        assert_eq!(weight_of("atomic_write は重要", "write"), TOPIC_BOOST);
        // The word after the particle is a plain content word.
        assert_eq!(weight_of("atomic_write は重要", "重要"), 1.0);
    }

    #[test]
    fn weighted_ga_is_topic_boost() {
        assert_eq!(weight_of("メモリが不足する", "メモリ"), TOPIC_BOOST);
    }

    #[test]
    fn weighted_object_particle_boost() {
        assert_eq!(weight_of("ファイルを書く", "ファイル"), OBJECT_BOOST);
        assert_eq!(weight_of("ファイルを書く", "書く"), 1.0);
    }

    #[test]
    fn weighted_case_particle_boost() {
        // で / に mark means/target — mid-tier boost.
        assert_eq!(weight_of("sshで接続する", "ssh"), CASE_BOOST);
        assert_eq!(weight_of("サーバーに送る", "サーバー"), CASE_BOOST);
    }

    #[test]
    fn weighted_plain_content_word_is_default() {
        // No trailing particle → default weight.
        assert_eq!(weight_of("使う", "使う"), 1.0);
    }

    #[test]
    fn weighted_stopwords_are_zero() {
        let weighted = tokenize_weighted("メモリ機能はセッション間で教訓を引き継ぐ");
        for wt in &weighted {
            if crate::stopwords::is_stopword(&wt.token) {
                assert_eq!(wt.weight, 0.0, "stopword {:?} must be 0.0", wt.token);
            }
        }
        // and the particles really are present as zero-weight tokens
        assert_eq!(
            weight_of("メモリ機能はセッション間で教訓を引き継ぐ", "は"),
            0.0
        );
        assert_eq!(
            weight_of("メモリ機能はセッション間で教訓を引き継ぐ", "を"),
            0.0
        );
    }

    #[test]
    fn weighted_cl_ngrams_are_zero() {
        let weighted = tokenize_weighted("hello メモリ");
        let ngrams: Vec<_> = weighted
            .iter()
            .filter(|wt| is_cl_ngram(&wt.token))
            .collect();
        assert!(!ngrams.is_empty(), "CL-CnG trigrams must still be emitted");
        for wt in ngrams {
            assert_eq!(wt.weight, 0.0, "CL-CnG {:?} must be 0.0", wt.token);
        }
    }

    #[test]
    fn weighted_boost_does_not_cross_stopword_group() {
        // `Xの設定は` — は boosts 設定 (the group immediately before it), and
        // must NOT skip back over it to boost anything earlier.
        assert_eq!(weight_of("メモリの設定は重要", "設定"), TOPIC_BOOST);
        assert_eq!(weight_of("メモリの設定は重要", "メモリ"), 1.0);
    }

    #[test]
    fn weighted_boost_blocked_by_sentence_boundary() {
        // Adversarial-review finding: Other segments are dropped before
        // adjacency is computed, so without a barrier the topic particle of
        // the NEXT sentence boosted the last word of the previous one.
        assert_eq!(weight_of("atomic_write。は重要", "atomic_write"), 1.0);
        assert_eq!(weight_of("重要。から始める", "重要"), 1.0);
        // Newlines separate list items / lines the same way.
        assert_eq!(weight_of("atomic_write\nは重要", "atomic_write"), 1.0);
    }

    #[test]
    fn weighted_boost_passes_through_quotes_and_brackets() {
        // Quotes/brackets around the word must stay transparent: the particle
        // still marks the quoted word as topic.
        assert_eq!(weight_of("「メモリ」は重要", "メモリ"), TOPIC_BOOST);
        assert_eq!(weight_of("(ファイル)を書く", "ファイル"), OBJECT_BOOST);
        // Plain whitespace (an Other segment) stays transparent too.
        assert_eq!(
            weight_of("atomic_write は重要", "atomic_write"),
            TOPIC_BOOST
        );
    }

    #[test]
    fn weighted_token_multiset_matches_tokenize() {
        // tokenize_weighted must not invent or drop tokens: same multiset as
        // tokenize() (order across a spacing chunk may differ).
        for text in [
            "atomic_write は重要",
            "メモリ機能はセッション間で教訓を引き継ぐ",
            "use atomic_write always",
            "getMemoryQuery を呼ぶ",
            "atomic_write。は重要",
            "「メモリ」は重要！\n次の行",
        ] {
            let mut plain = tokenize(text);
            let mut weighted: Vec<String> = tokenize_weighted(text)
                .into_iter()
                .map(|wt| wt.token)
                .collect();
            plain.sort();
            weighted.sort();
            assert_eq!(plain, weighted, "token multiset diverged for {text:?}");
        }
    }

    #[test]
    fn weighted_empty_input() {
        assert!(tokenize_weighted("").is_empty());
        assert!(tokenize_weighted("   ").is_empty());
    }
}
