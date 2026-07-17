//! Cheap token-count estimation for token budgeting (e.g. handoff-mcp's
//! staged document injection: `doc_inline_threshold` / `doc_query_max_tokens`).
//!
//! This is a heuristic, not a real model tokenizer: it does not depend on any
//! specific BPE/SentencePiece vocabulary, so it stays fast and dependency-free
//! at the cost of exactness. It is good enough to decide "does this fragment
//! fit the budget", not to bill API usage.

use crate::tokenize::is_non_spacing_script;

/// Estimate the number of model tokens `text` would consume.
///
/// Heuristic: `ascii_chars / 4 + cjk_chars / 1.5`, integer division only
/// (`usize`, no floating point). The `/ 1.5` division is implemented as
/// `cjk_chars * 2 / 3` (`x / 1.5 == x * 2 / 3`) to stay in `usize` arithmetic
/// without precision loss from repeated float rounding.
///
/// Character classes:
/// - **ASCII** (`char::is_ascii`): counted at ~4 chars/token, matching common
///   English/code BPE tokenizers.
/// - **CJK** (Hiragana / Katakana / CJK Unified Ideographs / Hangul — the same
///   non-spacing-script ranges [`tokenize`](crate::tokenize) already uses to
///   route text through the Japanese segmenter): counted at ~1.5 chars/token,
///   since most CJK tokenizers emit roughly one token per 1-2 characters.
/// - **Everything else** (non-ASCII Latin/Cyrillic/Greek diacritics, emoji,
///   other symbols, combining marks, whitespace, punctuation): counted like
///   ASCII (`/ 4`). These scripts are still space- or codepoint-delimited
///   rather than dense like CJK, so the ASCII ratio is the closer of the two
///   available buckets; treating them as free (0 cost) would let adversarial
///   or simply non-English/non-CJK content evade the token budget entirely.
///
/// The result is additive per class and always rounds down, so this function
/// is a *lower-bound-leaning* estimate — safe to use as a fast pre-filter
/// before a budget cutoff, not as an exact billing figure.
pub fn estimate_tokens(text: &str) -> usize {
    let mut ascii_chars = 0usize;
    let mut cjk_chars = 0usize;

    for c in text.chars() {
        if is_non_spacing_script(c) {
            cjk_chars += 1;
        } else {
            // ASCII and all other scripts share the same divisor (see doc
            // comment: "Everything else").
            ascii_chars += 1;
        }
    }

    ascii_chars / 4 + cjk_chars * 2 / 3
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_string_is_zero_tokens() {
        assert_eq!(estimate_tokens(""), 0);
    }

    #[test]
    fn pure_ascii_uses_four_chars_per_token() {
        // 12 ascii chars / 4 = 3
        assert_eq!(estimate_tokens("hello world!"), 3);
    }

    #[test]
    fn pure_cjk_uses_one_point_five_chars_per_token() {
        // 6 CJK chars * 2 / 3 = 4
        assert_eq!(estimate_tokens("メモリ機能です"), 4);
    }

    #[test]
    fn mixed_japanese_and_english_sums_both_buckets() {
        // "atomic_write" = 12 ascii chars -> 12/4 = 3
        // "を必ず使う" = 5 CJK chars -> 5*2/3 = 3 (integer division)
        let text = "atomic_writeを必ず使う";
        let expected = "atomic_write".chars().count() / 4 + "を必ず使う".chars().count() * 2 / 3;
        assert_eq!(estimate_tokens(text), expected);
        assert_eq!(estimate_tokens(text), 6);
    }

    #[test]
    fn emoji_and_symbols_fall_back_to_ascii_ratio() {
        // Emoji are neither ASCII nor in the CJK/Kana/Hangul ranges, so they
        // fall into the "everything else" bucket and are counted at /4.
        // 4 emoji chars / 4 = 1
        assert_eq!(estimate_tokens("🎉🎊🎈🎁"), 1);
    }

    #[test]
    fn never_negative_and_monotonic_with_length() {
        let short = estimate_tokens("a");
        let long = estimate_tokens(&"a".repeat(100));
        assert!(long >= short);
    }
}
