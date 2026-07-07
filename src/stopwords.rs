//! Japanese stopword filtering for the extraction stage (TF-IDF keywords /
//! corpus diff). This is **not** applied inside [`crate::tokenize::tokenize`]
//! itself — the tokenizer's BM25 term-frequency contract must stay untouched,
//! so filtering only happens where keywords are surfaced to a human.
//!
//! Because the tokenizer emits CJK text as character bi-grams (see
//! `tokenize.rs`), single-character function words (助詞 like `の`/`に`/`は`,
//! and single-character 助動詞 fragments like `た`/`だ`) rarely appear as their
//! own token: an unbroken Japanese sentence tokenizes into one continuous run
//! of overlapping bigrams with no trailing unigram at all, so a particle or
//! auxiliary fragment only ever shows up glued to an adjacent content
//! character (`は便`, `の機`, `能は`, `した`, ...). An exact-match lookup against
//! the whole token therefore only catches two cases: a lone trailing function
//! word in an odd-length run, and a multi-character auxiliary sequence
//! (`です`/`ます`/`ない`/...) that exactly fills a bigram. To also catch the
//! common "function word glued to a bigram" case, [`is_stopword`] additionally
//! treats a two-character token as a stopword when either of its two
//! characters is a single-character particle or auxiliary — the same
//! characters already listed as ADP/AUX entries in [`JA_STOPWORDS`].
//!
//! **Known limitation**: this only closes the *edge* of a multi-character
//! auxiliary sequence (the bigram touching `た`/`だ`, e.g. `した` from
//! `ました`). *Interior* bigrams of a multi-character auxiliary that don't
//! touch any single-char stopword on either side (e.g. `まし`/`りま`, also
//! from `ました`) are not caught by this heuristic and would need a
//! different approach (matching against known multi-character auxiliary
//! sequences rather than single characters) to close fully.

use std::collections::HashSet;

/// Japanese stopwords: functional words (助詞/助動詞/等位接続詞/限定詞 and
/// other grammatical function words) that carry little topical signal for
/// keyword extraction. Grouped by rough UPOS category for maintainability.
const JA_STOPWORDS: &[&str] = &[
    // ADP (助詞: 格助詞・係助詞・副助詞・終助詞など)
    "の",
    "に",
    "は",
    "を",
    "が",
    "で",
    "と",
    "も",
    "へ",
    "から",
    "まで",
    "より",
    "だけ",
    "ほど",
    "ばかり",
    "しか",
    "さえ",
    "すら",
    "でも",
    "か",
    "ね",
    "よ",
    "な",
    "わ",
    "ぞ",
    "ぜ",
    "とも",
    "けど",
    "けれど",
    "けれども",
    "ながら",
    "たり",
    "のに",
    "ので",
    "ては",
    "では",
    // AUX (助動詞)
    "です",
    "ます",
    "ました",
    "でした",
    "ません",
    "ない",
    "なかっ",
    "た",
    "だ",
    "だっ",
    "れる",
    "られる",
    "せる",
    "させる",
    "よう",
    "らしい",
    "そう",
    "だろう",
    "でしょう",
    // Other function words (軽動詞・形式名詞など)
    "する",
    "し",
    "さ",
    "され",
    "でき",
    "なる",
    "なり",
    "ある",
    "あり",
    "いる",
    "い",
    "おり",
    "おる",
    "こと",
    "もの",
    "ため",
    "ところ",
    "ほう",
    "わけ",
];

/// Single-character particles (係助詞・格助詞・終助詞) and single-character
/// auxiliary fragments (助動詞: `た`/`だ`) that the CJK bigram tokenizer almost
/// never emits as a standalone token — they show up glued to an adjacent
/// content or auxiliary character instead (`は便`, `の機`, `能は`, `した`,
/// `まし`, `りま`, ...). A two-character token containing one of these as
/// either character is treated as a stopword by [`is_stopword`] (see module
/// docs for the rationale).
const JA_SINGLE_CHAR_FUNCTION_CHARS: &[char] = &[
    // ADP (助詞)
    'の', 'に', 'は', 'を', 'が', 'で', 'と', 'も', 'へ', 'か', 'ね', 'よ', 'な', 'わ', 'ぞ', 'ぜ',
    // AUX (助動詞): 単独では滅多に出ないが、隣接文字と結合したbigramに現れる
    'た', 'だ',
];

/// Returns `true` if `token` is a Japanese stopword (functional word with
/// little topical signal). Intended for use at the **extraction** stage
/// (`tfidf_keywords`, `corpus_diff`), not inside the tokenizer itself.
///
/// Two checks are combined:
/// 1. Exact match against [`JA_STOPWORDS`] (catches lone trailing particles
///    and multi-character auxiliaries that exactly fill a token).
/// 2. For a two-character token, either character being a single-character
///    particle or auxiliary fragment (see [`JA_SINGLE_CHAR_FUNCTION_CHARS`])
///    — catches the common case where the CJK bigram tokenizer glues a
///    particle or auxiliary fragment to a content/auxiliary character.
pub fn is_stopword(token: &str) -> bool {
    static STOPWORDS: std::sync::LazyLock<HashSet<&'static str>> =
        std::sync::LazyLock::new(|| JA_STOPWORDS.iter().copied().collect());
    if STOPWORDS.contains(token) {
        return true;
    }

    let chars: Vec<char> = token.chars().collect();
    if chars.len() == 2 {
        return chars
            .iter()
            .any(|c| JA_SINGLE_CHAR_FUNCTION_CHARS.contains(c));
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn particles_are_stopwords() {
        assert!(is_stopword("の"));
        assert!(is_stopword("に"));
        assert!(is_stopword("は"));
        assert!(is_stopword("を"));
        assert!(is_stopword("が"));
    }

    #[test]
    fn auxiliaries_are_stopwords() {
        assert!(is_stopword("です"));
        assert!(is_stopword("ます"));
        assert!(is_stopword("ました"));
        assert!(is_stopword("ない"));
    }

    #[test]
    fn content_words_are_not_stopwords() {
        assert!(!is_stopword("メモリ"));
        assert!(!is_stopword("機能"));
        assert!(!is_stopword("学習"));
        assert!(!is_stopword("犬"));
    }

    #[test]
    fn empty_and_english_are_not_stopwords() {
        assert!(!is_stopword(""));
        assert!(!is_stopword("hello"));
    }

    #[test]
    fn particle_glued_bigrams_are_stopwords() {
        // The CJK bigram tokenizer glues a single-character particle to an
        // adjacent content character in unbroken sentences (no trailing
        // unigram to catch); is_stopword must still flag these bigrams.
        assert!(is_stopword("は便")); // trailing 便 + leading は
        assert!(is_stopword("の機")); // trailing 機 + leading の
        assert!(is_stopword("能は")); // 能 + trailing は
        assert!(is_stopword("を引")); // を + 引
    }

    #[test]
    fn content_bigrams_without_particles_are_not_stopwords() {
        assert!(!is_stopword("機能"));
        assert!(!is_stopword("メモ"));
        assert!(!is_stopword("教訓"));
    }

    #[test]
    fn aux_glued_bigrams_are_stopwords() {
        // The CJK bigram tokenizer glues a single-character auxiliary (た/だ)
        // to an adjacent content character just like it does for particles;
        // is_stopword must flag these auxiliary-fragment bigrams too, or
        // fragments like "した" (from 降りました) leak into keyword output.
        assert!(is_stopword("した")); // し + た (auxiliary た)
        assert!(is_stopword("んだ")); // ん + だ (auxiliary だ)
        assert!(is_stopword("だっ")); // already an exact JA_STOPWORDS entry
    }

    #[test]
    fn stopword_list_size_is_in_guideline_range() {
        // Guideline: 50〜80 words (deduplicated).
        let unique: HashSet<&str> = JA_STOPWORDS.iter().copied().collect();
        assert!(
            unique.len() >= 50 && unique.len() <= 80,
            "stopword list has {} unique entries, expected 50..=80",
            unique.len()
        );
    }
}
