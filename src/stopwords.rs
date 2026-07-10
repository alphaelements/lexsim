//! Japanese stopword filtering for the extraction stage (TF-IDF keywords /
//! corpus diff / co-occurrence / TextRank). This is **not** applied inside
//! [`crate::tokenize::tokenize`] itself — the tokenizer's BM25
//! term-frequency contract must stay untouched, so filtering only happens
//! where keywords are surfaced to a human.
//!
//! Japanese runs are now segmented into real words by a trained boundary
//! model (see `segmenter/`), so most particles and auxiliaries already
//! appear as clean standalone tokens (`は`, `の`, `です`, `ました`, ...) and
//! are caught by a plain exact-match lookup against [`JA_STOPWORDS`] — which
//! now also lists whole-word conjunctions/adverbs (`しかし`, `つまり`, ...)
//! and demonstratives (`これ`, `その`, ...) the segmenter emits as single
//! tokens.
//!
//! A **secondary bigram-glue heuristic** is kept for the CJK bigram fallback
//! path (see `segmenter/inference.rs`'s `push_bigrams`), which only fires for
//! non-Japanese-script non-spacing scripts (Hangul, unclassified CJK-adjacent
//! blocks, etc.) — a run of hiragana / katakana / kanji is always routed
//! through the trained boundary segmenter instead, never through the bigram
//! fallback. So the only bigrams this heuristic actually needs to catch are
//! ones that mix a Japanese function character with *any* non-Japanese-script
//! character — Hangul, Latin, digits, punctuation, not just Hangul (`は<hangul>`,
//! `Aた`, `の。`, ...) — a bigram of two Japanese-script characters is never
//! produced by the fallback path in the first place, so [`is_stopword`] does
//! not fire on one. This also sidesteps the false-positive class that
//! motivated this restriction: two-character *Japanese-script* content words
//! (`はし` 橋/箸, `にわ` 庭, `すし` 寿司, ...) are never touched by the
//! heuristic, because both of their characters are Japanese-script.
//!
//! "Japanese-script" here means exactly what
//! [`crate::segmenter::inference::is_japanese_run_char`] (and transitively
//! `segmenter::features::classify_char`) considers Japanese for segmentation
//! purposes — this module calls that same function rather than
//! re-implementing the Unicode ranges, so the two can never drift apart.
//!
//! `is_stopword` is also `pub` and may be called directly by downstream
//! crates on arbitrary strings, not just crate-internal tokenizer output — a
//! Japanese-script-only bigram like `はし` must stay safe there too, which is
//! exactly what this restriction guarantees regardless of caller.
//!
//! Reported by x-metrics referral ref-20260710-012108-589608700: the
//! `JA_SINGLE_CHAR_FUNCTION_CHARS` bigram heuristic was previously
//! unconditional and flagged Japanese-script content words like `はし` as
//! stopwords.
//!
//! **Known limitation**: for the bigram fallback path, this only closes the
//! *edge* of a multi-character auxiliary sequence (the bigram touching
//! `た`/`だ`/`す`, e.g. a fallback bigram touching the tail of `ました`).
//! *Interior* bigrams of a multi-character auxiliary that don't touch any
//! single-char stopword on either side are not caught by this heuristic and
//! would need a different approach (matching against known multi-character
//! auxiliary sequences rather than single characters) to close fully.

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
    // Conjunctions/adverbs (接続詞・副詞): the trained boundary segmenter now
    // emits these as standalone word tokens (they used to only show up glued
    // inside CJK bigrams), so they need an exact-match entry to be filtered.
    "また",
    "そして",
    "しかし",
    "ただし",
    "つまり",
    "なお",
    "ただ",
    "さらに",
    "および",
    "または",
    // Demonstratives (指示語)
    "この",
    "その",
    "あの",
    "どの",
    "これ",
    "それ",
    "あれ",
    "ここ",
    "そこ",
    // Generic time/manner adverbs (too generic to carry topical signal)
    "とても",
    "かなり",
    "すべて",
    "すぐ",
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
    'た', 'だ', 'す',
];

/// Returns `true` if `token` is a Japanese stopword (functional word with
/// little topical signal). Intended for use at the **extraction** stage
/// (`tfidf_keywords`, `corpus_diff`), not inside the tokenizer itself.
///
/// Two checks are combined:
/// 1. Exact match against [`JA_STOPWORDS`] (catches lone trailing particles
///    and multi-character auxiliaries that exactly fill a token).
/// 2. For a two-character token that is *not* entirely Japanese-script
///    (hiragana/katakana/kanji, per
///    [`crate::segmenter::inference::is_japanese_run_char`]), either
///    character being a single-character particle or auxiliary fragment
///    (see [`JA_SINGLE_CHAR_FUNCTION_CHARS`]) — catches the CJK bigram
///    fallback path gluing a particle/auxiliary fragment to a
///    non-Japanese-script character (Hangul, Latin, digits, punctuation,
///    ...). A bigram of two Japanese-script characters is always produced by
///    the trained segmenter, never the bigram fallback, so it is never
///    treated as a stopword by this heuristic — see module docs.
pub fn is_stopword(token: &str) -> bool {
    static STOPWORDS: std::sync::LazyLock<HashSet<&'static str>> =
        std::sync::LazyLock::new(|| JA_STOPWORDS.iter().copied().collect());
    if STOPWORDS.contains(token) {
        return true;
    }

    let chars: Vec<char> = token.chars().collect();
    if chars.len() == 2 {
        if chars
            .iter()
            .all(|&c| crate::segmenter::inference::is_japanese_run_char(c))
        {
            return false;
        }
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
    fn particle_glued_to_non_japanese_script_bigrams_are_stopwords() {
        // The bigram fallback path only ever fires when a Japanese-script
        // sub-run is a single character, or on non-Japanese-script runs
        // (Hangul, etc.) — see `segmenter::inference::push_bigrams`. So the
        // only bigrams the heuristic actually needs to catch mix a Japanese
        // function character with a non-Japanese-script character.
        assert!(is_stopword("は가")); // は + Hangul 가
        assert!(is_stopword("나の")); // Hangul 나 + の
        assert!(is_stopword("を나")); // を + Hangul 나
    }

    #[test]
    fn content_bigrams_without_particles_are_not_stopwords() {
        assert!(!is_stopword("機能"));
        assert!(!is_stopword("メモ"));
        assert!(!is_stopword("教訓"));
    }

    #[test]
    fn aux_glued_to_non_japanese_script_bigrams_are_stopwords() {
        // Same as particle_glued_to_non_japanese_script_bigrams_are_stopwords
        // but for the single-character auxiliary fragments (た/だ/す).
        assert!(is_stopword("た가")); // auxiliary た + Hangul 가
        assert!(is_stopword("나だ")); // Hangul 나 + auxiliary だ
        assert!(is_stopword("す가")); // auxiliary す + Hangul 가
        assert!(is_stopword("だっ")); // already an exact JA_STOPWORDS entry
    }

    #[test]
    fn japanese_script_only_bigrams_are_never_flagged_by_the_heuristic() {
        // Two-character tokens where both characters are Japanese-script
        // (hiragana/katakana/kanji) are always produced by the trained
        // boundary segmenter, never the bigram fallback path (which only
        // fires on non-Japanese-script runs) — so the bigram heuristic must
        // never flag them, even when a character happens to also be a
        // single-character particle/auxiliary fragment. Reported by
        // x-metrics referral ref-20260710-012108-589608700 (e.g. "はし"
        // 橋/箸 was being incorrectly filtered out as a stopword).
        assert!(!is_stopword("はし")); // 橋・箸 (bridge/chopsticks) — starts with は
        assert!(!is_stopword("にわ")); // 庭 (garden) — starts with に
        assert!(!is_stopword("かに")); // 蟹 (crab) — ends with に
        assert!(!is_stopword("なわ")); // 縄 (rope) — starts with な
        assert!(!is_stopword("すし")); // 寿司 (sushi) — starts with す
        assert!(!is_stopword("いす")); // 椅子 (chair) — ends with す
        assert!(!is_stopword("たこ")); // 蛸・凧 (octopus/kite) — starts with た
        assert!(!is_stopword("のり")); // 海苔・糊 (seaweed/glue) — starts with の
        assert!(!is_stopword("よる")); // 夜 (night) — starts with よ
    }

    #[test]
    fn is_stopword_uses_the_segmenters_japanese_script_definition() {
        // is_stopword must agree with is_japanese_run_char on every script
        // block it considers Japanese, including the less common ones
        // (Katakana Phonetic Extensions, CJK Compatibility Ideographs) — a
        // hand-duplicated Unicode range table here would risk drifting from
        // the segmenter's actual definition and reintroducing false
        // positives on those blocks. Calling
        // `segmenter::inference::is_japanese_run_char` directly (rather than
        // re-implementing the ranges) is what this test guards.
        assert!(!is_stopword("すㇰ")); // す + Katakana Phonetic Extension (U+31F0)
        assert!(!is_stopword("は\u{F900}")); // は + CJK Compatibility Ideograph (U+F900 豈)
    }

    #[test]
    fn stopword_list_size_is_in_guideline_range() {
        // Guideline: 50〜110 words (deduplicated). The trained boundary
        // segmenter now emits real word tokens (conjunctions, demonstratives,
        // generic adverbs) that previously only ever appeared glued inside
        // bigrams, so the word-level list grew accordingly.
        let unique: HashSet<&str> = JA_STOPWORDS.iter().copied().collect();
        assert!(
            unique.len() >= 50 && unique.len() <= 110,
            "stopword list has {} unique entries, expected 50..=110",
            unique.len()
        );
    }

    #[test]
    fn conjunctions_and_adverbs_are_stopwords() {
        // The segmenter now emits these as standalone word tokens; they must
        // be filtered at the extraction stage (little topical signal).
        for w in [
            "また",
            "そして",
            "しかし",
            "ただし",
            "つまり",
            "なお",
            "ただ",
            "さらに",
            "および",
            "または",
        ] {
            assert!(is_stopword(w), "{w:?} should be a stopword");
        }
    }

    #[test]
    fn demonstratives_are_stopwords() {
        for w in [
            "この", "その", "あの", "どの", "これ", "それ", "あれ", "ここ", "そこ",
        ] {
            assert!(is_stopword(w), "{w:?} should be a stopword");
        }
    }

    #[test]
    fn generic_adverbs_are_stopwords() {
        for w in ["とても", "かなり", "すべて", "すぐ"] {
            assert!(is_stopword(w), "{w:?} should be a stopword");
        }
    }
}
