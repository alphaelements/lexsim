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
    // Coordinating / exemplifying particles the segmenter emits standalone.
    "や",
    "など",
    "とか",
    // AUX (助動詞)
    "です",
    "ます",
    "ました",
    "でした",
    "ません",
    "ない",
    "なかっ",
    // 否定・様態の連用形。`問題 なく 動作`, `仕方 なく 実行`, `時間 が なく なった`
    // のように独立トークンで現れる（コーパス上 6 例）。漢字表記の `無く…` と
    // 形容詞連用形の `少なく` は別トークンなので内容語は失われない。
    "なく",
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
    // 口語の文末 `だろ`（`だろう` の縮約）。日本語ランは記号・ASCII・絵文字・
    // 文末で途切れるため、`これはバグだろ！` => [これ, は, バグ, だろ],
    // `そうだろ？` => [そう, だろ], `無理だろw` => [無理, だろ, w] のように
    // 独立トークンで現れる（x-metrics 実測: 口語ツイート 12 件中 9 件で
    // キーワード出力に漏出）。`だろ` と一致する内容語は存在しない。
    "だろ",
    "でしょう",
    // Single-character auxiliary the segmenter emits when it mis-splits `です`
    // into `で` + `す` (see x-metrics referral ref-20260710-060424-851178046).
    // `JA_SINGLE_CHAR_FUNCTION_CHARS` is a two-character-token heuristic and
    // never fires for a lone `す`, so it must be listed here.
    "す",
    "かも",
    // Other function words (軽動詞・形式名詞など)
    "する",
    "し",
    // サ変名詞 + `する` is still split by the segmenter (`実施` + `した`), so
    // these auxiliary fragments surface as standalone word tokens. `した` is
    // the 10th most frequent token in the training corpus (207 occurrences).
    "した",
    "して",
    "さ",
    "され",
    "でき",
    "なる",
    "なり",
    // 動詞「因る」の連体形。コーパス上、独立トークンとして現れる「よる」10 例は
    // すべて `に よる` の機能語用法。名詞の「夜」は漢字表記でのみ現れる（独立
    // トークン 6 例）ため、ひらがな「よる」を落としても内容語は失われない。
    // 方言の進行相（`落ちよる` `鳴りよる` 等 4 例）は動詞語幹に膠着した 3 文字
    // 以上のトークンなので、この 2 文字 exact-match には掛からない。
    "よる",
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
    "ぜひ",
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
                                       // `よる` used to be the `よ`-initial witness here, but it is now an
                                       // exact `JA_STOPWORDS` entry (`〜による`) and would short-circuit on the
                                       // exact match before ever reaching the heuristic. `よこ` restores that
                                       // coverage: `よ` *is* in `JA_SINGLE_CHAR_FUNCTION_CHARS`, so this
                                       // asserts the all-Japanese-script early return wins over the heuristic.
        assert!(!is_stopword("よこ")); // 横 (side) — starts with よ
        assert!(!is_stopword("ゆり")); // 百合 (lily) — neither char is a function char
    }

    #[test]
    fn single_character_function_words_are_stopwords() {
        // `JA_SINGLE_CHAR_FUNCTION_CHARS` is a *two-character-token* heuristic
        // and is never consulted for a one-character token, so single-char
        // function words must be listed in `JA_STOPWORDS` to be filtered.
        // Reported by x-metrics referral ref-20260710-060424-851178046: the
        // segmenter emits a lone `す` (from `です` split as `で` + `す`) and a
        // lone `や` (the coordinating particle in `猫や犬`).
        assert!(is_stopword("す"));
        assert!(is_stopword("や"));
    }

    #[test]
    fn single_char_stopwords_do_not_leak_into_two_char_content_words() {
        // Adding `す` to `JA_STOPWORDS` must not make two-character content
        // words containing it stopwords: the exact-match lookup is
        // whole-token, and the bigram heuristic never fires on an
        // all-Japanese-script bigram.
        assert!(!is_stopword("すし")); // 寿司 (sushi)
        assert!(!is_stopword("いす")); // 椅子 (chair)
        assert!(!is_stopword("やま")); // 山 (mountain) — starts with や
        assert!(!is_stopword("つや")); // 艶 (gloss) — ends with や
    }

    #[test]
    fn conjugation_fragments_the_segmenter_actually_emits_are_stopwords() {
        // The trained segmenter keeps inflected verb forms as single tokens
        // (commit 846c779), but a サ変 noun + `する` conjugation is still split
        // (`実施` + `した`, `変更` + `して`), leaving these auxiliary fragments
        // as standalone word tokens. They carry no topical signal.
        //
        // `した` is the 10th most frequent token in `training/seed_corpus.txt`
        // (207 occurrences) — more frequent than `する` (141) and `して` (96),
        // both of which were already listed.
        //
        // `なく` is the negative/adverbial form the segmenter splits off in
        // `問題 なく 動作`, `仕方 なく 実行`, `時間 が なく なった` (6 standalone
        // occurrences in the corpus). 0.5.0 wrongly assumed it only ever merged
        // into `なくなった` and left it out.
        for w in [
            "した", "して", "よる", "など", "とか", "ぜひ", "かも", "なく",
        ] {
            assert!(is_stopword(w), "{w} should be a stopword");
        }
    }

    #[test]
    fn naku_as_a_stopword_does_not_swallow_content_words() {
        // `なく` is only a stopword as an exact whole token. The adjectival
        // `少なく` and the kanji-spelled `無く…` are separate tokens and must
        // survive keyword extraction.
        assert!(!is_stopword("少なく"));
        assert!(!is_stopword("無く"));
        assert!(!is_stopword("無くした"));
    }

    #[test]
    fn daro_is_a_stopword() {
        // Sentence-final colloquial `だろ` (plain-form conjecture, the
        // truncated `だろう`) is emitted as a standalone token whenever the
        // Japanese run ends at punctuation / ASCII / emoji / end-of-text:
        // `これはバグだろ！` => [これ, は, バグ, だろ], `そうだろ？` =>
        // [そう, だろ], `無理だろw` => [無理, だろ, w]. Reported by x-metrics
        // (leaks in 9/12 natural colloquial tweets); prose-only probes miss
        // this because prose rarely ends a run right after `だろ`.
        assert!(is_stopword("だろ"));
        // The full form the segmenter emits mid-prose stays covered too.
        assert!(is_stopword("だろう"));
    }

    #[test]
    fn daro_as_a_stopword_does_not_swallow_content_words() {
        // `だろ` is only a stopword as an exact whole token. No Japanese
        // content word is spelled exactly `だろ`, and nearby two-character
        // Japanese-script content words are untouched (exact match +
        // all-Japanese-script bigrams never reach the heuristic).
        assert!(!is_stopword("ころ")); // 頃 (time/around)
        assert!(!is_stopword("どろ")); // 泥 (mud)
        assert!(!is_stopword("だるま")); // 達磨 (daruma)
    }

    #[test]
    fn merged_conjugations_are_not_expected_as_standalone_tokens() {
        // x-metrics' referral also asked for `でし` / `なっ` / `たい` / `んだ`.
        // Mid-prose the segmenter merges those into `でした` (or splits them
        // as `で` + `した`), `なった`, `試したい`, `読んだ`, so they are
        // deliberately *not* added to `JA_STOPWORDS` (yet). This test
        // documents that decision.
        //
        // (`なく` was in this list until 0.5.1; it *does* appear standalone —
        // see `naku_as_a_stopword_does_not_swallow_content_words`.)
        for w in ["でし", "なっ", "たい", "んだ"] {
            assert!(!is_stopword(w), "{w} should not be a stopword");
        }
        // NOTE: this "never emitted standalone" assumption is *per word*, not
        // a general rule — `なく` (0.5.1) and `だろ` (see `daro_is_a_stopword`)
        // both turned out to be emitted standalone at run boundaries
        // (punctuation/ASCII/emoji/end-of-text) and were promoted to
        // `JA_STOPWORDS`. `でし` and `なっ` also appear standalone at run
        // boundaries in colloquial text (`そうでし！`, `なっ、そうか`) but are
        // deliberately deferred pending frequency evidence; `たい` collides
        // with hiragana 鯛 (`たいが釣れた` => [たい, が, 釣れた]) and stays out.
        assert!(is_stopword("だろう"));
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
        // Guideline: 50〜130 words (deduplicated). The upper bound exists to
        // catch the list turning into an ever-growing denylist of whatever
        // word most recently leaked into someone's keyword output — the
        // whack-a-mole failure mode that motivated making the bigram
        // heuristic structural (referral ref-20260710-012108-589608700). It is
        // not a hard budget: raise it deliberately when the segmenter starts
        // emitting a new *class* of function word, as happened when the
        // trained boundary model began producing standalone conjunctions,
        // demonstratives and adverbs, and again when サ変 auxiliary fragments
        // (`した`, `して`) were added.
        let unique: HashSet<&str> = JA_STOPWORDS.iter().copied().collect();
        assert!(
            unique.len() >= 50 && unique.len() <= 130,
            "stopword list has {} unique entries, expected 50..=130",
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
