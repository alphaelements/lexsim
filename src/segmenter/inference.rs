//! Runtime inference: split a mixed-script "non-spacing" run (as produced by
//! `src/tokenize.rs`'s script segmentation) into individually-segmented
//! pieces, and push tokens for each piece to an output vector.
//!
//! This mirrors the contract of `tokenize.rs`'s private `push_cjk_bigrams`
//! (`&str` in, `&mut Vec<String>` out, order-preserving, duplicates kept) but
//! adds real Japanese word segmentation for hiragana/katakana/kanji runs,
//! using the trained boundary model ([`super::MODEL`]) via
//! [`super::features::extract_features`]. Non-Japanese non-spacing scripts
//! (Hangul, unclassified CJK-adjacent blocks, etc.) still fall back to the
//! bigram scheme, since the model was only ever trained on Japanese-run
//! boundary candidates (see the segmenter design spec §2.2, §3.5).
//!
//! Not yet wired into `tokenize()` — see the segmenter design spec's phased
//! rollout (this lands in a later phase).

use super::features::{classify_char, extract_features, CharClass, PriorDecision};
use super::MODEL;

/// Returns `true` if `c` belongs to one of the Japanese scripts the learned
/// segmenter operates on (hiragana, katakana, kanji incl. kanji numerals).
/// Matches `segmenter::corpus`'s training-time definition of a "Japanese run"
/// so inference-time behavior matches what the model was trained on.
fn is_japanese_run_char(c: char) -> bool {
    matches!(
        classify_char(c),
        CharClass::Hiragana | CharClass::Katakana | CharClass::Kanji | CharClass::KanjiNumeral
    )
}

/// Segment `text` (a run of non-spacing-script characters — see
/// `tokenize.rs`'s `ScriptClass::NonSpacing`) and push the resulting tokens to
/// `out`, preserving order and keeping duplicates (same contract as
/// `push_cjk_bigrams`).
///
/// `text` is first split into maximal sub-runs of "Japanese" (hiragana /
/// katakana / kanji) vs. "other non-spacing" (Hangul, etc.) characters:
/// - Japanese sub-runs are segmented with the trained boundary model: for
///   each character gap, [`extract_features`] produces the litsea-style
///   feature set, [`super::MODEL`]'s weights are summed with the model bias,
///   and a positive score marks a boundary. The run is then split at boundary
///   positions into word segments, each pushed as one token.
/// - Non-Japanese sub-runs (and single-character Japanese sub-runs) fall back
///   to the same character-bigram scheme `tokenize.rs` uses for CJK/Hangul
///   today (a trailing lone character becomes a unigram).
pub fn push_segmented_ja(text: &str, out: &mut Vec<String>) {
    let chars: Vec<char> = text.chars().collect();
    if chars.is_empty() {
        return;
    }

    for run in sub_runs(&chars) {
        if run.is_japanese {
            push_model_segmented(&chars[run.start..run.end], out);
        } else {
            push_bigrams(&chars[run.start..run.end], out);
        }
    }
}

/// A maximal sub-run of `chars` that is uniformly Japanese or uniformly
/// non-Japanese (both classes defined by [`is_japanese_run_char`]).
struct SubRun {
    start: usize,
    end: usize,
    is_japanese: bool,
}

/// Split `chars` into maximal runs of `is_japanese_run_char` vs not.
fn sub_runs(chars: &[char]) -> Vec<SubRun> {
    let mut runs = Vec::new();
    let mut start = 0usize;
    let mut current: Option<bool> = None;

    for (i, &c) in chars.iter().enumerate() {
        let is_ja = is_japanese_run_char(c);
        match current {
            Some(cur) if cur == is_ja => {}
            Some(cur) => {
                runs.push(SubRun {
                    start,
                    end: i,
                    is_japanese: cur,
                });
                start = i;
                current = Some(is_ja);
            }
            None => {
                start = i;
                current = Some(is_ja);
            }
        }
    }
    if let Some(cur) = current {
        runs.push(SubRun {
            start,
            end: chars.len(),
            is_japanese: cur,
        });
    }
    runs
}

/// Segment a Japanese-only run of characters using the trained boundary
/// model, and push each resulting word segment to `out`.
fn push_model_segmented(chars: &[char], out: &mut Vec<String>) {
    if chars.is_empty() {
        return;
    }
    if chars.len() == 1 {
        out.push(chars[0].to_string());
        return;
    }

    let bias = MODEL.bias() as f64;
    let mut prior: Vec<PriorDecision> = Vec::new();
    // `is_boundary_after[i]` is true iff there is a segment boundary between
    // `chars[i]` and `chars[i + 1]`.
    let mut is_boundary_after = vec![false; chars.len() - 1];

    for (i, slot) in is_boundary_after.iter_mut().enumerate() {
        let features = extract_features(chars, i, &prior);
        let mut score = bias;
        for f in &features {
            score += MODEL.lookup(f) as f64;
        }
        let boundary = score > 0.0;
        *slot = boundary;
        prior.insert(
            0,
            if boundary {
                PriorDecision::Boundary
            } else {
                PriorDecision::NoBoundary
            },
        );
        prior.truncate(3);
    }

    let mut seg_start = 0usize;
    for (i, &boundary) in is_boundary_after.iter().enumerate() {
        if boundary {
            out.push(chars[seg_start..=i].iter().collect());
            seg_start = i + 1;
        }
    }
    out.push(chars[seg_start..].iter().collect());
}

/// Character bi-grams for a non-Japanese non-spacing run; a trailing single
/// char becomes a unigram so a one-character run still produces a token.
/// Equivalent logic to `tokenize.rs`'s private `push_cjk_bigrams`, duplicated
/// here since that function isn't reachable from this module (see the P3
/// task notes: `push_cjk_bigrams` itself is not modified or exposed).
fn push_bigrams(chars: &[char], out: &mut Vec<String>) {
    if chars.is_empty() {
        return;
    }
    if chars.len() == 1 {
        out.push(chars[0].to_string());
        return;
    }
    for window in chars.windows(2) {
        out.push(window.iter().collect());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_function_splits_correctly() {
        // Spec §7.4 recommended test: "メモリ機能" → ["メモリ", "機能"].
        let mut out = Vec::new();
        push_segmented_ja("メモリ機能", &mut out);
        assert_eq!(out, vec!["メモリ".to_string(), "機能".to_string()]);
    }

    #[test]
    fn katakana_compound_word_not_split() {
        // "トークナイザー" is a single katakana compound noun and must not be
        // split (regression test against the bigram-era "メモリ" bug pattern).
        let mut out = Vec::new();
        push_segmented_ja("トークナイザー", &mut out);
        assert_eq!(out, vec!["トークナイザー".to_string()]);
    }

    #[test]
    fn verb_conjugation_splits_plausibly() {
        // "走った" ("ran", past tense) — segmentation must not panic and must
        // reconstruct the original text when segments are joined back.
        let mut out = Vec::new();
        push_segmented_ja("走った", &mut out);
        assert!(!out.is_empty());
        assert_eq!(out.concat(), "走った");
    }

    #[test]
    fn hangul_input_falls_back_to_bigrams() {
        // Hangul is not a Japanese run char, so it must use the bigram
        // fallback rather than the (Japanese-only-trained) model.
        let mut out = Vec::new();
        push_segmented_ja("안녕하세요", &mut out);
        let chars: Vec<char> = "안녕하세요".chars().collect();
        let expected: Vec<String> = chars.windows(2).map(|w| w.iter().collect()).collect();
        assert_eq!(out, expected);
    }

    #[test]
    fn single_char_input_is_unigram() {
        let mut out = Vec::new();
        push_segmented_ja("猫", &mut out);
        assert_eq!(out, vec!["猫".to_string()]);
    }

    #[test]
    fn empty_input_pushes_nothing() {
        let mut out = Vec::new();
        push_segmented_ja("", &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn mixed_japanese_and_hangul_runs_each_handled() {
        // Japanese run segmented by the model, Hangul run falls back to
        // bigrams — and every original character is preserved across the
        // concatenation of all pushed segments.
        let mut out = Vec::new();
        push_segmented_ja("メモリ機能안녕", &mut out);
        assert!(!out.is_empty());

        // Reconstructing via the sub-run boundaries: the Japanese portion
        // ("メモリ機能") must appear as whole-word pushes, and "안녕" (2 Hangul
        // chars) must appear as a bigram fallback (single 2-char run -> one
        // bigram token, no unigram since len == 2).
        assert!(out.contains(&"안녕".to_string()));
        assert!(out.iter().any(|t| t.contains('メ') || t.contains('機')));
    }

    #[test]
    fn concatenating_all_segments_reconstructs_original_japanese_run() {
        let mut out = Vec::new();
        push_segmented_ja("メモリ機能", &mut out);
        assert_eq!(out.concat(), "メモリ機能");
    }
}
