//! litsea-compatible wakachi (space-segmented) corpus loading: extracts
//! Japanese runs from segmented sentences and turns them into labeled
//! boundary-candidate training instances.
//!
//! Corpus format (see `training/seed_corpus.txt`):
//! - one sentence per line, words separated by ASCII spaces;
//! - lines starting with `#` (after trimming) are comments and are skipped;
//! - blank lines are skipped.
//!
//! A wakachi (segmented) line is joined back into plain text, and only the
//! Japanese runs (hiragana / katakana / kanji, contiguous) are kept as
//! training material — this matches the design's "hybrid boundary" approach
//! where the learned segmenter only ever looks at Japanese runs (see the
//! segmenter design spec §5.3). For each character gap inside such a run we
//! derive a `+1` (boundary) / `-1` (no boundary) label from the original
//! word-segmentation, then apply the feature template.

use super::features::{classify_char, extract_features, CharClass, PriorDecision};

/// A single training instance: the feature keys active at one boundary
/// candidate, and its gold label (`true` = boundary, `false` = no boundary).
#[derive(Debug, Clone)]
pub struct Instance {
    pub features: Vec<String>,
    pub label: bool,
}

/// Returns `true` if `c` belongs to one of the Japanese scripts the learned
/// segmenter operates on (hiragana, katakana, kanji incl. kanji numerals).
/// Punctuation and everything else are run boundaries.
fn is_japanese_run_char(c: char) -> bool {
    matches!(
        classify_char(c),
        CharClass::Hiragana | CharClass::Katakana | CharClass::Kanji | CharClass::KanjiNumeral
    )
}

/// Parse the corpus text into training instances.
///
/// `#`-prefixed and blank lines are skipped. Each remaining line is treated as
/// a wakachi (space-segmented) sentence; word boundaries at the seams between
/// space-separated tokens become gold `+1` (boundary) labels, and all other
/// intra-token character gaps become `-1` (no boundary) labels. Only gaps that
/// fall entirely within a contiguous Japanese run (see
/// `is_japanese_run_char`) are emitted — this mirrors the runtime's
/// hybrid design where the learned model is applied only to Japanese runs.
pub fn load_corpus(text: &str) -> Vec<Instance> {
    let mut instances = Vec::new();
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        instances.extend(instances_from_wakachi_line(trimmed));
    }
    instances
}

/// Convert one wakachi line into training instances, one per Japanese-run
/// internal character gap.
fn instances_from_wakachi_line(line: &str) -> Vec<Instance> {
    // Rebuild the plain (unsegmented) text and, in parallel, mark which
    // character gaps in that text were word boundaries in the wakachi input.
    let words: Vec<&str> = line.split(' ').filter(|w| !w.is_empty()).collect();

    let mut plain = String::new();
    let mut is_boundary_after: Vec<bool> = Vec::new(); // indexed by char position in `plain`
    for (wi, word) in words.iter().enumerate() {
        let char_count = word.chars().count();
        plain.push_str(word);
        for ci in 0..char_count {
            // A gap after the last char of a word is a word boundary
            // (unless it's the very last char of the whole line).
            let is_last_char_of_word = ci == char_count - 1;
            let is_last_word = wi == words.len() - 1;
            is_boundary_after.push(is_last_char_of_word && !is_last_word);
        }
    }

    let chars: Vec<char> = plain.chars().collect();
    if chars.len() < 2 {
        return Vec::new();
    }

    let mut out = Vec::new();
    let mut prior: Vec<PriorDecision> = Vec::new();

    for i in 0..chars.len() - 1 {
        let boundary_here = is_boundary_after[i];
        // Only train on gaps that lie inside a contiguous Japanese run: both
        // characters adjacent to the candidate must be Japanese-run chars.
        if is_japanese_run_char(chars[i]) && is_japanese_run_char(chars[i + 1]) {
            let features = extract_features(&chars, i, &prior);
            out.push(Instance {
                features,
                label: boundary_here,
            });
        }

        // Track prior decisions for the *next* candidate's UP/BP/UQ/BQ/TQ
        // features, nearest-first, regardless of whether this candidate was
        // in-run (mirrors litsea: prior decisions accumulate over the whole
        // line's boundary sequence).
        let decision = if boundary_here {
            PriorDecision::Boundary
        } else {
            PriorDecision::NoBoundary
        };
        prior.insert(0, decision);
        prior.truncate(3);
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn skips_comments_and_blank_lines() {
        let text = "# a comment\n\n猫 が 好き\n";
        let instances = load_corpus(text);
        assert!(!instances.is_empty());
    }

    #[test]
    fn wakachi_line_produces_boundary_labels() {
        // "猫 が 好き" -> plain "猫が好き"
        // word boundaries: after '猫' (end of word 1), after 'が' (end of word 2).
        // '好き' is the last word, so no boundary after 'き' (end of line).
        // Japanese-run gaps: (猫|が)=i0, (が|好)=i1, (好|き)=i2 — all Japanese chars.
        let instances = instances_from_wakachi_line("猫 が 好き");
        assert_eq!(instances.len(), 3);
        assert!(instances[0].label, "gap after 猫 should be a boundary");
        assert!(instances[1].label, "gap after が should be a boundary");
        assert!(
            !instances[2].label,
            "gap inside 好き should not be a boundary"
        );
    }

    #[test]
    fn non_japanese_gaps_are_excluded() {
        // "cat が 好き" -> plain "catが好き". The gap between 't' and 'が' spans
        // an ASCII char and a Japanese char, so it must not be emitted as a
        // training instance (only Japanese-run-internal gaps are kept).
        let instances = instances_from_wakachi_line("cat が 好き");
        // Only the Japanese-run gaps (が|好) and (好|き) should appear.
        assert_eq!(instances.len(), 2);
    }

    #[test]
    fn single_word_line_has_no_boundary_at_end() {
        let instances = instances_from_wakachi_line("好き");
        assert_eq!(instances.len(), 1);
        assert!(!instances[0].label);
    }

    #[test]
    fn features_are_the_42_template_keys() {
        let instances = instances_from_wakachi_line("猫 が 好き");
        for inst in &instances {
            assert_eq!(inst.features.len(), 42);
        }
    }

    #[test]
    fn load_corpus_seed_file_smoke() {
        // Sanity check against the actual seed corpus shape (small excerpt),
        // ensuring no panics and a non-trivial number of instances comes out.
        let text = "\
# header comment
セッション 開始 時 に handoff_load_context を 呼ん で コンテキスト を 復元 する 。
メモリ 機能 を 使っ て データ を 保存 する 。
";
        let instances = load_corpus(text);
        assert!(instances.len() > 10);
    }
}
