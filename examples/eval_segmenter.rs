//! Development-time quality evaluation for the Japanese boundary segmenter.
//!
//! Reads the same litsea-compatible wakachi (space-segmented) corpus used for
//! training, runs the embedded runtime model
//! ([`lexsim::segmenter::push_segmented_ja`]) on the plain-text
//! reconstruction of each sentence, and compares the predicted word list
//! against the corpus's gold word list to compute word-level
//! Precision/Recall/F1 (see the segmenter design spec's active-learning
//! quality-gate notes).
//!
//! This is a dev-only tool: it evaluates the *currently embedded*
//! `src/model_data/ja_segmenter.bin` (baked in via `include_bytes!`), so
//! re-run `train_segmenter --bin-output ...` first if you want to evaluate a
//! freshly retrained model.
//!
//! ```text
//! cargo run --example eval_segmenter -- --corpus training/seed_corpus.txt
//! ```

use std::fs;
use std::process;

use lexsim::segmenter::push_segmented_ja;

/// Word-level Precision/Recall/F1 for one prediction against its gold word
/// list.
///
/// Both `gold` and `predicted` are ordered word-token multisets (duplicates,
/// e.g. repeated punctuation, are meaningful and not deduplicated). A
/// "match" is counted positionally-independent but multiplicity-aware: for
/// each distinct word, the number of matches is `min(count_in_gold,
/// count_in_predicted)` (a standard multiset-intersection word-F1, as used
/// for e.g. bag-of-words segmentation scoring).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WordMetrics {
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
}

/// Compute word-level P/R/F1 for a single sentence's predicted vs. gold word
/// lists.
///
/// - If `predicted` is empty and `gold` is empty, returns perfect (1.0/1.0/1.0)
///   scores (there is nothing to get wrong).
/// - If exactly one of `predicted`/`gold` is empty (but not both), returns all
///   zeros (no possible matches).
pub fn word_metrics(gold: &[String], predicted: &[String]) -> WordMetrics {
    if gold.is_empty() && predicted.is_empty() {
        return WordMetrics {
            precision: 1.0,
            recall: 1.0,
            f1: 1.0,
        };
    }
    if gold.is_empty() || predicted.is_empty() {
        return WordMetrics {
            precision: 0.0,
            recall: 0.0,
            f1: 0.0,
        };
    }

    let matches = multiset_intersection_count(gold, predicted);
    let precision = matches as f64 / predicted.len() as f64;
    let recall = matches as f64 / gold.len() as f64;
    let f1 = if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    };
    WordMetrics {
        precision,
        recall,
        f1,
    }
}

/// Count of matched words between two multisets of word tokens, i.e.
/// `sum(min(count_in_a[w], count_in_b[w]))` over all distinct words `w`.
fn multiset_intersection_count(a: &[String], b: &[String]) -> usize {
    use std::collections::HashMap;

    let mut counts: HashMap<&str, i64> = HashMap::new();
    for w in a {
        *counts.entry(w.as_str()).or_insert(0) += 1;
    }
    let mut matched = 0i64;
    for w in b {
        let c = counts.entry(w.as_str()).or_insert(0);
        if *c > 0 {
            *c -= 1;
            matched += 1;
        }
    }
    matched as usize
}

/// Aggregate (micro-averaged) word metrics over a whole evaluation set: sums
/// match/gold/predicted counts across all sentences before computing a
/// single P/R/F1, so long sentences aren't under-weighted relative to short
/// ones.
fn aggregate_metrics(pairs: &[(Vec<String>, Vec<String>)]) -> WordMetrics {
    let mut total_matches = 0usize;
    let mut total_gold = 0usize;
    let mut total_predicted = 0usize;

    for (gold, predicted) in pairs {
        total_matches += multiset_intersection_count(gold, predicted);
        total_gold += gold.len();
        total_predicted += predicted.len();
    }

    let precision = if total_predicted > 0 {
        total_matches as f64 / total_predicted as f64
    } else {
        1.0
    };
    let recall = if total_gold > 0 {
        total_matches as f64 / total_gold as f64
    } else {
        1.0
    };
    let f1 = if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    };
    WordMetrics {
        precision,
        recall,
        f1,
    }
}

/// Parse a litsea-compatible wakachi corpus line (see
/// `src/segmenter/corpus.rs`) into its gold word list. Returns `None` for
/// comment/blank lines.
fn gold_words_from_line(line: &str) -> Option<Vec<String>> {
    let trimmed = line.trim();
    if trimmed.is_empty() || trimmed.starts_with('#') {
        return None;
    }
    Some(
        trimmed
            .split(' ')
            .filter(|w| !w.is_empty())
            .map(str::to_string)
            .collect(),
    )
}

/// Run the runtime segmenter on the plain-text reconstruction of a gold word
/// list (i.e. the words joined back together with no separators, matching
/// how `tokenize()` feeds non-spacing runs to `push_segmented_ja`).
fn predict_words(gold: &[String]) -> Vec<String> {
    let plain: String = gold.concat();
    let mut out = Vec::new();
    push_segmented_ja(&plain, &mut out);
    out
}

fn parse_corpus_path() -> String {
    let mut corpus_path = "training/seed_corpus.txt".to_string();
    let mut argv = std::env::args().skip(1);
    while let Some(arg) = argv.next() {
        match arg.as_str() {
            "--corpus" => {
                corpus_path = argv.next().unwrap_or_else(|| {
                    eprintln!("--corpus requires a value");
                    process::exit(2);
                });
            }
            other => {
                eprintln!("unknown argument: {other}");
                process::exit(2);
            }
        }
    }
    corpus_path
}

fn main() {
    let corpus_path = parse_corpus_path();

    let corpus_text = fs::read_to_string(&corpus_path).unwrap_or_else(|e| {
        eprintln!("failed to read corpus {corpus_path:?}: {e}");
        process::exit(1);
    });

    let pairs: Vec<(Vec<String>, Vec<String>)> = corpus_text
        .lines()
        .filter_map(gold_words_from_line)
        .map(|gold| {
            let predicted = predict_words(&gold);
            (gold, predicted)
        })
        .collect();

    if pairs.is_empty() {
        eprintln!("corpus produced zero evaluable sentences: {corpus_path:?}");
        process::exit(1);
    }

    let metrics = aggregate_metrics(&pairs);

    eprintln!("evaluated {} sentences from {:?}", pairs.len(), corpus_path);
    eprintln!("word precision: {:.4}", metrics.precision);
    eprintln!("word recall:    {:.4}", metrics.recall);
    eprintln!("word F1:        {:.4}", metrics.f1);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_segmentation_scores_perfectly() {
        let gold = vec!["メモリ".to_string(), "機能".to_string()];
        let predicted = gold.clone();
        let m = word_metrics(&gold, &predicted);
        assert_eq!(
            m,
            WordMetrics {
                precision: 1.0,
                recall: 1.0,
                f1: 1.0
            }
        );
    }

    #[test]
    fn merged_prediction_reduces_precision_and_recall() {
        // Gold: ["メモリ", "機能"] (2 words). Predicted (bigram-fallback style
        // over-merge, matching the historical "メモリ機能" bug pattern):
        // ["メモリ機能"] (1 word, no overlap with either gold word).
        let gold = vec!["メモリ".to_string(), "機能".to_string()];
        let predicted = vec!["メモリ機能".to_string()];
        let m = word_metrics(&gold, &predicted);
        assert_eq!(m.precision, 0.0);
        assert_eq!(m.recall, 0.0);
        assert_eq!(m.f1, 0.0);
    }

    #[test]
    fn partial_overlap_computes_expected_precision_recall_f1() {
        // Gold: ["猫", "が", "好き"] (3 words).
        // Predicted: ["猫", "が好き"] (2 words) — "猫" matches, "が好き" does not.
        let gold = vec!["猫".to_string(), "が".to_string(), "好き".to_string()];
        let predicted = vec!["猫".to_string(), "が好き".to_string()];
        let m = word_metrics(&gold, &predicted);
        // matches = 1; precision = 1/2 = 0.5; recall = 1/3.
        assert!((m.precision - 0.5).abs() < 1e-9);
        assert!((m.recall - (1.0 / 3.0)).abs() < 1e-9);
        let expected_f1 = 2.0 * 0.5 * (1.0 / 3.0) / (0.5 + 1.0 / 3.0);
        assert!((m.f1 - expected_f1).abs() < 1e-9);
    }

    #[test]
    fn both_empty_is_perfect_score() {
        let m = word_metrics(&[], &[]);
        assert_eq!(
            m,
            WordMetrics {
                precision: 1.0,
                recall: 1.0,
                f1: 1.0
            }
        );
    }

    #[test]
    fn only_predicted_empty_is_zero_score() {
        let gold = vec!["猫".to_string()];
        let m = word_metrics(&gold, &[]);
        assert_eq!(
            m,
            WordMetrics {
                precision: 0.0,
                recall: 0.0,
                f1: 0.0
            }
        );
    }

    #[test]
    fn only_gold_empty_is_zero_score() {
        let predicted = vec!["猫".to_string()];
        let m = word_metrics(&[], &predicted);
        assert_eq!(
            m,
            WordMetrics {
                precision: 0.0,
                recall: 0.0,
                f1: 0.0
            }
        );
    }

    #[test]
    fn duplicate_words_are_multiset_matched_not_deduplicated() {
        // Gold has "た" repeated 3 times (e.g. multiple past-tense verb endings
        // in a sentence); predicted also produces "た" 3 times plus one extra.
        let gold = vec!["た".to_string(), "た".to_string(), "た".to_string()];
        let predicted = vec![
            "た".to_string(),
            "た".to_string(),
            "た".to_string(),
            "た".to_string(),
        ];
        let m = word_metrics(&gold, &predicted);
        // matches = min(3, 4) = 3; precision = 3/4; recall = 3/3 = 1.0.
        assert!((m.precision - 0.75).abs() < 1e-9);
        assert_eq!(m.recall, 1.0);
    }

    #[test]
    fn gold_words_from_line_skips_comments_and_blanks() {
        assert_eq!(gold_words_from_line("# a comment"), None);
        assert_eq!(gold_words_from_line("   "), None);
        assert_eq!(
            gold_words_from_line("猫 が 好き"),
            Some(vec!["猫".to_string(), "が".to_string(), "好き".to_string()])
        );
    }

    #[test]
    fn memory_function_prediction_matches_gold_exactly() {
        // Regression guard mirroring the spec's canonical example: the
        // runtime model must reproduce the gold "メモリ 機能" split exactly,
        // giving a perfect word-F1 for this sentence.
        let gold = vec!["メモリ".to_string(), "機能".to_string()];
        let predicted = predict_words(&gold);
        assert_eq!(predicted, gold);
        let m = word_metrics(&gold, &predicted);
        assert_eq!(m.f1, 1.0);
    }

    #[test]
    fn aggregate_metrics_micro_averages_across_sentences() {
        let pairs = vec![
            (
                vec!["猫".to_string(), "が".to_string()],
                vec!["猫".to_string(), "が".to_string()],
            ),
            (
                vec!["犬".to_string(), "が".to_string()],
                vec!["犬が".to_string()],
            ),
        ];
        let m = aggregate_metrics(&pairs);
        // total matches = 2 (first pair) + 0 (second pair) = 2.
        // total predicted = 2 + 1 = 3; total gold = 2 + 2 = 4.
        assert!((m.precision - (2.0 / 3.0)).abs() < 1e-9);
        assert!((m.recall - 0.5).abs() < 1e-9);
    }
}
