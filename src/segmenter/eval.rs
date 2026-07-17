//! Quality evaluation for the Japanese boundary segmenter: word- and
//! boundary-level Precision/Recall/F1 of the runtime model against a gold
//! wakachi (space-segmented) corpus.
//!
//! Lives in the library (rather than in `examples/eval_segmenter.rs`) so both
//! the dev-time reporting tool and the `tests/segmenter_quality.rs` regression
//! gate can share one definition of "how well does the segmenter do".
//!
//! # Only Japanese runs are evaluated
//!
//! [`crate::tokenize::tokenize`] splits text by script class *first* and hands
//! only hiragana/katakana/kanji runs to
//! [`crate::segmenter::push_segmented_ja`]; ASCII, digits and punctuation
//! never reach the segmenter. An evaluator that reconstructs a whole gold
//! sentence with `words.concat()` therefore feeds the segmenter input that
//! never occurs in production — the spaces between ASCII words vanish, so
//! `["Cannot", "read"]` becomes `"Cannotread"`, which the non-Japanese bigram
//! fallback shreds into `["Ca", "an", "nn", ...]`. That inflates the error
//! count with failures the tokenizer can never produce.
//!
//! So [`evaluate_corpus`] extracts the *maximal all-Japanese-script spans* of
//! each gold word list and scores each span independently, approximating how
//! `tokenize()` actually feeds the segmenter. Words containing any
//! non-Japanese character are boundary markers between spans and are not
//! scored — segmenting them is not the segmenter's job.
//!
//! The approximation is not byte-exact: `tokenize()` NFKC-normalizes first,
//! and its `ScriptClass::NonSpacing` is not quite
//! [`is_japanese_run_char`] (CJK Extension B is classed differently; `々`
//! U+3005 is kanji in both since the iteration-mark fix). The gold corpora
//! contain no Extension B characters, so the two agree on everything this
//! module scores; a corpus that introduced them would need this run
//! extraction reconciled with `tokenize.rs`'s `classify`.

use super::inference::is_japanese_run_char;
use super::push_segmented_ja;

/// Precision/Recall/F1 for one evaluation run.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Metrics {
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
}

impl Metrics {
    /// Build metrics from raw true-positive / false-positive / false-negative
    /// counts. With no predictions and no gold items, precision and recall are
    /// vacuously 1.0 (nothing to get wrong).
    fn from_counts(tp: usize, fp: usize, r#fn: usize) -> Self {
        let precision = if tp + fp == 0 {
            1.0
        } else {
            tp as f64 / (tp + fp) as f64
        };
        let recall = if tp + r#fn == 0 {
            1.0
        } else {
            tp as f64 / (tp + r#fn) as f64
        };
        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };
        Metrics {
            precision,
            recall,
            f1,
        }
    }
}

/// Aggregate quality report over a whole corpus.
///
/// `#[non_exhaustive]`: new metrics may be added without a breaking change, so
/// construct one only via [`evaluate_corpus`] and match with `..`.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct Report {
    /// Boundary-level metrics: each internal character gap of a Japanese run
    /// is one item, positive iff the gold segmentation places a word boundary
    /// there. This is the metric the regression gate uses — it degrades
    /// smoothly and is insensitive to how runs are chunked into sentences.
    pub boundary: Metrics,
    /// Word-level metrics: multiset intersection of gold vs predicted word
    /// tokens. Stricter than `boundary` (one missed boundary can spoil two
    /// words) and reported for human consumption.
    pub word: Metrics,
    /// Number of Japanese runs evaluated (runs of fewer than two words carry
    /// no internal boundary decision and are skipped).
    pub runs: usize,
    /// Runs whose predicted word list exactly equals the gold word list.
    pub exact_runs: usize,
}

/// Parse one litsea-compatible wakachi corpus line into its gold word list.
/// Returns `None` for comment (`#`) and blank lines. Mirrors
/// [`crate::segmenter::corpus`]'s parsing rules.
pub fn gold_words_from_line(line: &str) -> Option<Vec<String>> {
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

/// True if every character of `word` is a Japanese-script character, i.e. the
/// word would live inside a run that `tokenize()` hands to the segmenter.
fn is_japanese_word(word: &str) -> bool {
    !word.is_empty() && word.chars().all(is_japanese_run_char)
}

/// Split a gold word list into maximal runs of consecutive all-Japanese words.
///
/// A word containing any non-Japanese character (ASCII, digits, `。`, `、`,
/// ...) terminates the current run and is itself dropped: `tokenize()` would
/// have routed it to a different script segment, never to the segmenter.
fn japanese_runs(gold: &[String]) -> Vec<Vec<String>> {
    let mut runs = Vec::new();
    let mut current: Vec<String> = Vec::new();
    for w in gold {
        if is_japanese_word(w) {
            current.push(w.clone());
        } else if !current.is_empty() {
            runs.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        runs.push(current);
    }
    runs
}

/// Character offsets of the internal word boundaries implied by a word list.
/// A single-word list has no internal boundary, so this returns empty.
fn boundary_offsets(words: &[String]) -> Vec<usize> {
    let mut offsets = Vec::new();
    let mut acc = 0usize;
    for w in words.iter().take(words.len().saturating_sub(1)) {
        acc += w.chars().count();
        offsets.push(acc);
    }
    offsets
}

/// `sum(min(count_in_a[w], count_in_b[w]))` over all distinct words — a
/// multiplicity-aware intersection size (repeated tokens are meaningful).
fn multiset_intersection_count(a: &[String], b: &[String]) -> usize {
    use std::collections::HashMap;
    let mut counts: HashMap<&str, i64> = HashMap::new();
    for w in a {
        *counts.entry(w.as_str()).or_insert(0) += 1;
    }
    let mut matched = 0usize;
    for w in b {
        let c = counts.entry(w.as_str()).or_insert(0);
        if *c > 0 {
            *c -= 1;
            matched += 1;
        }
    }
    matched
}

/// Segment the plain-text reconstruction of one all-Japanese gold run.
fn predict_run(gold_run: &[String]) -> Vec<String> {
    let plain: String = gold_run.concat();
    let mut out = Vec::new();
    push_segmented_ja(&plain, &mut out);
    out
}

/// Evaluate the embedded runtime model against a wakachi corpus.
///
/// `corpus_text` is the raw file content (comments and blank lines allowed).
/// Only maximal all-Japanese-script spans are scored — see the module docs for
/// why. Counts are micro-averaged across every scored run.
pub fn evaluate_corpus(corpus_text: &str) -> Report {
    let (mut b_tp, mut b_fp, mut b_fn) = (0usize, 0usize, 0usize);
    let (mut w_matches, mut w_gold, mut w_pred) = (0usize, 0usize, 0usize);
    let (mut runs, mut exact_runs) = (0usize, 0usize);

    for line in corpus_text.lines() {
        let Some(gold) = gold_words_from_line(line) else {
            continue;
        };
        for run in japanese_runs(&gold) {
            // A one-word run has no internal boundary decision to score.
            if run.len() < 2 {
                continue;
            }
            let predicted = predict_run(&run);
            runs += 1;
            if run == predicted {
                exact_runs += 1;
            }

            let gold_b: std::collections::HashSet<usize> =
                boundary_offsets(&run).into_iter().collect();
            let pred_b: std::collections::HashSet<usize> =
                boundary_offsets(&predicted).into_iter().collect();
            b_tp += gold_b.intersection(&pred_b).count();
            b_fp += pred_b.difference(&gold_b).count();
            b_fn += gold_b.difference(&pred_b).count();

            w_matches += multiset_intersection_count(&run, &predicted);
            w_gold += run.len();
            w_pred += predicted.len();
        }
    }

    // `w_matches` is a multiset intersection size, so it cannot exceed either
    // side; `saturating_sub` keeps that invariant from becoming a panic if a
    // future refactor breaks it.
    Report {
        boundary: Metrics::from_counts(b_tp, b_fp, b_fn),
        word: Metrics::from_counts(
            w_matches,
            w_pred.saturating_sub(w_matches),
            w_gold.saturating_sub(w_matches),
        ),
        runs,
        exact_runs,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words(ws: &[&str]) -> Vec<String> {
        ws.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn gold_words_skips_comments_and_blanks() {
        assert_eq!(gold_words_from_line("# comment"), None);
        assert_eq!(gold_words_from_line("   "), None);
        assert_eq!(
            gold_words_from_line("猫 が 鳴く"),
            Some(words(&["猫", "が", "鳴く"]))
        );
    }

    #[test]
    fn japanese_runs_split_on_non_japanese_words() {
        // This is the core of the ASCII-contamination fix: `AI` and `。` are
        // never fed to the segmenter by `tokenize()`, so they split the gold
        // word list into separate runs rather than being concatenated into it.
        let gold = words(&["AI", "に", "よる", "翻訳", "。"]);
        assert_eq!(japanese_runs(&gold), vec![words(&["に", "よる", "翻訳"])]);
    }

    #[test]
    fn japanese_runs_yields_multiple_spans() {
        let gold = words(&["彼", "は", "Rust", "を", "書く"]);
        assert_eq!(
            japanese_runs(&gold),
            vec![words(&["彼", "は"]), words(&["を", "書く"])]
        );
    }

    #[test]
    fn ascii_words_are_never_shredded_into_bigrams() {
        // Regression guard for the bug this module replaces: concatenating a
        // gold list containing multi-word ASCII spans made the evaluator feed
        // "Cannotread" to the bigram fallback, producing garbage predictions.
        // With run extraction, ASCII simply is not scored.
        let gold = words(&["Cannot", "read", "properties"]);
        assert!(japanese_runs(&gold).is_empty());
        let report = evaluate_corpus("Cannot read properties");
        assert_eq!(report.runs, 0, "pure-ASCII line contributes no runs");
    }

    #[test]
    fn boundary_offsets_are_cumulative_char_counts() {
        assert_eq!(boundary_offsets(&words(&["猫", "が", "鳴く"])), vec![1, 2]);
        assert_eq!(boundary_offsets(&words(&["単語"])), Vec::<usize>::new());
    }

    #[test]
    fn multiset_intersection_respects_multiplicity() {
        let a = words(&["の", "の", "花"]);
        let b = words(&["の", "花", "花"]);
        // one `の` and one `花` match; the duplicates do not double-count.
        assert_eq!(multiset_intersection_count(&a, &b), 2);
    }

    #[test]
    fn perfect_prediction_scores_one() {
        let m = Metrics::from_counts(10, 0, 0);
        assert_eq!(m.precision, 1.0);
        assert_eq!(m.recall, 1.0);
        assert_eq!(m.f1, 1.0);
    }

    #[test]
    fn empty_counts_are_vacuously_perfect() {
        let m = Metrics::from_counts(0, 0, 0);
        assert_eq!(m.precision, 1.0);
        assert_eq!(m.recall, 1.0);
    }

    #[test]
    fn single_word_runs_are_skipped() {
        // No internal boundary to decide, so it must not inflate the counts.
        let report = evaluate_corpus("猫");
        assert_eq!(report.runs, 0);
    }

    #[test]
    fn evaluating_the_real_corpus_beats_the_regression_floor() {
        // Sanity check that the module wired up against the embedded model
        // produces a plausible score on a handful of gold sentences whose
        // segmentation the model is known to reproduce.
        let corpus = "昨日 は 雨 が 降った\n彼 は 本 を 読んだ\n";
        let report = evaluate_corpus(corpus);
        assert_eq!(report.runs, 2);
        assert!(
            report.boundary.f1 > 0.5,
            "boundary F1 unexpectedly low: {:?}",
            report.boundary
        );
    }
}
