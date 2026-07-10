//! Development-time quality report for the Japanese boundary segmenter.
//!
//! Evaluates the *currently embedded* `src/model_data/ja_segmenter.bin` (baked
//! in via `include_bytes!`) against a litsea-compatible wakachi
//! (space-segmented) corpus, and prints boundary- and word-level
//! Precision/Recall/F1. Re-run `train_segmenter --bin-output ...` first if you
//! want to evaluate a freshly retrained model.
//!
//! The scoring itself lives in [`lexsim::segmenter::eval`] so that this tool
//! and the `tests/segmenter_quality.rs` regression gate can never disagree
//! about what the numbers mean.
//!
//! Only maximal all-Japanese-script spans of each gold sentence are scored,
//! mirroring how `tokenize()` feeds runs to the segmenter — ASCII, digits and
//! punctuation never reach it. See the `segmenter::eval` module docs.
//!
//! ```text
//! cargo run --example eval_segmenter -- --corpus training/seed_corpus.txt
//! ```

use std::fs;
use std::process;

use lexsim::segmenter::eval::evaluate_corpus;

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

    let report = evaluate_corpus(&corpus_text);

    if report.runs == 0 {
        eprintln!("corpus produced zero evaluable Japanese runs: {corpus_path:?}");
        process::exit(1);
    }

    let exact_ratio = report.exact_runs as f64 / report.runs as f64;

    eprintln!(
        "evaluated {} Japanese runs from {:?}",
        report.runs, corpus_path
    );
    eprintln!(
        "exact runs:        {}/{} ({:.1}%)",
        report.exact_runs,
        report.runs,
        100.0 * exact_ratio
    );
    eprintln!("boundary precision: {:.4}", report.boundary.precision);
    eprintln!("boundary recall:    {:.4}", report.boundary.recall);
    eprintln!("boundary F1:        {:.4}", report.boundary.f1);
    eprintln!("word precision:     {:.4}", report.word.precision);
    eprintln!("word recall:        {:.4}", report.word.recall);
    eprintln!("word F1:            {:.4}", report.word.f1);
}
