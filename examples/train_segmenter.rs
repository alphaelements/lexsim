//! Development-time training tool for the Japanese boundary segmenter.
//!
//! Reads a litsea-compatible wakachi (space-segmented) corpus, extracts
//! Japanese-run boundary-candidate training instances, trains a deterministic
//! AdaBoost classifier, and writes the resulting model as JSON.
//!
//! This is a dev-only tool: it requires the `cli` feature (for `serde_json`)
//! and is not part of the crate's default build path.
//!
//! ```text
//! cargo run --example train_segmenter --features cli -- \
//!     --corpus training/seed_corpus.txt --output training/model.json \
//!     --bin-output src/model_data/ja_segmenter.bin --iterations 1000
//! ```
//!
//! `--corpus` may be repeated to train on multiple files, and/or point at a
//! directory (in which case every `*.txt` file directly inside it is loaded,
//! in sorted-filename order):
//!
//! ```text
//! cargo run --example train_segmenter --features cli -- \
//!     --corpus training/seed_corpus.txt --corpus training/dialect_corpus.txt \
//!     --output training/model.json
//!
//! cargo run --example train_segmenter --features cli -- \
//!     --corpus training/ --output training/model.json
//! ```

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process;

use lexsim::segmenter::adaboost::{evaluate, train, Model, Sample};
use lexsim::segmenter::binary::{encode, fnv1a_64};
use lexsim::segmenter::corpus::load_corpus;

struct Args {
    corpus_paths: Vec<String>,
    output_path: String,
    bin_output_path: Option<String>,
    iterations: usize,
}

fn parse_args() -> Args {
    let mut corpus_paths: Vec<String> = Vec::new();
    let mut output_path = "training/model.json".to_string();
    let mut bin_output_path: Option<String> = None;
    let mut iterations = 1000usize;

    let mut argv = std::env::args().skip(1);
    while let Some(arg) = argv.next() {
        match arg.as_str() {
            "--corpus" => {
                let path = argv.next().unwrap_or_else(|| {
                    eprintln!("--corpus requires a value");
                    process::exit(2);
                });
                corpus_paths.push(path);
            }
            "--output" => {
                output_path = argv.next().unwrap_or_else(|| {
                    eprintln!("--output requires a value");
                    process::exit(2);
                });
            }
            "--bin-output" => {
                bin_output_path = Some(argv.next().unwrap_or_else(|| {
                    eprintln!("--bin-output requires a value");
                    process::exit(2);
                }));
            }
            "--iterations" => {
                let val = argv.next().unwrap_or_else(|| {
                    eprintln!("--iterations requires a value");
                    process::exit(2);
                });
                iterations = val.parse().unwrap_or_else(|_| {
                    eprintln!("--iterations must be a positive integer, got {val:?}");
                    process::exit(2);
                });
            }
            other => {
                eprintln!("unknown argument: {other}");
                process::exit(2);
            }
        }
    }

    if corpus_paths.is_empty() {
        corpus_paths.push("training/seed_corpus.txt".to_string());
    }

    Args {
        corpus_paths,
        output_path,
        bin_output_path,
        iterations,
    }
}

/// Expand a single `--corpus` argument into a list of corpus file paths.
///
/// - A path to a regular file is returned as-is (one-element list).
/// - A path to a directory is expanded to every `*.txt` file directly inside
///   it (not recursive), sorted by filename so that concatenation order is
///   deterministic.
fn expand_corpus_path(path: &str) -> Result<Vec<PathBuf>, String> {
    let p = Path::new(path);
    let metadata = fs::metadata(p).map_err(|e| format!("failed to stat {path:?}: {e}"))?;

    if metadata.is_dir() {
        let mut files: Vec<PathBuf> = fs::read_dir(p)
            .map_err(|e| format!("failed to read directory {path:?}: {e}"))?
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .filter(|path| {
                path.is_file() && path.extension().and_then(|ext| ext.to_str()) == Some("txt")
            })
            .collect();
        files.sort();
        Ok(files)
    } else {
        Ok(vec![p.to_path_buf()])
    }
}

/// Resolve and concatenate all `--corpus` arguments (files and/or
/// directories) into a single corpus text, in argument order (directory
/// contents are sorted by filename). Files are joined with a blank line so
/// that a missing trailing newline in one file can't merge its last line
/// with the next file's first line.
fn load_corpus_paths(corpus_paths: &[String]) -> Vec<PathBuf> {
    let mut resolved = Vec::new();
    for path in corpus_paths {
        match expand_corpus_path(path) {
            Ok(files) => resolved.extend(files),
            Err(e) => {
                eprintln!("{e}");
                process::exit(1);
            }
        }
    }
    resolved
}

fn read_corpus_text(files: &[PathBuf]) -> String {
    let mut combined = String::new();
    for file in files {
        let text = fs::read_to_string(file).unwrap_or_else(|e| {
            eprintln!("failed to read corpus {file:?}: {e}");
            process::exit(1);
        });
        if !combined.is_empty() && !combined.ends_with('\n') {
            combined.push('\n');
        }
        combined.push_str(&text);
        combined.push('\n');
    }
    combined
}

fn main() {
    let args = parse_args();

    let corpus_files = load_corpus_paths(&args.corpus_paths);
    if corpus_files.is_empty() {
        eprintln!("no corpus files resolved from {:?}", args.corpus_paths);
        process::exit(1);
    }
    let corpus_text = read_corpus_text(&corpus_files);

    let instances = load_corpus(&corpus_text);
    if instances.is_empty() {
        eprintln!(
            "corpus produced zero training instances: {:?}",
            args.corpus_paths
        );
        process::exit(1);
    }

    let samples: Vec<Sample> = instances
        .iter()
        .map(|inst| Sample {
            features: inst.features.clone(),
            label: if inst.label { 1.0 } else { -1.0 },
        })
        .collect();

    eprintln!(
        "loaded {} training instances from {:?}",
        samples.len(),
        corpus_files
    );

    let model = train(&samples, args.iterations);
    let metrics = evaluate(&model, &samples);

    eprintln!("training complete ({} iterations)", args.iterations);
    eprintln!("features learned: {}", model.weights.len());
    eprintln!("accuracy:  {:.4}", metrics.accuracy);
    eprintln!("precision: {:.4}", metrics.precision);
    eprintln!("recall:    {:.4}", metrics.recall);

    write_model_json(&model, &args.output_path);
    eprintln!("model written to {:?}", args.output_path);

    if let Some(bin_output_path) = &args.bin_output_path {
        let training_data_hash = fnv1a_64(&corpus_text);
        let bytes = encode(model.bias, &model.weights, training_data_hash);
        fs::write(bin_output_path, &bytes).unwrap_or_else(|e| {
            eprintln!("failed to write binary model to {bin_output_path:?}: {e}");
            process::exit(1);
        });
        eprintln!(
            "binary model written to {:?} ({} bytes, training_data_hash={:#x})",
            bin_output_path,
            bytes.len(),
            training_data_hash
        );
    }
}

/// Serialize the model as `{"bias": f64, "weights": {"feature_key": weight, ...}}`.
fn write_model_json(model: &Model, output_path: &str) {
    let weights: HashMap<&str, f64> = model
        .weights
        .iter()
        .map(|(k, v)| (k.as_str(), *v))
        .collect();
    let json = serde_json::json!({
        "bias": model.bias,
        "weights": weights,
    });
    let serialized = serde_json::to_string_pretty(&json).unwrap_or_else(|e| {
        eprintln!("failed to serialize model: {e}");
        process::exit(1);
    });
    fs::write(output_path, serialized).unwrap_or_else(|e| {
        eprintln!("failed to write model to {output_path:?}: {e}");
        process::exit(1);
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    /// Creates a unique scratch directory under the OS temp dir for a test,
    /// named after the test (via `label`) to avoid collisions when tests run
    /// in parallel.
    fn scratch_dir(label: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "train_segmenter_test_{label}_{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("create scratch dir");
        dir
    }

    #[test]
    fn expand_corpus_path_single_file_returns_itself() {
        let dir = scratch_dir("single_file");
        let file = dir.join("corpus.txt");
        fs::write(&file, "猫 が 好き\n").unwrap();

        let files = expand_corpus_path(file.to_str().unwrap()).unwrap();
        assert_eq!(files, vec![file]);
    }

    #[test]
    fn expand_corpus_path_directory_returns_sorted_txt_files() {
        let dir = scratch_dir("dir_expand");
        fs::write(dir.join("b_corpus.txt"), "犬 が 好き\n").unwrap();
        fs::write(dir.join("a_corpus.txt"), "猫 が 好き\n").unwrap();
        fs::write(dir.join("notes.md"), "# not a corpus\n").unwrap();

        let files = expand_corpus_path(dir.to_str().unwrap()).unwrap();
        let names: Vec<String> = files
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().to_string())
            .collect();
        assert_eq!(names, vec!["a_corpus.txt", "b_corpus.txt"]);
    }

    #[test]
    fn expand_corpus_path_missing_path_is_an_error() {
        let result = expand_corpus_path("/nonexistent/path/does-not-exist.txt");
        assert!(result.is_err());
    }

    #[test]
    fn load_corpus_paths_concatenates_multiple_files_in_order() {
        let dir = scratch_dir("multi_file");
        let first = dir.join("first.txt");
        let second = dir.join("second.txt");
        fs::write(&first, "猫 が 好き\n").unwrap();
        fs::write(&second, "犬 が 好き\n").unwrap();

        let files = load_corpus_paths(&[
            first.to_str().unwrap().to_string(),
            second.to_str().unwrap().to_string(),
        ]);
        let text = read_corpus_text(&files);

        let cat_idx = text.find("猫").expect("first file content present");
        let dog_idx = text.find("犬").expect("second file content present");
        assert!(
            cat_idx < dog_idx,
            "expected first.txt content before second.txt content"
        );
    }

    #[test]
    fn read_corpus_text_joins_files_missing_trailing_newline_safely() {
        let dir = scratch_dir("no_trailing_newline");
        let first = dir.join("first.txt");
        let second = dir.join("second.txt");
        // Intentionally no trailing newline on the first file.
        fs::write(&first, "猫 が 好き").unwrap();
        fs::write(&second, "犬 が 好き\n").unwrap();

        let text = read_corpus_text(&[first, second]);

        // The last line of `first` must not be glued to the first line of
        // `second` (which would corrupt word-boundary training instances).
        assert!(
            !text.contains("好き犬"),
            "files must not be concatenated without a newline separator: {text:?}"
        );
    }

    #[test]
    fn load_corpus_paths_expands_directory_argument() {
        let dir = scratch_dir("dir_arg");
        fs::write(dir.join("one.txt"), "猫 が 好き\n").unwrap();
        fs::write(dir.join("two.txt"), "犬 が 好き\n").unwrap();

        let files = load_corpus_paths(&[dir.to_str().unwrap().to_string()]);
        assert_eq!(files.len(), 2);
    }
}
