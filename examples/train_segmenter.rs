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

use std::collections::HashMap;
use std::fs;
use std::process;

use lexsim::segmenter::adaboost::{evaluate, train, Model, Sample};
use lexsim::segmenter::binary::{encode, fnv1a_64};
use lexsim::segmenter::corpus::load_corpus;

struct Args {
    corpus_path: String,
    output_path: String,
    bin_output_path: Option<String>,
    iterations: usize,
}

fn parse_args() -> Args {
    let mut corpus_path = "training/seed_corpus.txt".to_string();
    let mut output_path = "training/model.json".to_string();
    let mut bin_output_path: Option<String> = None;
    let mut iterations = 1000usize;

    let mut argv = std::env::args().skip(1);
    while let Some(arg) = argv.next() {
        match arg.as_str() {
            "--corpus" => {
                corpus_path = argv.next().unwrap_or_else(|| {
                    eprintln!("--corpus requires a value");
                    process::exit(2);
                });
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

    Args {
        corpus_path,
        output_path,
        bin_output_path,
        iterations,
    }
}

fn main() {
    let args = parse_args();

    let corpus_text = fs::read_to_string(&args.corpus_path).unwrap_or_else(|e| {
        eprintln!("failed to read corpus {:?}: {e}", args.corpus_path);
        process::exit(1);
    });

    let instances = load_corpus(&corpus_text);
    if instances.is_empty() {
        eprintln!(
            "corpus produced zero training instances: {:?}",
            args.corpus_path
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
        args.corpus_path
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
