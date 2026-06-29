use std::io::{self, Read};
use std::process;

use serde::Serialize;
use serde_json::Value;

use lexsim::{content_hash, jaccard, tokenize, Corpus};

fn main() {
    let subcommand = match std::env::args().nth(1) {
        Some(s) => s,
        None => {
            print_usage();
            process::exit(2);
        }
    };

    match subcommand.as_str() {
        "tokenize" => run(cmd_tokenize),
        "jaccard" => run(cmd_jaccard),
        "bm25" => run(cmd_bm25),
        "hash" => run(cmd_hash),
        _ => {
            print_usage();
            process::exit(2);
        }
    }
}

fn run(handler: fn(&Value) -> Result<Value, String>) {
    let mut input = String::new();
    if let Err(e) = io::stdin().read_to_string(&mut input) {
        let err = serde_json::json!({"error": format!("failed to read stdin: {e}")});
        println!("{}", serde_json::to_string(&err).unwrap());
        process::exit(1);
    }

    let parsed: Value = match serde_json::from_str(&input) {
        Ok(v) => v,
        Err(e) => {
            let err = serde_json::json!({"error": format!("invalid JSON: {e}")});
            println!("{}", serde_json::to_string(&err).unwrap());
            process::exit(1);
        }
    };

    match handler(&parsed) {
        Ok(output) => {
            println!("{}", serde_json::to_string(&output).unwrap());
        }
        Err(msg) => {
            let err = serde_json::json!({"error": msg});
            println!("{}", serde_json::to_string(&err).unwrap());
            process::exit(1);
        }
    }
}

fn cmd_tokenize(input: &Value) -> Result<Value, String> {
    let texts = input
        .get("texts")
        .and_then(|v| v.as_array())
        .ok_or("missing or invalid \"texts\" array")?;

    let results: Vec<TokenizeResult> = texts
        .iter()
        .map(|v| {
            let text = v.as_str().unwrap_or("");
            let tokens = tokenize(text);
            let count = tokens.len();
            TokenizeResult { tokens, count }
        })
        .collect();

    Ok(serde_json::to_value(TokenizeOutput { results }).unwrap())
}

fn cmd_jaccard(input: &Value) -> Result<Value, String> {
    if let Some(pairs) = input.get("pairs").and_then(|v| v.as_array()) {
        let scores: Vec<f64> = pairs
            .iter()
            .map(|p| {
                let a = p.get("a").and_then(|v| v.as_str()).unwrap_or("");
                let b = p.get("b").and_then(|v| v.as_str()).unwrap_or("");
                round6(jaccard(a, b))
            })
            .collect();
        Ok(serde_json::json!({"scores": scores}))
    } else if input.get("a").is_some() && input.get("b").is_some() {
        let a = input["a"].as_str().unwrap_or("");
        let b = input["b"].as_str().unwrap_or("");
        let score = round6(jaccard(a, b));
        Ok(serde_json::json!({"score": score}))
    } else {
        Err("expected \"a\" and \"b\" fields, or a \"pairs\" array".to_string())
    }
}

fn cmd_bm25(input: &Value) -> Result<Value, String> {
    let corpus_arr = input
        .get("corpus")
        .and_then(|v| v.as_array())
        .ok_or("missing or invalid \"corpus\" array")?;
    let query = input
        .get("query")
        .and_then(|v| v.as_str())
        .ok_or("missing or invalid \"query\" string")?;

    let docs: Vec<String> = corpus_arr
        .iter()
        .map(|v| v.as_str().unwrap_or("").to_string())
        .collect();

    let corpus = Corpus::build(&docs);
    let scores: Vec<f64> = corpus.bm25_scores(query).into_iter().map(round6).collect();

    Ok(serde_json::json!({"scores": scores}))
}

fn cmd_hash(input: &Value) -> Result<Value, String> {
    let texts = input
        .get("texts")
        .and_then(|v| v.as_array())
        .ok_or("missing or invalid \"texts\" array")?;

    let hashes: Vec<String> = texts
        .iter()
        .map(|v| content_hash(v.as_str().unwrap_or("")))
        .collect();

    Ok(serde_json::json!({"hashes": hashes}))
}

fn round6(v: f64) -> f64 {
    (v * 1_000_000.0).round() / 1_000_000.0
}

fn print_usage() {
    eprintln!("Usage: lexsim <tokenize|jaccard|bm25|hash>");
    eprintln!();
    eprintln!("Subcommands:");
    eprintln!("  tokenize  Tokenize texts (stdin JSON → stdout JSON)");
    eprintln!("  jaccard   Compute Jaccard similarity (stdin JSON → stdout JSON)");
    eprintln!("  bm25      Compute BM25 scores (stdin JSON → stdout JSON)");
    eprintln!("  hash      Compute content hashes (stdin JSON → stdout JSON)");
}

#[derive(Serialize)]
struct TokenizeResult {
    tokens: Vec<String>,
    count: usize,
}

#[derive(Serialize)]
struct TokenizeOutput {
    results: Vec<TokenizeResult>,
}
