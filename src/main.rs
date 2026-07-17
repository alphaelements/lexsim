use std::io::{self, Read};
use std::process;

use serde_json::Value;

use lexsim::{
    analyze_sentiment, content_hash, corpus_diff, jaccard, textrank_keywords, tokenize,
    tokenize_ngrams, Corpus,
};

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
        "keywords" => run(cmd_keywords),
        "diff" => run(cmd_diff),
        "sentiment" => run(cmd_sentiment),
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

    let ngram = input.get("ngram").and_then(|v| v.as_u64()).unwrap_or(1) as usize;

    if ngram == 0 || ngram > 3 {
        return Err("\"ngram\" must be 1, 2, or 3".to_string());
    }

    let tokens: Vec<Vec<String>> = texts
        .iter()
        .map(|v| {
            let text = v.as_str().unwrap_or("");
            if ngram <= 1 {
                tokenize(text)
            } else {
                tokenize_ngrams(text, ngram)
            }
        })
        .collect();

    Ok(serde_json::json!({"tokens": tokens}))
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

    let weighted = match input.get("weighted") {
        None => false,
        Some(Value::Bool(b)) => *b,
        Some(_) => return Err("\"weighted\" must be a boolean".to_string()),
    };

    let scores: Vec<f64> = if weighted {
        Corpus::build_weighted(&docs).bm25_scores_weighted(query)
    } else {
        Corpus::build(&docs).bm25_scores(query)
    }
    .into_iter()
    .map(round6)
    .collect();

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

/// Default word-window size for context-aware extraction methods
/// (co-occurrence / TextRank).
const DEFAULT_WINDOW_SIZE: usize = 4;

fn cmd_keywords(input: &Value) -> Result<Value, String> {
    let texts = input
        .get("texts")
        .and_then(|v| v.as_array())
        .ok_or("missing or invalid \"texts\" array")?;

    let top_n = input.get("top_n").and_then(|v| v.as_u64()).unwrap_or(20) as usize;
    let method = input
        .get("method")
        .and_then(|v| v.as_str())
        .unwrap_or("tfidf");
    let window_size = input
        .get("window_size")
        .and_then(|v| v.as_u64())
        .unwrap_or(DEFAULT_WINDOW_SIZE as u64) as usize;

    let docs: Vec<String> = texts
        .iter()
        .map(|v| v.as_str().unwrap_or("").to_string())
        .collect();

    let entries = match method {
        "tfidf" => Corpus::build(&docs).tfidf_keywords(top_n),
        "textrank" => {
            let combined = docs.join("\n");
            textrank_keywords(&combined, window_size, top_n)
        }
        "cooccurrence" => {
            let query = input
                .get("query")
                .and_then(|v| v.as_str())
                .ok_or("\"cooccurrence\" method requires a \"query\" string")?;
            Corpus::build(&docs).cooccurrence_keywords(query, window_size, top_n)
        }
        other => {
            return Err(format!(
            "unknown \"method\" {other:?}: expected \"tfidf\", \"textrank\", or \"cooccurrence\""
        ))
        }
    };

    let keywords: Vec<Value> = entries
        .into_iter()
        .map(|e| {
            serde_json::json!({
                "keyword": e.keyword,
                "score": round6(e.score),
                "count": e.count,
            })
        })
        .collect();

    Ok(serde_json::json!({"keywords": keywords}))
}

fn cmd_diff(input: &Value) -> Result<Value, String> {
    let corpus_a = input
        .get("corpus_a")
        .and_then(|v| v.as_array())
        .ok_or("missing or invalid \"corpus_a\" array")?;
    let corpus_b = input
        .get("corpus_b")
        .and_then(|v| v.as_array())
        .ok_or("missing or invalid \"corpus_b\" array")?;

    let top_n = input.get("top_n").and_then(|v| v.as_u64()).unwrap_or(20) as usize;

    let docs_a: Vec<String> = corpus_a
        .iter()
        .map(|v| v.as_str().unwrap_or("").to_string())
        .collect();
    let docs_b: Vec<String> = corpus_b
        .iter()
        .map(|v| v.as_str().unwrap_or("").to_string())
        .collect();

    let (a_dist, b_dist) = corpus_diff(&docs_a, &docs_b, top_n);

    let a_json: Vec<Value> = a_dist
        .into_iter()
        .map(|e| serde_json::json!({"keyword": e.keyword, "ratio": round6(e.ratio)}))
        .collect();
    let b_json: Vec<Value> = b_dist
        .into_iter()
        .map(|e| serde_json::json!({"keyword": e.keyword, "ratio": round6(e.ratio)}))
        .collect();

    Ok(serde_json::json!({
        "a_distinctive": a_json,
        "b_distinctive": b_json,
    }))
}

fn cmd_sentiment(input: &Value) -> Result<Value, String> {
    let texts = input
        .get("texts")
        .and_then(|v| v.as_array())
        .ok_or("missing or invalid \"texts\" array")?;

    let results: Vec<Value> = texts
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let text = v.as_str().unwrap_or("");
            let r = analyze_sentiment(text);
            serde_json::json!({
                "text_index": i,
                "polarity": r.polarity.as_str(),
                "confidence": round6(r.confidence),
            })
        })
        .collect();

    Ok(serde_json::json!({"results": results}))
}

fn round6(v: f64) -> f64 {
    (v * 1_000_000.0).round() / 1_000_000.0
}

fn print_usage() {
    eprintln!("Usage: lexsim <subcommand>");
    eprintln!();
    eprintln!("Subcommands:");
    eprintln!("  tokenize   Tokenize texts (stdin JSON → stdout JSON)");
    eprintln!("  jaccard    Compute Jaccard similarity (stdin JSON → stdout JSON)");
    eprintln!(
        "  bm25       Compute BM25 scores (stdin JSON → stdout JSON); \"weighted\": true for particle-context weighted scoring"
    );
    eprintln!("  hash       Compute content hashes (stdin JSON → stdout JSON)");
    eprintln!(
        "  keywords   Extract top-N keywords (stdin JSON → stdout JSON); \"method\": \"tfidf\" (default) | \"textrank\" | \"cooccurrence\" (requires \"query\")"
    );
    eprintln!(
        "  diff       Compare two corpora for distinctive keywords (stdin JSON → stdout JSON)"
    );
    eprintln!(
        "  sentiment  Classify text sentiment as positive/neutral/negative (stdin JSON → stdout JSON)"
    );
}
