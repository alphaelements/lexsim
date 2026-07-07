#![cfg(feature = "cli")]

use std::process::Command;

fn lexsim_bin() -> String {
    let mut path = std::env::current_exe()
        .unwrap()
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();
    path.push("lexsim");
    path.to_string_lossy().to_string()
}

fn run_cli(subcommand: &str, stdin: &str) -> (String, String, i32) {
    let output = Command::new(lexsim_bin())
        .arg(subcommand)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .and_then(|mut child| {
            use std::io::Write;
            child
                .stdin
                .take()
                .unwrap()
                .write_all(stdin.as_bytes())
                .unwrap();
            child.wait_with_output()
        })
        .expect("failed to run lexsim binary");

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    let code = output.status.code().unwrap_or(-1);
    (stdout, stderr, code)
}

fn run_cli_no_args() -> (String, String, i32) {
    let output = Command::new(lexsim_bin())
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .and_then(|child| child.wait_with_output())
        .expect("failed to run lexsim binary");

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    let code = output.status.code().unwrap_or(-1);
    (stdout, stderr, code)
}

// ── tokenize ──

#[test]
fn tokenize_basic() {
    let input = r#"{"texts": ["hello world"]}"#;
    let (stdout, _, code) = run_cli("tokenize", input);
    assert_eq!(code, 0, "exit code should be 0");
    let v: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    let tokens_outer = v["tokens"].as_array().unwrap();
    assert_eq!(tokens_outer.len(), 1);
    let tokens = tokens_outer[0].as_array().unwrap();
    assert!(tokens.len() > 0);
    assert!(tokens.iter().any(|t| t.as_str() == Some("hello")));
    assert!(tokens.iter().any(|t| t.as_str() == Some("world")));
}

#[test]
fn tokenize_batch_preserves_order() {
    let input = r#"{"texts": ["aaa", "bbb", "ccc"]}"#;
    let (stdout, _, code) = run_cli("tokenize", input);
    assert_eq!(code, 0);
    let v: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    let tokens = v["tokens"].as_array().unwrap();
    assert_eq!(tokens.len(), 3);
}

#[test]
fn tokenize_japanese() {
    let input = r#"{"texts": ["メモリ機能"]}"#;
    let (stdout, _, code) = run_cli("tokenize", input);
    assert_eq!(code, 0);
    let v: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    let tokens = v["tokens"][0].as_array().unwrap();
    assert!(tokens.iter().any(|t| t.as_str() == Some("メモリ")));
}

#[test]
fn tokenize_empty_texts() {
    let input = r#"{"texts": []}"#;
    let (stdout, _, code) = run_cli("tokenize", input);
    assert_eq!(code, 0);
    let v: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    assert_eq!(v["tokens"].as_array().unwrap().len(), 0);
}

// ── jaccard ──

#[test]
fn jaccard_single_pair() {
    let input = r#"{"a": "hello world", "b": "hello world"}"#;
    let (stdout, _, code) = run_cli("jaccard", input);
    assert_eq!(code, 0);
    let v: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    let score = v["score"].as_f64().unwrap();
    assert!(
        (score - 1.0).abs() < 1e-6,
        "identical texts should have score 1.0, got {score}"
    );
}

#[test]
fn jaccard_batch() {
    let input = r#"{"pairs": [{"a": "hello", "b": "hello"}, {"a": "aaa", "b": "zzz"}]}"#;
    let (stdout, _, code) = run_cli("jaccard", input);
    assert_eq!(code, 0);
    let v: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    let scores = v["scores"].as_array().unwrap();
    assert_eq!(scores.len(), 2);
    assert!(scores[0].as_f64().unwrap() > scores[1].as_f64().unwrap());
}

#[test]
fn jaccard_missing_fields() {
    let input = r#"{"x": "hello"}"#;
    let (stdout, _, code) = run_cli("jaccard", input);
    assert_eq!(code, 1);
    let v: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    assert!(v["error"].as_str().is_some());
}

// ── bm25 ──

#[test]
fn bm25_basic() {
    let input = r#"{
        "corpus": ["always use atomic_write for JSON", "the cat sat on the mat"],
        "query": "atomic write json"
    }"#;
    let (stdout, _, code) = run_cli("bm25", input);
    assert_eq!(code, 0);
    let v: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    let scores = v["scores"].as_array().unwrap();
    assert_eq!(scores.len(), 2);
    assert!(scores[0].as_f64().unwrap() > scores[1].as_f64().unwrap());
}

#[test]
fn bm25_missing_query() {
    let input = r#"{"corpus": ["hello"]}"#;
    let (stdout, _, code) = run_cli("bm25", input);
    assert_eq!(code, 1);
    let v: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    assert!(v["error"].as_str().unwrap().contains("query"));
}

#[test]
fn bm25_missing_corpus() {
    let input = r#"{"query": "hello"}"#;
    let (stdout, _, code) = run_cli("bm25", input);
    assert_eq!(code, 1);
    let v: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    assert!(v["error"].as_str().unwrap().contains("corpus"));
}

// ── hash ──

#[test]
fn hash_basic() {
    let input = r#"{"texts": ["hello world", "Hello World", "hello  world"]}"#;
    let (stdout, _, code) = run_cli("hash", input);
    assert_eq!(code, 0);
    let v: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    let hashes = v["hashes"].as_array().unwrap();
    assert_eq!(hashes.len(), 3);
    // All three should produce the same hash (case/whitespace insensitive)
    assert_eq!(hashes[0], hashes[1]);
    assert_eq!(hashes[1], hashes[2]);
    // 16-char hex
    let h = hashes[0].as_str().unwrap();
    assert_eq!(h.len(), 16);
    assert!(h.chars().all(|c| c.is_ascii_hexdigit()));
}

#[test]
fn hash_different_content_different_hash() {
    let input = r#"{"texts": ["hello world", "goodbye world"]}"#;
    let (stdout, _, code) = run_cli("hash", input);
    assert_eq!(code, 0);
    let v: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    let hashes = v["hashes"].as_array().unwrap();
    assert_ne!(hashes[0], hashes[1]);
}

// ── error handling ──

#[test]
fn invalid_json_returns_error() {
    let (stdout, _, code) = run_cli("tokenize", "not json");
    assert_eq!(code, 1);
    let v: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    assert!(v["error"].as_str().unwrap().contains("invalid JSON"));
}

#[test]
fn unknown_subcommand_exits_2() {
    let (_, stderr, code) = run_cli("unknown", "{}");
    assert_eq!(code, 2);
    assert!(stderr.contains("Usage:"));
}

#[test]
fn no_subcommand_exits_2() {
    let (_, stderr, code) = run_cli_no_args();
    assert_eq!(code, 2);
    assert!(stderr.contains("Usage:"));
}

// ── edge cases ──

#[test]
fn tokenize_mixed_language() {
    let input = r#"{"texts": ["atomic_write を使う"]}"#;
    let (stdout, _, code) = run_cli("tokenize", input);
    assert_eq!(code, 0);
    let v: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    let tokens = v["tokens"][0].as_array().unwrap();
    assert!(tokens.iter().any(|t| t.as_str() == Some("atomic")));
    assert!(tokens.iter().any(|t| t.as_str() == Some("write")));
}

#[test]
fn bm25_empty_corpus() {
    let input = r#"{"corpus": [], "query": "hello"}"#;
    let (stdout, _, code) = run_cli("bm25", input);
    assert_eq!(code, 0);
    let v: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    assert_eq!(v["scores"].as_array().unwrap().len(), 0);
}

#[test]
fn jaccard_empty_texts() {
    let input = r#"{"a": "", "b": ""}"#;
    let (stdout, _, code) = run_cli("jaccard", input);
    assert_eq!(code, 0);
    let v: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    let score = v["score"].as_f64().unwrap();
    assert!((score - 1.0).abs() < 1e-6, "empty vs empty should be 1.0");
}

#[test]
fn hash_empty_text() {
    let input = r#"{"texts": [""]}"#;
    let (stdout, _, code) = run_cli("hash", input);
    assert_eq!(code, 0);
    let v: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    let h = v["hashes"][0].as_str().unwrap();
    assert_eq!(h.len(), 16);
}

#[test]
fn tokenize_returns_no_results_key() {
    let input = r#"{"texts": ["hello world foo bar"]}"#;
    let (stdout, _, code) = run_cli("tokenize", input);
    assert_eq!(code, 0);
    let v: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    assert!(v.get("results").is_none(), "should not have 'results' key");
    assert!(v.get("tokens").is_some(), "should have 'tokens' key");
    let tokens = v["tokens"][0].as_array().unwrap();
    assert!(tokens.len() > 0);
}

// ── keywords ──

#[test]
fn keywords_basic() {
    let input =
        r#"{"texts": ["rust programming", "rust systems programming", "rust memory safety"]}"#;
    let (stdout, _, code) = run_cli("keywords", input);
    assert_eq!(code, 0);
    let v: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    let keywords = v["keywords"].as_array().unwrap();
    assert!(!keywords.is_empty());
    let first = &keywords[0];
    assert!(first["keyword"].as_str().is_some());
    assert!(first["score"].as_f64().unwrap() > 0.0);
    assert!(first["count"].as_u64().unwrap() > 0);
    for kw in keywords {
        let k = kw["keyword"].as_str().unwrap();
        assert!(
            !k.starts_with('\u{1}'),
            "CL-CnG token leaked into keywords output: {k:?}"
        );
    }
}

#[test]
fn keywords_single_document() {
    let input = r#"{"texts": ["Rust is a systems programming language"], "top_n": 5}"#;
    let (stdout, _, code) = run_cli("keywords", input);
    assert_eq!(code, 0);
    let v: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    let keywords = v["keywords"].as_array().unwrap();
    assert!(
        !keywords.is_empty(),
        "single-doc corpus must still return keywords"
    );
}

#[test]
fn keywords_top_n() {
    let input = r#"{"texts": ["a b c d e f g h i j"], "top_n": 3}"#;
    let (stdout, _, code) = run_cli("keywords", input);
    assert_eq!(code, 0);
    let v: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    let keywords = v["keywords"].as_array().unwrap();
    assert!(keywords.len() <= 3);
}

#[test]
fn keywords_empty_texts() {
    let input = r#"{"texts": []}"#;
    let (stdout, _, code) = run_cli("keywords", input);
    assert_eq!(code, 0);
    let v: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    assert_eq!(v["keywords"].as_array().unwrap().len(), 0);
}

#[test]
fn keywords_japanese() {
    let input = r#"{"texts": ["メモリ機能はセッション間で教訓を引き継ぐ", "メモリ機能で過去の学びを保持する"], "top_n": 5}"#;
    let (stdout, _, code) = run_cli("keywords", input);
    assert_eq!(code, 0);
    let v: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    let keywords = v["keywords"].as_array().unwrap();
    assert!(!keywords.is_empty());
    // Regression: particle-glued bigrams (の/は/を/で) must not leak into
    // keyword output at the CLI-integration layer, not just the unit layer.
    for kw in keywords {
        let word = kw["keyword"].as_str().unwrap();
        for particle in ["の", "は", "を", "で"] {
            assert!(
                !word.contains(particle),
                "keyword {word:?} unexpectedly contains particle {particle:?}"
            );
        }
    }
}

#[test]
fn keywords_japanese_excludes_aux_glued_bigrams() {
    // Regression test for the rework finding: auxiliary-verb fragments
    // (した, from 降りました) must not leak into keyword output alongside
    // genuine content words at the CLI-integration layer.
    let input = r#"{"texts": ["昨日は雨が降りました", "今日も雨が降りました"], "top_n": 10}"#;
    let (stdout, _, code) = run_cli("keywords", input);
    assert_eq!(code, 0);
    let v: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    let keywords = v["keywords"].as_array().unwrap();
    assert!(!keywords.iter().any(|k| k["keyword"] == "した"));
    assert!(keywords.iter().any(|k| k["keyword"] == "昨日"));
    assert!(keywords.iter().any(|k| k["keyword"] == "今日"));
}

#[test]
fn keywords_missing_texts() {
    let input = r#"{"top_n": 5}"#;
    let (stdout, _, code) = run_cli("keywords", input);
    assert_eq!(code, 1);
    let v: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    assert!(v["error"].as_str().unwrap().contains("texts"));
}

// ── diff ──

#[test]
fn diff_basic() {
    let input = r#"{
        "corpus_a": ["rust systems programming", "rust memory safety"],
        "corpus_b": ["python data science", "python machine learning"]
    }"#;
    let (stdout, _, code) = run_cli("diff", input);
    assert_eq!(code, 0);
    let v: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    let a_dist = v["a_distinctive"].as_array().unwrap();
    let b_dist = v["b_distinctive"].as_array().unwrap();
    assert!(!a_dist.is_empty());
    assert!(!b_dist.is_empty());
    assert!(a_dist[0]["ratio"].as_f64().unwrap() > 1.0);
}

#[test]
fn diff_identical_corpora() {
    let input = r#"{
        "corpus_a": ["hello world"],
        "corpus_b": ["hello world"]
    }"#;
    let (stdout, _, code) = run_cli("diff", input);
    assert_eq!(code, 0);
    let v: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    assert!(v["a_distinctive"].as_array().unwrap().is_empty());
    assert!(v["b_distinctive"].as_array().unwrap().is_empty());
}

#[test]
fn diff_with_top_n() {
    let input = r#"{
        "corpus_a": ["rust rust rust systems programming language compiler"],
        "corpus_b": ["python python python data science machine learning"],
        "top_n": 2
    }"#;
    let (stdout, _, code) = run_cli("diff", input);
    assert_eq!(code, 0);
    let v: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    assert!(v["a_distinctive"].as_array().unwrap().len() <= 2);
    assert!(v["b_distinctive"].as_array().unwrap().len() <= 2);
}

#[test]
fn diff_missing_corpus_a() {
    let input = r#"{"corpus_b": ["hello"]}"#;
    let (stdout, _, code) = run_cli("diff", input);
    assert_eq!(code, 1);
    let v: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    assert!(v["error"].as_str().unwrap().contains("corpus_a"));
}

#[test]
fn diff_missing_corpus_b() {
    let input = r#"{"corpus_a": ["hello"]}"#;
    let (stdout, _, code) = run_cli("diff", input);
    assert_eq!(code, 1);
    let v: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    assert!(v["error"].as_str().unwrap().contains("corpus_b"));
}

// ── tokenize n-gram ──

#[test]
fn tokenize_bigram() {
    let input = r#"{"texts": ["hello world foo"], "ngram": 2}"#;
    let (stdout, _, code) = run_cli("tokenize", input);
    assert_eq!(code, 0);
    let v: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    let tokens = v["tokens"][0].as_array().unwrap();
    assert!(tokens.iter().any(|t| t.as_str() == Some("hello world")));
    assert!(tokens.iter().any(|t| t.as_str() == Some("hello")));
}

#[test]
fn tokenize_trigram() {
    let input = r#"{"texts": ["hello world foo bar"], "ngram": 3}"#;
    let (stdout, _, code) = run_cli("tokenize", input);
    assert_eq!(code, 0);
    let v: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    let tokens = v["tokens"][0].as_array().unwrap();
    assert!(tokens.iter().any(|t| t.as_str() == Some("hello world foo")));
}

#[test]
fn tokenize_ngram_default_is_unigram() {
    let input_no_ngram = r#"{"texts": ["hello world"]}"#;
    let input_ngram_1 = r#"{"texts": ["hello world"], "ngram": 1}"#;
    let (out1, _, _) = run_cli("tokenize", input_no_ngram);
    let (out2, _, _) = run_cli("tokenize", input_ngram_1);
    assert_eq!(out1, out2);
}

#[test]
fn tokenize_ngram_invalid_value() {
    let input = r#"{"texts": ["hello"], "ngram": 5}"#;
    let (stdout, _, code) = run_cli("tokenize", input);
    assert_eq!(code, 1);
    let v: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    assert!(v["error"].as_str().unwrap().contains("ngram"));
}

// ── sentiment ──

#[test]
fn sentiment_positive_english() {
    let input = r#"{"texts": ["This is great and amazing work"]}"#;
    let (stdout, _, code) = run_cli("sentiment", input);
    assert_eq!(code, 0);
    let v: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    let results = v["results"].as_array().unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0]["text_index"].as_u64().unwrap(), 0);
    assert_eq!(results[0]["polarity"].as_str().unwrap(), "positive");
    assert!(results[0]["confidence"].as_f64().unwrap() > 0.5);
}

#[test]
fn sentiment_negative_english() {
    let input = r#"{"texts": ["This is terrible and horrible"]}"#;
    let (stdout, _, code) = run_cli("sentiment", input);
    assert_eq!(code, 0);
    let v: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    let results = v["results"].as_array().unwrap();
    assert_eq!(results[0]["polarity"].as_str().unwrap(), "negative");
}

#[test]
fn sentiment_neutral() {
    let input = r#"{"texts": ["The function returns a value"]}"#;
    let (stdout, _, code) = run_cli("sentiment", input);
    assert_eq!(code, 0);
    let v: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    let results = v["results"].as_array().unwrap();
    assert_eq!(results[0]["polarity"].as_str().unwrap(), "neutral");
}

#[test]
fn sentiment_japanese_positive() {
    let input = r#"{"texts": ["素晴らしい機能で便利です"]}"#;
    let (stdout, _, code) = run_cli("sentiment", input);
    assert_eq!(code, 0);
    let v: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    let results = v["results"].as_array().unwrap();
    assert_eq!(results[0]["polarity"].as_str().unwrap(), "positive");
}

#[test]
fn sentiment_japanese_negative() {
    let input = r#"{"texts": ["バグが多くて不安定です"]}"#;
    let (stdout, _, code) = run_cli("sentiment", input);
    assert_eq!(code, 0);
    let v: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    let results = v["results"].as_array().unwrap();
    assert_eq!(results[0]["polarity"].as_str().unwrap(), "negative");
}

#[test]
fn sentiment_batch() {
    let input = r#"{"texts": ["great work", "terrible bug", "the code runs"]}"#;
    let (stdout, _, code) = run_cli("sentiment", input);
    assert_eq!(code, 0);
    let v: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    let results = v["results"].as_array().unwrap();
    assert_eq!(results.len(), 3);
    assert_eq!(results[0]["text_index"].as_u64().unwrap(), 0);
    assert_eq!(results[1]["text_index"].as_u64().unwrap(), 1);
    assert_eq!(results[2]["text_index"].as_u64().unwrap(), 2);
}

#[test]
fn sentiment_empty_texts() {
    let input = r#"{"texts": []}"#;
    let (stdout, _, code) = run_cli("sentiment", input);
    assert_eq!(code, 0);
    let v: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    assert_eq!(v["results"].as_array().unwrap().len(), 0);
}

#[test]
fn sentiment_missing_texts() {
    let input = r#"{}"#;
    let (stdout, _, code) = run_cli("sentiment", input);
    assert_eq!(code, 1);
    let v: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    assert!(v["error"].as_str().unwrap().contains("texts"));
}

#[test]
fn tokenize_non_string_element_treated_as_empty() {
    let input = r#"{"texts": [123, null, true]}"#;
    let (stdout, _, code) = run_cli("tokenize", input);
    assert_eq!(code, 0);
    let v: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    let tokens = v["tokens"].as_array().unwrap();
    assert_eq!(tokens.len(), 3);
    for t in tokens {
        assert!(t.as_array().unwrap().is_empty());
    }
}
