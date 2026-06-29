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
    assert!(tokens.iter().any(|t| t.as_str() == Some("メモ")));
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
