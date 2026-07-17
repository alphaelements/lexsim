//! Integration tests for particle-context weighted BM25, exercised through
//! the public crate API exactly as a downstream consumer (handoff-mcp's
//! memory injection) would use it: `Corpus::build_weighted` over a realistic
//! memory corpus + `bm25_scores_weighted` over natural-language queries.

use lexsim::{tokenize_weighted, Corpus, TOPIC_BOOST};

/// A realistic handoff-mcp memory corpus: short Japanese/mixed notes with
/// embedded identifiers, of the kind `memory_query` scores.
fn memory_corpus() -> Vec<String> {
    vec![
        // 0: atomic_write convention
        "ファイル書き込みは必ず atomic_write を使う。torn read を防止するため".to_string(),
        // 1: memory injection criteria
        "メモリ注入の基準はスコア上位5件、閾値 0.35 以上のみ注入する".to_string(),
        // 2: gantt chart config
        "ガントチャートの表示設定は config.toml の gantt セクションで変更する".to_string(),
        // 3: release procedure
        "リリースは手動で cargo publish を実行し、annotated tag を打つ".to_string(),
        // 4: generic short note (stopword-heavy)
        "この設定はとても便利です。また、これも使うことがあります".to_string(),
    ]
}

fn argmax(scores: &[f64]) -> usize {
    scores
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap()
}

#[test]
fn identifier_query_ranks_identifier_memory_first() {
    // Spec scenario 1: 「atomic_write のフックについて教えて」→ the
    // atomic_write memory must rank first.
    let docs = memory_corpus();
    let corpus = Corpus::build_weighted(&docs);
    let scores = corpus.bm25_scores_weighted("atomic_write のフックについて教えて");
    assert_eq!(argmax(&scores), 0, "scores: {scores:?}");
}

#[test]
fn topic_query_ranks_topic_memory_first() {
    // Spec scenario 2: 「メモリ注入の基準は？」→ the injection-criteria
    // memory must rank first, and unrelated memories must not score via
    // particles or trigram noise.
    let docs = memory_corpus();
    let corpus = Corpus::build_weighted(&docs);
    let scores = corpus.bm25_scores_weighted("メモリ注入の基準は？");
    assert_eq!(argmax(&scores), 1, "scores: {scores:?}");
}

#[test]
fn stopword_heavy_query_does_not_hit_stopword_heavy_memory() {
    // A query that is mostly particles/demonstratives must not surface the
    // stopword-heavy memory (doc 4) just because it shares function words.
    let docs = memory_corpus();
    let corpus = Corpus::build_weighted(&docs);
    let scores = corpus.bm25_scores_weighted("これはとても重要ですか？");
    assert_eq!(
        scores[4], 0.0,
        "stopword overlap alone must score zero: {scores:?}"
    );
}

#[test]
fn weighted_reduces_noise_relative_to_plain_bm25() {
    // The motivating defect: plain BM25 lets CL-CnG trigrams and particles
    // give a nonzero score to unrelated memories. For a query sharing only
    // function words and trigram fragments with doc 4, plain BM25 scores it
    // above zero while weighted BM25 scores it exactly zero.
    let docs = memory_corpus();
    let plain = Corpus::build(&docs);
    let weighted = Corpus::build_weighted(&docs);
    let query = "これはとても重要ですか？";
    assert!(
        plain.bm25_scores(query)[4] > 0.0,
        "precondition: plain BM25 shows the noise this feature removes"
    );
    assert_eq!(weighted.bm25_scores_weighted(query)[4], 0.0);
}

#[test]
fn tokenize_weighted_public_api_roundtrip() {
    // Downstream callers (handoff-mcp) tokenize the query themselves to
    // inject extra terms; the tokens must plug into
    // bm25_scores_weighted_tokens unchanged.
    let docs = memory_corpus();
    let corpus = Corpus::build_weighted(&docs);
    let mut query_tokens = tokenize_weighted("メモリ注入の基準は？");
    // A topic-marked term keeps its boost through the public roundtrip.
    assert!(query_tokens
        .iter()
        .any(|wt| wt.token == "基準" && wt.weight == TOPIC_BOOST));
    // Caller-injected extra term, as memory_query does with file basenames.
    query_tokens.push(lexsim::WeightedToken {
        token: "atomic".to_string(),
        weight: 1.0,
    });
    let scores = corpus.bm25_scores_weighted_tokens(&query_tokens);
    assert!(scores[0] > 0.0, "injected term must contribute: {scores:?}");
    assert_eq!(
        argmax(&scores),
        1,
        "topic match still dominates: {scores:?}"
    );
}
