//! Similarity scoring: Jaccard (symmetric, for dedup) and BM25 (asymmetric,
//! for query relevance). Both share the crate's tokenizer so the same notion of
//! "term" drives dedup and retrieval.

use std::collections::{HashMap, HashSet};

use crate::stopwords::is_stopword;
use crate::tokenize::{is_cl_ngram, tokenize};

/// A keyword with its TF-IDF score and raw count across the corpus.
#[derive(Debug, Clone)]
pub struct KeywordEntry {
    pub keyword: String,
    pub score: f64,
    pub count: u32,
}

/// BM25 term-frequency saturation parameter (standard default).
const BM25_K1: f64 = 1.2;
/// BM25 length-normalization parameter (standard default).
const BM25_B: f64 = 0.75;

/// Jaccard similarity of two texts: `|A ∩ B| / |A ∪ B|` over their token
/// **sets**. Symmetric, in `[0, 1]`. Used to answer "are these the same
/// memory?". Two empty texts are defined as identical (1.0); an empty vs.
/// non-empty pair is 0.0.
pub fn jaccard(a: &str, b: &str) -> f64 {
    let sa: HashSet<String> = tokenize(a).into_iter().collect();
    let sb: HashSet<String> = tokenize(b).into_iter().collect();
    jaccard_sets(&sa, &sb)
}

/// Jaccard over pre-tokenized sets (avoids re-tokenizing in clustering loops).
pub fn jaccard_sets(a: &HashSet<String>, b: &HashSet<String>) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let inter = a.intersection(b).count();
    let union = a.len() + b.len() - inter;
    if union == 0 {
        0.0
    } else {
        inter as f64 / union as f64
    }
}

/// Convenience: token set for one text (deduplicated). Useful for callers doing
/// many pairwise Jaccard comparisons.
pub fn token_set(text: &str) -> HashSet<String> {
    tokenize(text).into_iter().collect()
}

/// A BM25 corpus built over a fixed set of documents. Rebuild per query batch —
/// for the handoff use case (tens to low hundreds of memories) this is
/// sub-millisecond and avoids any persistent-index staleness.
pub struct Corpus {
    /// Per-document term-frequency maps.
    doc_tf: Vec<HashMap<String, u32>>,
    /// Per-document length (total token count).
    doc_len: Vec<f64>,
    /// Per-document ordered token stream (including CL-CnG). Needed for
    /// order-sensitive extraction (co-occurrence windows); `doc_tf` alone
    /// loses ordering.
    doc_tokens: Vec<Vec<String>>,
    /// Document frequency: how many docs contain each term.
    df: HashMap<String, u32>,
    /// Mean document length.
    avgdl: f64,
    /// Number of documents.
    n: usize,
}

impl Corpus {
    /// Build a corpus from document texts. Document `i` in the input maps to
    /// index `i` in [`bm25_scores`](Corpus::bm25_scores) output.
    pub fn build(docs: &[String]) -> Self {
        let mut doc_tf = Vec::with_capacity(docs.len());
        let mut doc_len = Vec::with_capacity(docs.len());
        let mut doc_tokens = Vec::with_capacity(docs.len());
        let mut df: HashMap<String, u32> = HashMap::new();
        let mut total_len = 0.0;

        for doc in docs {
            let toks = tokenize(doc);
            let mut tf: HashMap<String, u32> = HashMap::new();
            for t in &toks {
                *tf.entry(t.clone()).or_insert(0) += 1;
            }
            for term in tf.keys() {
                *df.entry(term.clone()).or_insert(0) += 1;
            }
            doc_len.push(toks.len() as f64);
            total_len += toks.len() as f64;
            doc_tf.push(tf);
            doc_tokens.push(toks);
        }

        let n = docs.len();
        let avgdl = if n > 0 { total_len / n as f64 } else { 0.0 };

        Corpus {
            doc_tf,
            doc_len,
            doc_tokens,
            df,
            avgdl,
            n,
        }
    }

    /// Number of documents in the corpus.
    pub fn len(&self) -> usize {
        self.n
    }

    /// True when the corpus has no documents.
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// BM25 score of `query` against every document, returned in document order
    /// (index `i` = score of document `i` passed to [`build`](Corpus::build)).
    /// Scores are non-negative; 0.0 means no query term occurs in the document.
    pub fn bm25_scores(&self, query: &str) -> Vec<f64> {
        let query_terms = tokenize(query);
        self.bm25_scores_tokens(&query_terms)
    }

    /// Extract top-N keywords from the corpus ranked by TF-IDF score.
    ///
    /// Uses smoothed IDF: `ln((N + 1) / df)` so single-document and
    /// homogeneous corpora still produce results instead of all-zero scores.
    /// CL-CnG (internal character n-gram) tokens are excluded from output.
    pub fn tfidf_keywords(&self, top_n: usize) -> Vec<KeywordEntry> {
        if self.n == 0 {
            return Vec::new();
        }

        let mut global_tf: HashMap<&str, u32> = HashMap::new();
        for tf_map in &self.doc_tf {
            for (term, &count) in tf_map {
                if !is_cl_ngram(term) && !is_stopword(term) {
                    *global_tf.entry(term.as_str()).or_insert(0) += count;
                }
            }
        }

        let total_tokens: f64 = global_tf.values().map(|&c| c as f64).sum();
        if total_tokens == 0.0 {
            return Vec::new();
        }

        let mut entries: Vec<KeywordEntry> = global_tf
            .iter()
            .filter_map(|(&term, &count)| {
                let df = *self.df.get(term)? as f64;
                let tf = count as f64 / total_tokens;
                let idf = ((self.n as f64 + 1.0) / df).ln();
                let score = tf * idf;
                if score <= 0.0 {
                    return None;
                }
                Some(KeywordEntry {
                    keyword: term.to_string(),
                    score,
                    count,
                })
            })
            .collect();

        entries.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        entries.truncate(top_n);
        entries
    }

    /// Co-occurrence keywords: words that frequently appear together with a
    /// query term within a context window of `window_size` word tokens.
    ///
    /// Algorithm: tokenize `query` to get seed terms (CL-CnG and stopwords
    /// excluded); for each document, slide a window of `window_size` word
    /// tokens (CL-CnG tokens excluded so the window is word-level, matching
    /// the segmenter's real word tokens) and, whenever a window contains a
    /// query term, count every other non-query, non-stopword term in that
    /// window as one co-occurrence. The final score is
    /// `co_occurrence_count * IDF(term)`, so terms that are both frequent
    /// near the query *and* rare corpus-wide (hence topically distinctive)
    /// rank highest. Returns the top `top_n` terms sorted by score
    /// descending.
    pub fn cooccurrence_keywords(
        &self,
        query: &str,
        window_size: usize,
        top_n: usize,
    ) -> Vec<KeywordEntry> {
        if self.n == 0 || window_size == 0 {
            return Vec::new();
        }

        let query_terms: HashSet<String> = tokenize(query)
            .into_iter()
            .filter(|t| !is_cl_ngram(t) && !is_stopword(t))
            .collect();
        if query_terms.is_empty() {
            return Vec::new();
        }

        // Co-occurrence is inherently order-sensitive, so it needs the
        // original ordered token stream rather than the order-independent
        // per-doc TF maps used by `tfidf_keywords`; that stream is retained
        // in `self.doc_tokens`.
        let mut cooc_count: HashMap<String, u32> = HashMap::new();
        for tokens in &self.doc_tokens {
            let word_tokens: Vec<&String> = tokens.iter().filter(|t| !is_cl_ngram(t)).collect();
            if word_tokens.is_empty() {
                continue;
            }
            for window in word_tokens.windows(window_size.max(1)) {
                let has_query_term = window.iter().any(|t| query_terms.contains(t.as_str()));
                if !has_query_term {
                    continue;
                }
                for term in window {
                    if query_terms.contains(term.as_str()) || is_stopword(term) {
                        continue;
                    }
                    *cooc_count.entry(term.to_string()).or_insert(0) += 1;
                }
            }
        }

        let mut entries: Vec<KeywordEntry> = cooc_count
            .into_iter()
            .filter_map(|(term, count)| {
                let df = *self.df.get(&term)? as f64;
                let idf = ((self.n as f64 + 1.0) / df).ln();
                let score = count as f64 * idf;
                if score <= 0.0 {
                    return None;
                }
                Some(KeywordEntry {
                    keyword: term,
                    score,
                    count,
                })
            })
            .collect();

        entries.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        entries.truncate(top_n);
        entries
    }

    /// BM25 over pre-tokenized query terms (lets callers inject extra terms,
    /// e.g. file basenames, without re-tokenizing).
    pub fn bm25_scores_tokens(&self, query_terms: &[String]) -> Vec<f64> {
        let mut scores = vec![0.0; self.n];
        if self.n == 0 || self.avgdl == 0.0 {
            return scores;
        }

        // Deduplicate query terms — repeated query terms shouldn't multiply a
        // document's score beyond BM25's per-term contribution.
        let unique_terms: HashSet<&String> = query_terms.iter().collect();

        for term in unique_terms {
            let df = match self.df.get(term) {
                Some(&df) if df > 0 => df as f64,
                _ => continue,
            };
            // BM25 IDF with the +1 inside the log to keep it non-negative.
            let idf = (((self.n as f64 - df + 0.5) / (df + 0.5)) + 1.0).ln();

            for (i, tf_map) in self.doc_tf.iter().enumerate() {
                let tf = match tf_map.get(term) {
                    Some(&tf) => tf as f64,
                    None => continue,
                };
                let dl = self.doc_len[i];
                let denom = tf + BM25_K1 * (1.0 - BM25_B + BM25_B * dl / self.avgdl);
                scores[i] += idf * (tf * (BM25_K1 + 1.0)) / denom;
            }
        }
        scores
    }
}

/// PageRank damping factor (standard TextRank default).
const TEXTRANK_DAMPING: f64 = 0.85;
/// Maximum PageRank iterations before giving up on convergence.
const TEXTRANK_MAX_ITERATIONS: usize = 100;
/// Convergence threshold: stop iterating once the largest per-node score
/// change drops below this.
const TEXTRANK_CONVERGENCE_THRESHOLD: f64 = 1e-6;

/// Extract keywords using a simplified TextRank graph algorithm.
///
/// Builds an undirected co-occurrence graph from word tokens (CL-CnG tokens
/// and stopwords excluded) within a sliding window of `window_size` tokens —
/// two terms get an edge (weight = co-occurrence count) whenever they share
/// a window — then runs PageRank-style iteration over that graph
/// (damping=0.85, up to 100 iterations, convergence threshold 1e-6) to find
/// the most central terms. Returns the top `top_n` terms by final score,
/// descending.
pub fn textrank_keywords(text: &str, window_size: usize, top_n: usize) -> Vec<KeywordEntry> {
    let word_tokens: Vec<String> = tokenize(text)
        .into_iter()
        .filter(|t| !is_cl_ngram(t) && !is_stopword(t))
        .collect();

    if word_tokens.is_empty() {
        return Vec::new();
    }

    // Term occurrence counts double as the KeywordEntry::count field.
    let mut term_count: HashMap<&str, u32> = HashMap::new();
    for t in &word_tokens {
        *term_count.entry(t.as_str()).or_insert(0) += 1;
    }

    // Undirected weighted adjacency: edge weight = number of windows in
    // which the two terms co-occur.
    let mut adjacency: HashMap<&str, HashMap<&str, f64>> = HashMap::new();
    let window = window_size.max(2);
    if word_tokens.len() >= 2 {
        for win in word_tokens.windows(window.min(word_tokens.len())) {
            for i in 0..win.len() {
                for j in (i + 1)..win.len() {
                    let (a, b) = (win[i].as_str(), win[j].as_str());
                    if a == b {
                        continue;
                    }
                    *adjacency.entry(a).or_default().entry(b).or_insert(0.0) += 1.0;
                    *adjacency.entry(b).or_default().entry(a).or_insert(0.0) += 1.0;
                }
            }
        }
    }

    let terms: Vec<&str> = term_count.keys().copied().collect();
    let n = terms.len();
    if n == 0 {
        return Vec::new();
    }

    let index: HashMap<&str, usize> = terms.iter().enumerate().map(|(i, &t)| (t, i)).collect();
    let out_weight: Vec<f64> = terms
        .iter()
        .map(|t| adjacency.get(t).map(|nb| nb.values().sum()).unwrap_or(0.0))
        .collect();

    let mut scores = vec![1.0 / n as f64; n];
    for _ in 0..TEXTRANK_MAX_ITERATIONS {
        let mut next = vec![(1.0 - TEXTRANK_DAMPING) / n as f64; n];
        for (i, &term) in terms.iter().enumerate() {
            let Some(neighbors) = adjacency.get(term) else {
                continue;
            };
            for (&nbr, &w) in neighbors {
                let j = index[nbr];
                if out_weight[j] > 0.0 {
                    next[i] += TEXTRANK_DAMPING * scores[j] * w / out_weight[j];
                }
            }
        }
        let max_delta = next
            .iter()
            .zip(scores.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        scores = next;
        if max_delta < TEXTRANK_CONVERGENCE_THRESHOLD {
            break;
        }
    }

    let mut entries: Vec<KeywordEntry> = terms
        .iter()
        .enumerate()
        .map(|(i, &term)| KeywordEntry {
            keyword: term.to_string(),
            score: scores[i],
            count: term_count[term],
        })
        .collect();

    entries.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    entries.truncate(top_n);
    entries
}

/// A keyword distinctive to one side of a corpus comparison.
#[derive(Debug, Clone)]
pub struct DistinctiveKeyword {
    pub keyword: String,
    pub ratio: f64,
}

/// Compare two corpora and return keywords distinctive to each side.
///
/// "Distinctive" means the keyword's normalized frequency in one corpus is
/// significantly higher than in the other. `ratio` = freq_this / freq_other
/// (with a small smoothing term to avoid division by zero).
///
/// Returns `(a_distinctive, b_distinctive)` sorted by descending ratio.
pub fn corpus_diff(
    corpus_a: &[String],
    corpus_b: &[String],
    top_n: usize,
) -> (Vec<DistinctiveKeyword>, Vec<DistinctiveKeyword>) {
    let tf_a = normalized_tf(corpus_a);
    let tf_b = normalized_tf(corpus_b);

    let all_terms: HashSet<&String> = tf_a.keys().chain(tf_b.keys()).collect();

    // Smoothing: assume each unseen term appeared once in a corpus of
    // comparable size. This keeps ratios interpretable (typically 1–20)
    // rather than astronomical when a term is exclusive to one side.
    let size_a = tf_a.values().count().max(1) as f64;
    let size_b = tf_b.values().count().max(1) as f64;
    let smooth_a = 1.0 / (size_a + 1.0);
    let smooth_b = 1.0 / (size_b + 1.0);

    let mut a_distinctive: Vec<DistinctiveKeyword> = Vec::new();
    let mut b_distinctive: Vec<DistinctiveKeyword> = Vec::new();

    for term in &all_terms {
        let fa = tf_a.get(*term).copied().unwrap_or(0.0);
        let fb = tf_b.get(*term).copied().unwrap_or(0.0);

        let ratio_a = (fa + smooth_a) / (fb + smooth_b);
        let ratio_b = (fb + smooth_b) / (fa + smooth_a);

        if ratio_a > 1.5 {
            a_distinctive.push(DistinctiveKeyword {
                keyword: term.to_string(),
                ratio: ratio_a,
            });
        }
        if ratio_b > 1.5 {
            b_distinctive.push(DistinctiveKeyword {
                keyword: term.to_string(),
                ratio: ratio_b,
            });
        }
    }

    a_distinctive.sort_by(|a, b| {
        b.ratio
            .partial_cmp(&a.ratio)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    b_distinctive.sort_by(|a, b| {
        b.ratio
            .partial_cmp(&a.ratio)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    a_distinctive.truncate(top_n);
    b_distinctive.truncate(top_n);

    (a_distinctive, b_distinctive)
}

fn normalized_tf(docs: &[String]) -> HashMap<String, f64> {
    let mut counts: HashMap<String, u32> = HashMap::new();
    let mut total = 0u32;

    for doc in docs {
        for token in tokenize(doc) {
            if is_cl_ngram(&token) || is_stopword(&token) {
                continue;
            }
            *counts.entry(token).or_insert(0) += 1;
            total += 1;
        }
    }

    if total == 0 {
        return HashMap::new();
    }

    counts
        .into_iter()
        .map(|(term, count)| (term, count as f64 / total as f64))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jaccard_identical_is_one() {
        assert!((jaccard("hello world", "hello world") - 1.0).abs() < 1e-9);
    }

    #[test]
    fn jaccard_disjoint_is_low() {
        // Completely different content scores low (CL n-grams may add a tiny
        // overlap, but it must be well under 0.5).
        assert!(jaccard("hello world", "zzz qqq") < 0.2);
    }

    #[test]
    fn jaccard_partial_overlap() {
        let s = jaccard("atomic write always", "atomic write never");
        assert!(s > 0.3 && s < 1.0, "got {s}");
    }

    #[test]
    fn jaccard_empty_both_identical() {
        assert_eq!(jaccard("", ""), 1.0);
    }

    #[test]
    fn jaccard_empty_one_side_zero() {
        assert_eq!(jaccard("hello", ""), 0.0);
    }

    #[test]
    fn jaccard_japanese_near_duplicate() {
        let s = jaccard(
            "メモリ機能はセッション間で教訓を引き継ぐ",
            "メモリ機能はセッション間で教訓を保持する",
        );
        assert!(s > 0.5, "near-duplicate JP should score high, got {s}");
    }

    #[test]
    fn bm25_ranks_relevant_doc_first() {
        let docs = vec![
            "always use atomic_write for handoff files".to_string(),
            "the cat sat on the mat in the sun".to_string(),
            "configure the milestone schedule and assignee".to_string(),
        ];
        let corpus = Corpus::build(&docs);
        let scores = corpus.bm25_scores("atomic write");
        // Doc 0 must outrank the others.
        assert!(scores[0] > scores[1]);
        assert!(scores[0] > scores[2]);
    }

    #[test]
    fn bm25_japanese_query() {
        let docs = vec![
            "メモリ機能はセッション間で教訓を引き継ぐ".to_string(),
            "ガントチャートでスケジュールを表示する".to_string(),
        ];
        let corpus = Corpus::build(&docs);
        let scores = corpus.bm25_scores("メモリ");
        assert!(scores[0] > scores[1], "JP query should match doc 0");
    }

    #[test]
    fn bm25_no_match_is_zero() {
        let docs = vec!["hello world".to_string()];
        let corpus = Corpus::build(&docs);
        let scores = corpus.bm25_scores("completely unrelated query xyzzy");
        assert_eq!(scores[0], 0.0);
    }

    #[test]
    fn bm25_empty_corpus() {
        let corpus = Corpus::build(&[]);
        assert!(corpus.is_empty());
        assert!(corpus.bm25_scores("anything").is_empty());
    }

    #[test]
    fn bm25_cross_language_identifier_match() {
        // An English identifier embedded in a Japanese memory is found by an
        // English query via shared word tokens.
        let docs = vec![
            "atomic_write を必ず使う（torn read 防止）".to_string(),
            "ガントチャートの表示設定".to_string(),
        ];
        let corpus = Corpus::build(&docs);
        let scores = corpus.bm25_scores("atomic_write");
        assert!(scores[0] > scores[1]);
        assert!(scores[0] > 0.0);
    }

    #[test]
    fn tfidf_keywords_ranks_distinctive_terms() {
        let docs = vec![
            "rust rust rust programming".to_string(),
            "python programming language".to_string(),
            "rust language systems".to_string(),
        ];
        let corpus = Corpus::build(&docs);
        let kw = corpus.tfidf_keywords(5);
        assert!(!kw.is_empty());
        assert!(kw[0].score > 0.0);
        assert!(kw[0].count > 0);
    }

    #[test]
    fn tfidf_keywords_single_doc_not_empty() {
        let docs = vec!["Rust is a systems programming language".to_string()];
        let corpus = Corpus::build(&docs);
        let kw = corpus.tfidf_keywords(5);
        assert!(!kw.is_empty(), "single-doc corpus must produce keywords");
        assert!(kw[0].score > 0.0);
    }

    #[test]
    fn tfidf_keywords_no_cl_ngram_leakage() {
        let docs = vec!["hello world rust".to_string()];
        let corpus = Corpus::build(&docs);
        let kw = corpus.tfidf_keywords(20);
        for entry in &kw {
            assert!(
                !entry.keyword.starts_with('\u{1}'),
                "CL-CnG leaked: {:?}",
                entry.keyword
            );
        }
    }

    #[test]
    fn tfidf_keywords_empty_corpus() {
        let corpus = Corpus::build(&[]);
        assert!(corpus.tfidf_keywords(10).is_empty());
    }

    #[test]
    fn tfidf_keywords_respects_top_n() {
        let docs = vec!["a b c d e f g h i j".to_string()];
        let corpus = Corpus::build(&docs);
        let kw = corpus.tfidf_keywords(3);
        assert!(kw.len() <= 3);
    }

    #[test]
    fn tfidf_keywords_japanese() {
        let docs = vec![
            "メモリ機能はセッション間で教訓を引き継ぐ".to_string(),
            "メモリ機能で過去の学びを保持する".to_string(),
        ];
        let corpus = Corpus::build(&docs);
        let kw = corpus.tfidf_keywords(10);
        assert!(!kw.is_empty());
    }

    #[test]
    fn tfidf_keywords_excludes_stopwords() {
        let docs = vec![
            "この機能は便利です".to_string(),
            "この機能は快適です".to_string(),
        ];
        let corpus = Corpus::build(&docs);
        let kw = corpus.tfidf_keywords(20);
        // "です" is emitted as a standalone bi-gram token by the tokenizer and
        // must be filtered out at the extraction stage as a stopword.
        assert!(
            !kw.iter().any(|k| k.keyword == "です"),
            "keywords should not include the stopword です: {:?}",
            kw.iter().map(|k| &k.keyword).collect::<Vec<_>>()
        );
        // Content word must still survive.
        assert!(kw.iter().any(|k| k.keyword == "機能"));
    }

    #[test]
    fn normalized_tf_excludes_stopwords() {
        let docs = vec!["この機能は便利です".to_string()];
        let tf = normalized_tf(&docs);
        assert!(!tf.contains_key("です"));
        assert!(tf.contains_key("機能"));
    }

    #[test]
    fn tfidf_keywords_excludes_particle_contaminated_bigrams() {
        // Regression test for the rework finding: unbroken Japanese sentences
        // tokenize into an unbroken run of bigrams with no trailing unigram, so
        // particles like は/の/で only ever appear glued to an adjacent content
        // character (は便, の機, 能は, ...). An exact-match stopword list alone
        // cannot catch these; the extraction stage must also drop bigrams where
        // one side is a single-character particle.
        let docs = vec![
            "この機能は便利です".to_string(),
            "この機能は快適です".to_string(),
        ];
        let corpus = Corpus::build(&docs);
        let kw = corpus.tfidf_keywords(10);
        let keywords: Vec<&str> = kw.iter().map(|k| k.keyword.as_str()).collect();
        for particle in ["の", "は", "を", "で"] {
            assert!(
                !keywords.iter().any(|k| k.contains(particle)),
                "keyword {:?} unexpectedly contains particle {:?} in {:?}",
                keywords.iter().find(|k| k.contains(particle)),
                particle,
                keywords
            );
        }
        // Content words must still survive the stricter filter.
        assert!(keywords.contains(&"機能"));
    }

    #[test]
    fn normalized_tf_excludes_particle_contaminated_bigrams() {
        let docs = vec!["メモリ機能はセッション間で教訓を引き継ぐ".to_string()];
        let tf = normalized_tf(&docs);
        for term in tf.keys() {
            for particle in ["の", "は", "を", "で"] {
                assert!(
                    !term.contains(particle),
                    "term {:?} unexpectedly contains particle {:?}",
                    term,
                    particle
                );
            }
        }
        assert!(tf.contains_key("メモ") || tf.contains_key("機能"));
    }

    #[test]
    fn tfidf_keywords_excludes_aux_contaminated_bigrams() {
        // Regression test for the rework finding: single-character auxiliary
        // fragments (た/だ) were omitted from the bigram-glue filter, so
        // "した" (し + auxiliary た, from 降りました) leaked into keyword
        // output alongside genuine content words 昨日/今日.
        //
        // Note: "まし"/"りま" (interior fragments of the multi-character
        // auxiliary ました that don't touch a single-char stopword on either
        // side) are a structurally different, known limitation of the
        // single-char bigram-glue heuristic and are intentionally out of
        // scope here; see src/stopwords.rs module docs.
        let docs = vec![
            "昨日は雨が降りました".to_string(),
            "今日も雨が降りました".to_string(),
        ];
        let corpus = Corpus::build(&docs);
        let kw = corpus.tfidf_keywords(10);
        let keywords: Vec<&str> = kw.iter().map(|k| k.keyword.as_str()).collect();
        assert!(
            !keywords.contains(&"した"),
            "keyword list unexpectedly contains auxiliary-glued bigram した: {:?}",
            keywords
        );
        // Content words must still survive the stricter filter.
        assert!(keywords.contains(&"昨日"));
        assert!(keywords.contains(&"今日"));
    }

    #[test]
    fn corpus_diff_finds_distinctive_keywords() {
        let a = vec![
            "rust systems programming".to_string(),
            "rust memory safety".to_string(),
        ];
        let b = vec![
            "python data science".to_string(),
            "python machine learning".to_string(),
        ];
        let (a_dist, b_dist) = corpus_diff(&a, &b, 10);
        assert!(!a_dist.is_empty());
        assert!(!b_dist.is_empty());
        assert!(a_dist[0].ratio > 1.5);
        assert!(b_dist[0].ratio > 1.5);
    }

    #[test]
    fn corpus_diff_identical_corpora_no_distinctive() {
        let a = vec!["hello world".to_string()];
        let b = vec!["hello world".to_string()];
        let (a_dist, b_dist) = corpus_diff(&a, &b, 10);
        assert!(a_dist.is_empty());
        assert!(b_dist.is_empty());
    }

    #[test]
    fn corpus_diff_empty_corpora() {
        let (a_dist, b_dist) = corpus_diff(&[], &[], 10);
        assert!(a_dist.is_empty());
        assert!(b_dist.is_empty());
    }

    #[test]
    fn cooccurrence_keywords_finds_context_terms() {
        let docs = vec![
            "メモリ機能はセッション間で教訓を保持する仕組みです".to_string(),
            "ガントチャートは日程を表示する機能です".to_string(),
        ];
        let corpus = Corpus::build(&docs);
        let kw = corpus.cooccurrence_keywords("メモリ", 4, 10);
        assert!(!kw.is_empty(), "expected co-occurring terms near メモリ");
        // Terms that share a window with the query term in doc 0 should
        // surface (e.g. 機能/セッション/教訓), not unrelated doc-1-only terms.
        assert!(kw
            .iter()
            .any(|k| k.keyword == "機能" || k.keyword == "セッション"));
    }

    #[test]
    fn cooccurrence_keywords_excludes_query_term_and_stopwords() {
        let docs = vec!["メモリ機能はセッション間で教訓を保持する仕組みです".to_string()];
        let corpus = Corpus::build(&docs);
        let kw = corpus.cooccurrence_keywords("メモリ", 4, 20);
        assert!(
            !kw.iter().any(|k| k.keyword == "メモリ"),
            "query term must not co-occur with itself"
        );
        assert!(
            !kw.iter().any(|k| is_stopword(&k.keyword)),
            "stopwords must be excluded: {:?}",
            kw.iter().map(|k| &k.keyword).collect::<Vec<_>>()
        );
    }

    #[test]
    fn cooccurrence_keywords_respects_top_n() {
        let docs = vec!["メモリ機能はセッション間で教訓を保持する仕組みです".to_string()];
        let corpus = Corpus::build(&docs);
        let kw = corpus.cooccurrence_keywords("メモリ", 4, 2);
        assert!(kw.len() <= 2);
    }

    #[test]
    fn cooccurrence_keywords_empty_corpus() {
        let corpus = Corpus::build(&[]);
        assert!(corpus.cooccurrence_keywords("query", 4, 10).is_empty());
    }

    #[test]
    fn cooccurrence_keywords_no_match_is_empty() {
        let docs = vec!["hello world rust programming".to_string()];
        let corpus = Corpus::build(&docs);
        let kw = corpus.cooccurrence_keywords("zzz_nonexistent", 4, 10);
        assert!(kw.is_empty());
    }

    #[test]
    fn textrank_keywords_extracts_central_terms() {
        let text = "メモリ機能はセッション間で教訓を保持する。教訓はメモリ機能に蓄積される。";
        let kw = textrank_keywords(text, 4, 10);
        assert!(!kw.is_empty());
        // 機能/メモリ/教訓 co-occur repeatedly and should rank as central terms.
        assert!(kw
            .iter()
            .any(|k| k.keyword == "機能" || k.keyword == "メモリ" || k.keyword == "教訓"));
    }

    #[test]
    fn textrank_keywords_excludes_cl_ngram_and_stopwords() {
        let text = "この機能はとても便利です。この機能は快適です。".to_string();
        let kw = textrank_keywords(&text, 4, 20);
        for entry in &kw {
            assert!(
                !entry.keyword.starts_with('\u{1}'),
                "CL-CnG leaked: {:?}",
                entry.keyword
            );
            assert!(
                !is_stopword(&entry.keyword),
                "stopword leaked: {:?}",
                entry.keyword
            );
        }
    }

    #[test]
    fn textrank_keywords_respects_top_n() {
        let text = "rust programming language systems memory safety concurrency performance";
        let kw = textrank_keywords(text, 3, 3);
        assert!(kw.len() <= 3);
    }

    #[test]
    fn textrank_keywords_empty_text() {
        assert!(textrank_keywords("", 4, 10).is_empty());
    }

    #[test]
    fn textrank_keywords_single_word_no_panic() {
        // A single content word has no co-occurrence window partner; must not
        // panic and should still surface the lone term.
        let kw = textrank_keywords("機能", 4, 10);
        assert!(kw.iter().any(|k| k.keyword == "機能") || kw.is_empty());
    }
}
