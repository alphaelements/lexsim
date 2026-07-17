//! Similarity scoring: Jaccard (symmetric, for dedup) and BM25 (asymmetric,
//! for query relevance). Both share the crate's tokenizer so the same notion of
//! "term" drives dedup and retrieval.

use std::collections::{HashMap, HashSet};

use crate::stopwords::is_stopword;
use crate::tokenize::{is_cl_ngram, tokenize, tokenize_weighted, WeightedToken};

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
    /// Per-document term-frequency maps. Values are `f64` to support
    /// weighted document tokens ([`build_weighted_tokens`](Corpus::build_weighted_tokens))
    /// where a term's effective TF is the sum of its per-occurrence weights.
    doc_tf: Vec<HashMap<String, f64>>,
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
        Self::build_filtered(docs, false)
    }

    /// Build a corpus with stopwords and stopword-only CL-CnG trigrams
    /// excluded from the term statistics (TF, DF, document length).
    /// Content-word-derived trigrams (those whose character window overlaps
    /// at least one content word) are **retained** so that fuzzy
    /// substring matching still works for paraphrase pairs that share no
    /// exact word tokens.
    ///
    /// This is the counterpart of
    /// [`bm25_scores_weighted`](Corpus::bm25_scores_weighted): particles
    /// and pure-stopword character-trigram noise stop accumulating score
    /// across unrelated documents, while content-derived trigrams keep the
    /// fuzzy-match recall that plain BM25 had.
    ///
    /// [`build`](Corpus::build) is unchanged and keeps the raw token stream
    /// (the tokenizer's BM25 term-frequency contract).
    pub fn build_weighted(docs: &[String]) -> Self {
        let weighted_docs: Vec<Vec<WeightedToken>> =
            docs.iter().map(|d| tokenize_weighted(d)).collect();
        Self::build_weighted_tokens(&weighted_docs)
    }

    /// Build a corpus from pre-weighted document tokens. Each document is a
    /// `Vec<WeightedToken>` — tokens with `weight <= 0.0` are excluded from
    /// the corpus statistics, and each term's effective TF is the sum of its
    /// per-occurrence weights (so a keyword that appears once with `weight =
    /// 2.0` counts as if it appeared twice).
    ///
    /// This lets callers boost document-side terms explicitly — for example,
    /// boosting topic keywords extracted from metadata — instead of resorting
    /// to TF hacks (repeating keywords N times to inflate their count).
    ///
    /// ```
    /// use lexsim::{Corpus, WeightedToken};
    ///
    /// let doc_tokens = vec![vec![
    ///     WeightedToken { token: "memory".to_string(), weight: 2.0 },
    ///     WeightedToken { token: "function".to_string(), weight: 1.0 },
    /// ]];
    /// let corpus = Corpus::build_weighted_tokens(&doc_tokens);
    /// let scores = corpus.bm25_scores("memory");
    /// assert!(scores[0] > 0.0);
    /// ```
    pub fn build_weighted_tokens(docs: &[Vec<WeightedToken>]) -> Self {
        let mut doc_tf = Vec::with_capacity(docs.len());
        let mut doc_len = Vec::with_capacity(docs.len());
        let mut doc_tokens = Vec::with_capacity(docs.len());
        let mut df: HashMap<String, u32> = HashMap::new();
        let mut total_len = 0.0;

        for doc in docs {
            let mut tf: HashMap<String, f64> = HashMap::new();
            let mut toks = Vec::new();
            for wt in doc {
                if !wt.weight.is_finite() || wt.weight <= 0.0 {
                    continue;
                }
                *tf.entry(wt.token.clone()).or_insert(0.0) += wt.weight;
                toks.push(wt.token.clone());
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

    fn build_filtered(docs: &[String], exclude_noise: bool) -> Self {
        let mut doc_tf = Vec::with_capacity(docs.len());
        let mut doc_len = Vec::with_capacity(docs.len());
        let mut doc_tokens = Vec::with_capacity(docs.len());
        let mut df: HashMap<String, u32> = HashMap::new();
        let mut total_len = 0.0;

        for doc in docs {
            let mut toks = tokenize(doc);
            if exclude_noise {
                toks.retain(|t| !is_cl_ngram(t) && !is_stopword(t));
            }
            let mut tf: HashMap<String, f64> = HashMap::new();
            for t in &toks {
                *tf.entry(t.clone()).or_insert(0.0) += 1.0;
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

        let mut global_tf: HashMap<&str, f64> = HashMap::new();
        for tf_map in &self.doc_tf {
            for (term, &count) in tf_map {
                if !is_cl_ngram(term) && !is_stopword(term) {
                    *global_tf.entry(term.as_str()).or_insert(0.0) += count;
                }
            }
        }

        let total_tokens: f64 = global_tf.values().sum();
        if total_tokens == 0.0 {
            return Vec::new();
        }

        let mut entries: Vec<KeywordEntry> = global_tf
            .iter()
            .filter_map(|(&term, &count)| {
                let df = *self.df.get(term)? as f64;
                let tf = count / total_tokens;
                let idf = ((self.n as f64 + 1.0) / df).ln();
                let score = tf * idf;
                if score <= 0.0 {
                    return None;
                }
                Some(KeywordEntry {
                    keyword: term.to_string(),
                    score,
                    count: count.round() as u32,
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

    /// Particle-context weighted BM25 score of `query` against every
    /// document, in document order. The query is tokenized with
    /// [`tokenize_weighted`]: topic/object/case-marked terms contribute more
    /// (`score × weight`), stopwords and CL-CnG trigrams contribute nothing.
    ///
    /// Build the corpus with [`build_weighted`](Corpus::build_weighted) so
    /// the document statistics use the same content-word-only filtering.
    ///
    /// ```
    /// use lexsim::Corpus;
    ///
    /// let memories = vec![
    ///     "atomic_write を必ず使う（torn read 防止）".to_string(),
    ///     "ガントチャートの表示設定".to_string(),
    /// ];
    /// let corpus = Corpus::build_weighted(&memories);
    /// let scores = corpus.bm25_scores_weighted("atomic_write のフックについて");
    /// assert!(scores[0] > scores[1]);
    /// ```
    pub fn bm25_scores_weighted(&self, query: &str) -> Vec<f64> {
        self.bm25_scores_weighted_tokens(&tokenize_weighted(query))
    }

    /// Weighted BM25 over pre-weighted query tokens (the counterpart of
    /// [`bm25_scores_tokens`](Corpus::bm25_scores_tokens) — lets callers
    /// inject extra terms with hand-assigned weights).
    ///
    /// Tokens with a non-finite weight or `weight <= 0.0` are skipped. When
    /// the same term occurs with several weights, the highest wins (a term
    /// that is topic-marked anywhere in the query keeps that role;
    /// occurrences are not summed — same de-duplication contract as
    /// `bm25_scores_tokens`).
    pub fn bm25_scores_weighted_tokens(&self, query_tokens: &[WeightedToken]) -> Vec<f64> {
        let mut scores = vec![0.0; self.n];
        if self.n == 0 || self.avgdl == 0.0 {
            return scores;
        }

        let mut term_weight: HashMap<&str, f64> = HashMap::new();
        for wt in query_tokens {
            if !wt.weight.is_finite() || wt.weight <= 0.0 {
                continue;
            }
            let entry = term_weight.entry(wt.token.as_str()).or_insert(0.0);
            if wt.weight > *entry {
                *entry = wt.weight;
            }
        }

        for (term, weight) in term_weight {
            let df = match self.df.get(term) {
                Some(&df) if df > 0 => df as f64,
                _ => continue,
            };
            let idf = (((self.n as f64 - df + 0.5) / (df + 0.5)) + 1.0).ln();

            for (i, tf_map) in self.doc_tf.iter().enumerate() {
                let tf = match tf_map.get(term) {
                    Some(&tf) => tf,
                    None => continue,
                };
                let dl = self.doc_len[i];
                let denom = tf + BM25_K1 * (1.0 - BM25_B + BM25_B * dl / self.avgdl);
                scores[i] += weight * idf * (tf * (BM25_K1 + 1.0)) / denom;
            }
        }
        scores
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
                    Some(&tf) => tf,
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
    fn weighted_corpus_excludes_stopwords_and_stopword_only_ngrams() {
        let docs = vec!["この機能は便利です".to_string()];
        let plain = Corpus::build(&docs);
        let weighted = Corpus::build_weighted(&docs);
        // Stopwords score in the plain corpus but are absent from the
        // weighted one.
        assert!(plain.bm25_scores_tokens(&["は".to_string()])[0] > 0.0);
        assert_eq!(weighted.bm25_scores_tokens(&["は".to_string()])[0], 0.0);
        // Content words still score.
        assert!(weighted.bm25_scores_tokens(&["機能".to_string()])[0] > 0.0);
    }

    #[test]
    fn weighted_corpus_retains_content_derived_ngrams() {
        // Content-word-derived trigrams must be present in the weighted
        // corpus so fuzzy substring matching still works.
        let docs = vec!["メモリ機能".to_string()];
        let weighted = Corpus::build_weighted(&docs);
        // A trigram covering content characters (e.g. "メモリ") must score.
        let trigram = format!("\u{1}メモリ");
        assert!(
            weighted.bm25_scores_tokens(&[trigram.clone()])[0] > 0.0,
            "content-derived trigram {trigram:?} must be in the weighted corpus"
        );
    }

    #[test]
    fn weighted_scores_boost_topic_terms() {
        // Same tf/df/doc-length for alpha and delta; the only difference is
        // the query role: alpha is topic-marked (は), delta is bare. The
        // topic-marked doc must win by exactly the boost factor.
        let docs = vec!["alpha content".to_string(), "delta content".to_string()];
        let corpus = Corpus::build_weighted(&docs);
        let scores = corpus.bm25_scores_weighted("alphaは delta");
        assert!(scores[0] > scores[1], "topic-boosted term must outrank");
        let ratio = scores[0] / scores[1];
        assert!(
            (ratio - crate::tokenize::TOPIC_BOOST).abs() < 1e-9,
            "expected boost ratio {}, got {ratio}",
            crate::tokenize::TOPIC_BOOST
        );
    }

    #[test]
    fn weighted_scores_stopword_only_query_is_zero() {
        let docs = vec!["この機能は便利です".to_string()];
        let corpus = Corpus::build_weighted(&docs);
        let scores = corpus.bm25_scores_weighted("これは");
        assert_eq!(scores[0], 0.0);
    }

    #[test]
    fn weighted_scores_tokens_max_weight_wins_on_duplicates() {
        // The same term occurring both boosted and unboosted in one query:
        // the boosted role must win (not double-count, not average).
        let docs = vec!["alpha content".to_string(), "delta content".to_string()];
        let corpus = Corpus::build_weighted(&docs);
        let query = vec![
            WeightedToken {
                token: "alpha".to_string(),
                weight: 1.0,
            },
            WeightedToken {
                token: "alpha".to_string(),
                weight: 2.0,
            },
            WeightedToken {
                token: "delta".to_string(),
                weight: 1.0,
            },
        ];
        let scores = corpus.bm25_scores_weighted_tokens(&query);
        let ratio = scores[0] / scores[1];
        assert!((ratio - 2.0).abs() < 1e-9, "expected 2.0, got {ratio}");
    }

    #[test]
    fn weighted_scores_tokens_non_finite_weights_are_skipped() {
        // Caller-supplied weights are untrusted: NaN and infinity must not
        // poison the scores (adversarial-review NIT).
        let docs = vec!["alpha content".to_string()];
        let corpus = Corpus::build_weighted(&docs);
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1.0] {
            let query = vec![WeightedToken {
                token: "alpha".to_string(),
                weight: bad,
            }];
            let scores = corpus.bm25_scores_weighted_tokens(&query);
            assert_eq!(scores[0], 0.0, "weight {bad} must be skipped");
        }
    }

    #[test]
    fn weighted_scores_empty_corpus() {
        let corpus = Corpus::build_weighted(&[]);
        assert!(corpus.is_empty());
        assert!(corpus.bm25_scores_weighted("anything").is_empty());
    }

    #[test]
    fn weighted_ranks_identifier_memory_first() {
        // The spec's motivating scenario: an identifier-bearing memory must
        // outrank generic memories for an identifier query, without CL-CnG
        // or particle noise diluting the ranking.
        let docs = vec![
            "atomic_write を必ず使う（torn read 防止）".to_string(),
            "ガントチャートの表示設定を変更する".to_string(),
            "メモリ注入の基準はスコア上位5件です".to_string(),
        ];
        let corpus = Corpus::build_weighted(&docs);
        let scores = corpus.bm25_scores_weighted("atomic_write のフックについて教えて");
        assert!(scores[0] > scores[1], "{scores:?}");
        assert!(scores[0] > scores[2], "{scores:?}");
    }

    #[test]
    fn textrank_keywords_single_word_no_panic() {
        // A single content word has no co-occurrence window partner; must not
        // panic and should still surface the lone term.
        let kw = textrank_keywords("機能", 4, 10);
        assert!(kw.iter().any(|k| k.keyword == "機能") || kw.is_empty());
    }

    #[test]
    fn build_weighted_tokens_boosts_doc_side_tf() {
        // A keyword with weight 2.0 should have higher effective TF than
        // the same keyword with weight 1.0 in another doc.
        let doc_a = vec![WeightedToken {
            token: "memory".to_string(),
            weight: 2.0,
        }];
        let doc_b = vec![WeightedToken {
            token: "memory".to_string(),
            weight: 1.0,
        }];
        let corpus = Corpus::build_weighted_tokens(&[doc_a, doc_b]);
        let scores = corpus.bm25_scores("memory");
        assert!(
            scores[0] > scores[1],
            "boosted doc must outscore unboosted: {scores:?}"
        );
    }

    #[test]
    fn build_weighted_tokens_excludes_zero_weight() {
        let doc = vec![
            WeightedToken {
                token: "content".to_string(),
                weight: 1.0,
            },
            WeightedToken {
                token: "noise".to_string(),
                weight: 0.0,
            },
        ];
        let corpus = Corpus::build_weighted_tokens(&[doc]);
        assert!(corpus.bm25_scores("content")[0] > 0.0);
        assert_eq!(corpus.bm25_scores("noise")[0], 0.0);
    }

    #[test]
    fn build_weighted_tokens_empty_docs() {
        let corpus = Corpus::build_weighted_tokens(&[]);
        assert!(corpus.is_empty());
    }

    #[test]
    fn build_weighted_tokens_matches_build_weighted_for_plain_text() {
        // build_weighted(docs) is now implemented as
        // build_weighted_tokens(tokenize_weighted(doc) for each doc).
        // The scores must agree.
        let docs = vec![
            "atomic_write を必ず使う".to_string(),
            "ガントチャートの表示設定".to_string(),
        ];
        let from_text = Corpus::build_weighted(&docs);
        let from_tokens = Corpus::build_weighted_tokens(
            &docs
                .iter()
                .map(|d| tokenize_weighted(d))
                .collect::<Vec<_>>(),
        );
        let q = "atomic_write";
        let s1 = from_text.bm25_scores(q);
        let s2 = from_tokens.bm25_scores(q);
        for (a, b) in s1.iter().zip(s2.iter()) {
            assert!((a - b).abs() < 1e-9, "scores must match: {s1:?} vs {s2:?}");
        }
    }
}
