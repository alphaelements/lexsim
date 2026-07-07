# Changelog

## 0.4.0

### Added

- **Hybrid word segmenter** for Japanese: AdaBoost model (42 features, 2 KB
  embedded binary) trained on 1,276 diverse sentences replaces character-bigram
  tokenization — dramatically improves recall for BM25, Jaccard, and keyword
  extraction on Japanese content
- **Japanese stopword filter** (`is_stopword`): 180+ particles, auxiliaries,
  demonstratives, conjunctions, and adverbs; also filters particle/auxiliary-
  contaminated bigrams
- **TextRank keyword extraction** (`textrank_keywords`): graph-based single-text
  keyword extraction using co-occurrence windows
- **Co-occurrence keyword extraction** (`Corpus::cooccurrence_keywords`):
  find terms that frequently co-occur with a query term across a corpus
- **Normalized TF** (`Corpus::normalized_tf`): length-normalized term frequency
  with stopword exclusion
- `segmenter` module exposed as public API for advanced use (AdaBoost training,
  binary model format, feature extraction, inference)
- Criterion benchmarks for tokenize/jaccard/BM25 performance budgets
- Training infrastructure: `train_segmenter` example, `eval_segmenter` example,
  seed corpus (`training/seed_corpus.txt`)

### Changed

- **Breaking (tokenizer output):** Japanese text now produces word-level tokens
  instead of character bigrams. Downstream BM25 scores and Jaccard coefficients
  will differ from v0.3 — values improve in quality but are not numerically
  compatible
- Verb conjugation forms (e.g. "食べ", "食べる", "食べた") are merged into base
  form tokens for better recall
- Package description updated to reflect hybrid segmentation and TextRank

## 0.3.0

### Added

- `keywords` CLI subcommand (TF-IDF top-N keyword extraction)
- `diff` CLI subcommand (compare two corpora for distinctive keywords)
- `sentiment` CLI subcommand (dictionary-based ja/en sentiment polarity)
- `Corpus::tfidf_keywords` and `corpus_diff` library functions
- `analyze_sentiment` library function
- `tokenize_ngrams` for n-gram tokenization

## 0.2.1

### Changed

- **Breaking (CLI):** `tokenize` output format changed from
  `{"results":[{"tokens":[...],"count":N},...]}` to `{"tokens":[[...],...]}`
  for consistency with `jaccard`/`bm25`/`hash` flat-object style
- Removed `serde` `derive` feature from CLI dependencies (no longer needed)

## 0.2.0

### Added

- CLI binary (`lexsim`) behind the `cli` feature flag. Reads JSON from stdin,
  writes JSON to stdout. Subcommands:
  - `tokenize` — tokenize texts into word tokens, CJK bigrams, and CL-CnG trigrams
  - `jaccard` — compute Jaccard similarity (single pair or batch)
  - `bm25` — compute BM25 relevance scores against a corpus
  - `hash` — compute stable content hashes (FNV-1a 64-bit)
- Install via `cargo install lexsim --features cli`
- Library users are unaffected: serde/serde_json are optional dependencies

## 0.1.0

### Added

- Initial release: dictionary-free, multilingual lexical similarity engine
- `tokenize` / `normalize` — NFKC + UAX#29 + CJK bigrams + CL-CnG
- `jaccard` / `jaccard_sets` / `token_set` — symmetric set similarity
- `Corpus::build` / `Corpus::bm25_scores` — asymmetric BM25 ranking
- `content_hash` / `fnv1a_hex` — stable FNV-1a content hashing
- `Scorer` trait + `LexicalScorer` implementation
