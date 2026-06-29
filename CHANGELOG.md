# Changelog

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
