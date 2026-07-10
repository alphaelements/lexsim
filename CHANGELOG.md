# Changelog

## 0.5.0

### ⚠ Breaking

- **`content_hash` values change.** The boundary segmenter was retrained (see
  *Fixed* below), so `tokenize()` emits different tokens for some inputs and
  every `content_hash` derived from them changes. Downstream deduplication
  caches keyed on `content_hash` are invalidated and will re-hash on first
  use. No API signature changed.

### Fixed

- **`is_stopword` missed single-character function words.** `is_stopword("す")`
  and `is_stopword("や")` returned `false`. `JA_SINGLE_CHAR_FUNCTION_CHARS` is
  consulted *only* for two-character tokens, so listing a character there has
  no effect on a lone one-character token — it must be in `JA_STOPWORDS`.
  (The 0.4.1 entry below claims this fragment was handled; it was not. The
  heuristic it fixed never applied to single-character tokens.) Added the
  function words the trained segmenter actually emits standalone: `す`, `や`,
  `した`, `して`, `など`, `とか`, `かも`, `よる`, `ぜひ`. Reported by x-metrics.

- **Segmenter over-split `漢字 + です` and `漢字 + たち`.** `これは記事です`
  segmented as `["これ","は","記事","で","す"]` and `子供たち` as
  `["子供","た","ち"]`. The training corpus contained `です` only after a
  hiragana stem (`幸い です`) and `たち` only once (`人たち`), and the model's
  features are character-class n-grams with no lexicon — so the unseen
  `漢字|で` and `漢字|た` junctions scored as boundaries. Added
  `training/context_supplement.txt` covering the missing left contexts and
  retrained. `これは記事です` → `["これ","は","記事","です"]`,
  `子供たち` → `["子供たち"]`.

  Known limitation: `可愛い` still splits as `可愛` + `い`. Adding examples did
  not fix it and cost boundary precision, so it was left alone; the stem
  `可愛` survives as a content word. `高い`/`美しい` and other adjectives are
  unaffected. Likewise `人々` → `人` + `々` and `原因は設定ミスです` →
  `設定` + `ミス` are pre-existing splits, unchanged by this release.

- **`examples/eval_segmenter.rs` reported misleading accuracy.** It rebuilt
  each gold sentence with `words.concat()`, which deletes the spaces between
  ASCII words and fed the segmenter input `tokenize()` never produces
  (`"Cannotread"` → shredded into character bigrams). It reported word F1
  0.5492 for a model whose real accuracy on the Japanese runs it actually sees
  is boundary F1 0.9041 / word F1 0.8065. Scoring now extracts maximal
  all-Japanese-script spans, matching how `tokenize()` feeds the segmenter.

### Added

- **`segmenter::eval`**: boundary- and word-level Precision/Recall/F1 of the
  embedded model against a gold wakachi corpus, shared by
  `examples/eval_segmenter.rs` and the new regression gate.

- **`tests/segmenter_quality.rs`**: fails if segmentation accuracy regresses
  below the pinned floors, plus behavioural tests pinning the `です` / `たち` /
  サ変 `した` / inflected-verb segmentations so a retrain cannot silently undo
  them.

## 0.4.1

### Fixed

- **`is_stopword` false positives/negatives in the two-character bigram
  heuristic**: the fallback heuristic (for the CJK bigram-glue path used only
  by non-Japanese-script runs, e.g. Hangul) previously fired on *any*
  two-character token containing a single-character particle/auxiliary,
  including real Japanese-script content words that happen to share a
  character with a particle (e.g. `はし` 橋/箸, `すし` 寿司, `たこ` 蛸/凧
  were incorrectly filtered out as stopwords). It also missed the auxiliary
  fragment `す` (trailing character of `です`/`ます`), so fragments like
  glued `す`-bigrams leaked into keyword output. Fixed structurally: the
  heuristic now only fires when the token is *not* entirely Japanese-script
  (hiragana/katakana/kanji) — a bigram of two Japanese-script characters is
  never produced by the fallback path in the first place (Japanese-script
  runs of length ≥2 are always routed through the trained boundary
  segmenter), so it should never have been treated as a stopword by this
  heuristic. `stopwords.rs` now delegates the Japanese-script check to
  `segmenter::inference::is_japanese_run_char` directly instead of
  maintaining a separate Unicode range table, eliminating drift risk between
  the two. Reported by x-metrics.

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
