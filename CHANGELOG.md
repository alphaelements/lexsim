# Changelog

## 0.6.0

### Added

- **Particle-context weighted BM25** — uses Japanese case particles as a
  dictionary-free stand-in for part-of-speech tagging, so topic terms score
  higher and function-word/trigram noise scores zero. Motivated by
  handoff-mcp's `memory_query` relevance precision.
  - `tokenize_weighted(text) -> Vec<WeightedToken>`: same token multiset as
    `tokenize()`, each token weighted by the particle following it —
    `Xは`/`Xが` → `TOPIC_BOOST` (2.0), `Xを` → `OBJECT_BOOST` (1.8),
    `Xで`/`Xに`/`Xから`/`Xへ`/`Xまで`/`Xより` → `CASE_BOOST` (1.5); stopwords
    and CL-CnG trigrams → 0.0. An identifier's sub-tokens share its boost
    (`atomic_write は` boosts `atomic_write`, `atomic`, `write`).
  - `Corpus::build_weighted(docs)`: corpus with stopwords and CL-CnG trigrams
    excluded from TF/DF/document-length statistics.
  - `Corpus::bm25_scores_weighted(query)` / `bm25_scores_weighted_tokens
    (&[WeightedToken])`: BM25 where each term's contribution is multiplied by
    its weight; zero-weight tokens are skipped, duplicate terms keep their
    highest weight.
  - CLI: `lexsim bm25` accepts `"weighted": true`.
  - Existing APIs (`tokenize`, `Corpus::build`, `bm25_scores`, ...) are
    unchanged; `content_hash` is unaffected.
- **`estimate_tokens(text: &str) -> usize`** — a cheap, dependency-free
  estimate of how many model tokens a string would consume, for callers that
  need to stay within a token budget without invoking a real tokenizer.
  Heuristic: ASCII characters count at ~4 chars/token, CJK characters
  (Hiragana/Katakana/Han/Hangul) count at ~1.5 chars/token, and everything
  else (other scripts, emoji, symbols) counts at the ASCII rate. This is an
  approximation, not an exact model-tokenizer count.

### Fixed

- **The iteration mark `々` (U+3005) no longer splits off its stem.**
  `tokenize("人々")` returned `["人", "々"]` and `tokenize("佐々木")` returned
  `["佐", "々", "木"]`. The root cause was not the boundary model: `々` was
  missing from both `is_non_spacing_script` (tokenize.rs) and the segmenter's
  kanji class (features.rs), so 人々 fractured at the script-segmentation
  stage before the AdaBoost model ever saw the junction. `々` is now classed
  as kanji in both places and the model was retrained (same corpora, same
  1000 iterations), which makes the 漢字|々 junctions in the existing corpus
  trainable at all — the 9 seed sentences suffice, and the model generalizes
  to 々 words absent from the corpus (木々, 山々, 隅々, 佐々木, 代々木).
  Supplement sentences were tried and rejected: every variant regressed
  precision or word F1 (see the note in `training/context_supplement.txt`).
  Corpus metrics improved: boundary F1 0.9041 → 0.9045, boundary precision
  0.8875 → 0.8914, word F1 0.8065 → 0.8069.

  Known limitation: a kanji word directly following a 々 word can merge with
  it (時々雨 → one token). 々|漢字 boundaries are too rare in the corpus to
  learn; unlike the old behaviour, no bare `々` token leaks out. Pinned in
  `tests/segmenter_quality.rs`.

  `content_hash` values change for texts containing `々` (token stream
  differs), and `ja_segmenter.bin` was retrained.

- **`だろ` is now a stopword.** The colloquial sentence-final `だろ` (truncated
  `だろう`) is emitted as a standalone token whenever a Japanese run ends at
  punctuation, ASCII, emoji, or end-of-text: `これはバグだろ！` =>
  `[これ, は, バグ, だろ]`, `そうだろ？` => `[そう, だろ]`, `無理だろw` =>
  `[無理, だろ, w]`. x-metrics measured it leaking into keyword output in 9 of
  12 natural colloquial tweets. No content word is spelled exactly `だろ`, and
  the full form `だろう` (already a stopword) is unaffected.

  This fix by itself leaves `content_hash` unchanged — stopwords only apply
  at the extraction stage (but see the `々` fix above, which does change it).

## 0.5.1

### Fixed

- **`なく` is now a stopword.** 0.5.0 excluded it on the mistaken assumption
  that it only ever merges into `なくなった`. It does not: the segmenter emits
  it standalone in `問題 なく 動作`, `仕方 なく 実行` and `時間 が なく なった`
  (6 standalone occurrences in the training corpus). The adjectival `少なく` and
  the kanji-spelled `無く` / `無くした` are separate tokens and are unaffected.

`content_hash` is unchanged — the segmenter is untouched and stopwords only
apply at the extraction stage.

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
