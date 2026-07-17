# lexsim

[![CI](https://github.com/alphaelements/lexsim/actions/workflows/ci.yml/badge.svg)](https://github.com/alphaelements/lexsim/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/lexsim.svg)](https://crates.io/crates/lexsim)
[![docs.rs](https://docs.rs/lexsim/badge.svg)](https://docs.rs/lexsim)
[![license](https://img.shields.io/crates/l/lexsim.svg)](#license)

A **dictionary-free, multilingual lexical similarity engine** for Rust:
tokenize + Jaccard + BM25 + TextRank + a stable content hash, sharing one
tokenizer.

It answers two questions with the same notion of "term":

- **"Are these the same?"** → `jaccard` (symmetric set similarity) — detect
  near-duplicates (e.g. before saving a record).
- **"Is this relevant to that?"** → `Corpus::bm25_scores` (asymmetric ranking) —
  pull the items relevant to a query.

Plus `content_hash` for change-detection / re-injection tracking, and
`textrank_keywords` / `tfidf_keywords` / `corpus_diff` for keyword extraction.

## Hybrid word segmentation (v0.4)

Since v0.4, Japanese text is segmented at the **word level** using a built-in
AdaBoost model trained on 1,276 diverse sentences — no external dictionary
required. This replaces the character-bigram approach used in earlier versions,
dramatically improving recall for BM25, Jaccard, and keyword extraction on
Japanese content.

For non-CJK scripts, UAX#29 word boundaries are used as before.

## Why dictionary-free

Morphological dictionaries (e.g. for Japanese) are multi-megabyte and
language-specific. `lexsim` instead combines:

- **AdaBoost word segmenter** (2 KB embedded model) for Japanese, trained on a
  curated 1,276-sentence corpus with 42-feature extraction,
- **UAX#29 word boundaries** for space-delimited scripts,
- **NFKC normalization** to unify full/half-width and variant forms,
- **CL-CnG** (Cross-Language Character N-Grams) so identifiers, proper nouns, and
  spelling variants match across languages.

The result reaches dictionary-like recall with **zero external dictionary**, in
sub-millisecond time for the corpus sizes this targets (tens to hundreds of short
documents). Its only runtime dependencies are `unicode-segmentation` and
`unicode-normalization`.

## Install

```toml
[dependencies]
lexsim = "0.6"
```

## Usage

Detect a near-duplicate before saving (symmetric `jaccard`):

```rust
use lexsim::jaccard;

let existing = "always use atomic_write for handoff files";
let incoming = "always use atomic_write when writing handoff files";
assert!(jaccard(existing, incoming) > 0.5); // clearly a near-duplicate

let unrelated = "the cat sat on the mat";
assert!(jaccard(existing, unrelated) < 0.2);
```

Rank stored items by relevance to a query (asymmetric BM25 via `Corpus`):

```rust
use lexsim::Corpus;

let memories = vec![
    "always use atomic_write for handoff files".to_string(),
    "configure the milestone schedule and assignee".to_string(),
];
let corpus = Corpus::build(&memories);
let scores = corpus.bm25_scores("atomic write");
assert!(scores[0] > scores[1]); // memory 0 is the most relevant
```

The tokenizer is dictionary-free, so a Japanese query matches a Japanese
document — and an English identifier embedded in Japanese text is still found:

```rust
use lexsim::Corpus;

let memories = vec![
    "atomic_write を必ず使う（torn read 防止）".to_string(),
    "ガントチャートの表示設定".to_string(),
];
let corpus = Corpus::build(&memories);
assert!(corpus.bm25_scores("メモリ atomic_write")[0] > 0.0);
```

For higher-precision retrieval, the **particle-context weighted** variant uses
Japanese case particles as a dictionary-free stand-in for part-of-speech
tagging: `Xは`/`Xが` marks X as the topic (boost ×2.0), `Xを` as the object
(×1.8), `Xで`/`Xに`/… as case-marked complements (×1.5), while stopwords and
CL-CnG trigrams stop contributing score entirely:

```rust
use lexsim::{tokenize_weighted, Corpus, TOPIC_BOOST};

let memories = vec![
    "メモリ注入の基準はスコア上位5件".to_string(),
    "この設定はとても便利です".to_string(),
];
let corpus = Corpus::build_weighted(&memories); // stopwords/CL-CnG excluded
let scores = corpus.bm25_scores_weighted("メモリ注入の基準は？");
assert!(scores[0] > scores[1]);

// A particle-only query no longer matches anything.
assert_eq!(corpus.bm25_scores_weighted("これはですか")[0], 0.0);

// The weighted tokens themselves are public (e.g. to inject extra terms
// via Corpus::bm25_scores_weighted_tokens).
let tokens = tokenize_weighted("メモリは重要");
assert!(tokens.iter().any(|t| t.token == "メモリ" && t.weight == TOPIC_BOOST));
```

## CLI

`lexsim` also ships a CLI binary for use from other languages (e.g. via
`child_process.execFile()` in Node.js). It reads JSON from stdin and writes JSON
to stdout.

```sh
cargo install lexsim --features cli
```

Subcommands: `tokenize`, `jaccard`, `bm25`, `hash`, `keywords`, `diff`,
`sentiment`.

```sh
echo '{"texts": ["hello world"]}' | lexsim tokenize
# → {"tokens":[["hello","world",...]]}

echo '{"texts": ["hello world foo"], "ngram": 2}' | lexsim tokenize
# → includes bigrams: "hello world", "world foo"

echo '{"a": "hello world", "b": "hello there"}' | lexsim jaccard
# → {"score":0.6}

echo '{"corpus": ["atomic write", "cat mat"], "query": "atomic"}' | lexsim bm25
# → {"scores":[0.693147,0.0]}

echo '{"corpus": ["メモリ注入の基準はスコア上位5件", "この設定はとても便利です"], "query": "メモリ注入の基準は？", "weighted": true}' | lexsim bm25
# → particle-context weighted scoring (stopwords/CL-CnG excluded)

echo '{"texts": ["hello world", "Hello World"]}' | lexsim hash
# → {"hashes":["<same>","<same>"]}

echo '{"texts": ["Rust systems", "Rust safety", "Python data"], "top_n": 3}' | lexsim keywords
# → {"keywords":[{"keyword":"rust","score":0.11,"count":2}, ...]}

echo '{"corpus_a": ["Rust systems"], "corpus_b": ["Python data"]}' | lexsim diff
# → {"a_distinctive":[{"keyword":"rust","ratio":2.7}], "b_distinctive":[...]}

echo '{"texts": ["This is great!", "Terrible bug"]}' | lexsim sentiment
# → {"results":[{"text_index":0,"polarity":"positive","confidence":0.7}, ...]}
```

## Public API

| Item | Purpose |
|------|---------|
| `tokenize` / `tokenize_ngrams` / `normalize` | dictionary-free, multilingual tokenizer (NFKC + UAX#29 + hybrid word segmentation + CL-CnG) |
| `jaccard` / `jaccard_sets` / `token_set` | symmetric set similarity for dedup |
| `Corpus::build` / `Corpus::bm25_scores` | asymmetric BM25 ranking for retrieval |
| `Corpus::build_weighted` / `Corpus::bm25_scores_weighted` | particle-context weighted BM25 (topic/object boost, stopword & CL-CnG exclusion) |
| `tokenize_weighted` / `WeightedToken` | tokens with particle-context weights (`TOPIC_BOOST` / `OBJECT_BOOST` / `CASE_BOOST`) |
| `Corpus::tfidf_keywords` | TF-IDF top-N keyword extraction |
| `textrank_keywords` | graph-based TextRank keyword extraction (single-text) |
| `corpus_diff` | compare two corpora for distinctive keywords |
| `analyze_sentiment` | dictionary-based sentiment polarity classification (ja/en) |
| `content_hash` / `fnv1a_hex` | stable content hashing for change detection |
| `estimate_tokens` | cheap heuristic estimate of model-token count, for token-budget keeping |
| `is_stopword` | Japanese stopword filter (particles, auxiliaries, demonstratives) |
| `segmenter` | AdaBoost-based Japanese word segmenter (public module for advanced use) |
| `Scorer` / `LexicalScorer` | trait + lexical impl; lets an embedding-based scorer slot in later behind one call site |

## Future extension

Purely lexical matching is weak on *cross-language synonyms* (the same idea in
two languages with no shared tokens). The `Scorer` trait marks where an
embedding-based stage could be added later without touching callers; only the
lexical scorer is implemented today.

## License

MIT OR Apache-2.0.

Extracted from [handoff-mcp](https://github.com/alphaelements/handoff-mcp), where
it powers cross-session memory retrieval and dedup.
