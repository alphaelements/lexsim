# lexsim

[![CI](https://github.com/alphaelements/lexsim/actions/workflows/ci.yml/badge.svg)](https://github.com/alphaelements/lexsim/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/lexsim.svg)](https://crates.io/crates/lexsim)
[![docs.rs](https://docs.rs/lexsim/badge.svg)](https://docs.rs/lexsim)
[![license](https://img.shields.io/crates/l/lexsim.svg)](#license)

A **dictionary-free, multilingual lexical similarity engine** for Rust:
tokenize + Jaccard + BM25 + a stable content hash, sharing one tokenizer.

It answers two questions with the same notion of "term":

- **"Are these the same?"** → `jaccard` (symmetric set similarity) — detect
  near-duplicates (e.g. before saving a record).
- **"Is this relevant to that?"** → `Corpus::bm25_scores` (asymmetric ranking) —
  pull the items relevant to a query.

Plus `content_hash` for change-detection / re-injection tracking.

## Why dictionary-free

Morphological dictionaries (e.g. for Japanese) are multi-megabyte and
language-specific. `lexsim` instead combines Unicode-standard techniques:

- **UAX#29 word boundaries** for space-delimited scripts,
- **CJK character bi-grams** (the Apache Lucene approach) for non-spacing scripts
  (Japanese / Chinese / Korean),
- **NFKC normalization** to unify full/half-width and variant forms,
- **CL-CnG** (Cross-Language Character N-Grams) so identifiers, proper nouns, and
  spelling variants match across languages.

The result reaches dictionary-like recall with **zero dictionary**, in
sub-millisecond time for the corpus sizes this targets (tens to hundreds of short
documents). Its only dependencies are `unicode-segmentation` and
`unicode-normalization`.

## Install

```toml
[dependencies]
lexsim = "0.3"
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
| `tokenize` / `tokenize_ngrams` / `normalize` | dictionary-free, multilingual tokenizer (NFKC + UAX#29 + CJK bi-grams + CL-CnG) |
| `jaccard` / `jaccard_sets` / `token_set` | symmetric set similarity for dedup |
| `Corpus::build` / `Corpus::bm25_scores` | asymmetric BM25 ranking for retrieval |
| `Corpus::tfidf_keywords` | TF-IDF top-N keyword extraction |
| `corpus_diff` | compare two corpora for distinctive keywords |
| `analyze_sentiment` | dictionary-based sentiment polarity classification (ja/en) |
| `content_hash` / `fnv1a_hex` | stable content hashing for change detection |
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
