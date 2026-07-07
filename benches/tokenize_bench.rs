//! Performance-budget benchmarks for the hybrid segmenter (design spec §6.3).
//!
//! Three groups are measured:
//! - `tokenize()` over three script mixes (JP-only, EN-only, JP+EN mixed), to
//!   track the latency regression introduced by the trained Japanese boundary
//!   segmenter relative to the previous plain-bigram scheme (design spec §6.2
//!   budget: worst case <= 3x the bigram baseline).
//! - `Corpus::build()` at three corpus sizes (1 / 10 / 100 docs), to confirm
//!   indexing throughput is not regressed (design spec §6.2 budget: 1-doc
//!   build <= 25us).
//! - `push_segmented_ja()` in isolation, to isolate the trained-model
//!   inference cost from the rest of `tokenize()`'s script-splitting/NFKC
//!   overhead.
//!
//! Run with `cargo bench` (release build; `criterion` is a dev-dependency and
//! does not affect consumer builds). Results are written to
//! `target/criterion/**/report/index.html` and printed to stdout/stderr by
//! `criterion` itself; compare the printed mean/median against the budgets in
//! the design spec before merging changes to the segmenter.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use lexsim::segmenter::push_segmented_ja;
use lexsim::{tokenize, Corpus};

/// Pure Japanese sentence (~138 bytes with the surrounding punctuation used in
/// the design spec's benchmark corpus).
const JP_ONLY: &str = "メモリ機能を使ってデータを保存する設定画面で変更できる";

/// Pure English sentence, exercising the UAX#29 word-boundary path only.
const EN_ONLY: &str = "The quick brown fox jumps over the lazy dog";

/// Japanese/English mixed sentence (~254 bytes), exercising script switching
/// plus CL-CnG cross-language n-gram matching.
const MIXED: &str = "handoff_load_context を呼び出してセッションを復元する";

fn tokenize_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("tokenize");
    group.bench_function("tokenize_jp_only", |b| {
        b.iter(|| tokenize(black_box(JP_ONLY)))
    });
    group.bench_function("tokenize_en_only", |b| {
        b.iter(|| tokenize(black_box(EN_ONLY)))
    });
    group.bench_function("tokenize_mixed", |b| b.iter(|| tokenize(black_box(MIXED))));
    group.finish();
}

fn corpus_build_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("corpus_build");

    let doc_1: Vec<String> = vec![JP_ONLY.to_string()];
    let doc_10: Vec<String> = (0..10)
        .map(|i| format!("{JP_ONLY} {EN_ONLY} {MIXED} #{i}"))
        .collect();
    let doc_100: Vec<String> = (0..100)
        .map(|i| format!("{JP_ONLY} {EN_ONLY} {MIXED} #{i}"))
        .collect();

    group.bench_function("corpus_build_1doc", |b| {
        b.iter(|| Corpus::build(black_box(&doc_1)))
    });
    group.bench_function("corpus_build_10doc", |b| {
        b.iter(|| Corpus::build(black_box(&doc_10)))
    });
    group.bench_function("corpus_build_100doc", |b| {
        b.iter(|| Corpus::build(black_box(&doc_100)))
    });
    group.finish();
}

fn push_segmented_ja_benchmark(c: &mut Criterion) {
    c.bench_function("push_segmented_ja", |b| {
        b.iter(|| {
            let mut out = Vec::new();
            push_segmented_ja(black_box(JP_ONLY), &mut out);
            out
        })
    });
}

criterion_group!(
    benches,
    tokenize_benchmarks,
    corpus_build_benchmarks,
    push_segmented_ja_benchmark
);
criterion_main!(benches);
