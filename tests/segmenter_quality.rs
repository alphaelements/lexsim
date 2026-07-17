//! Regression gate for the Japanese boundary segmenter's accuracy.
//!
//! Retraining the model (`examples/train_segmenter.rs`) or editing
//! `training/seed_corpus.txt` can silently degrade segmentation. This test
//! pins the embedded model's accuracy against the gold corpus so a regression
//! fails CI instead of shipping.
//!
//! # Which metric gates
//!
//! The gate is on **boundary-level F1**: every internal character gap of a
//! Japanese run is one classification, positive iff gold places a word
//! boundary there. It degrades smoothly and does not depend on how runs are
//! chunked. Word-level F1 (multiset intersection of word tokens) is stricter —
//! one missed boundary spoils two words — and is asserted with a looser floor.
//!
//! Only maximal all-Japanese-script spans are scored, mirroring how
//! `tokenize()` feeds the segmenter. See `segmenter::eval` for why evaluating
//! whole mixed-script sentences would measure failures that cannot occur in
//! production.
//!
//! # Updating the floors
//!
//! The floors sit slightly below the measured baseline to absorb
//! platform/float noise, not to leave room for regressions. If a change
//! *improves* accuracy, raise them. If a change lowers accuracy, that is the
//! bug — do not lower the floor to make this pass.

use lexsim::segmenter::eval::evaluate_corpus;

/// Measured on 2026-07-17 against the model trained on
/// `seed_corpus.txt` + `context_supplement.txt`, scored over **both**:
///
/// ```text
/// runs=3280 exact=1779 (54.2%)
/// boundary P=0.8914 R=0.9179 F1=0.9045
/// word     P=0.7974 R=0.8165 F1=0.8069
/// ```
///
/// The 2026-07-10 supplement fixed the high-frequency `漢字 + です` and
/// `漢字 + たち` mis-splits that motivated it (x-metrics referral
/// ref-20260710-060424-851178046) while leaving overall accuracy flat
/// (boundary F1 0.9037 → 0.9041). The 2026-07-17 iteration-mark fix
/// (々 U+3005 classed as kanji, model retrained so 漢字|々 junctions are
/// trainable at all) nudged everything up: boundary F1 0.9041 → 0.9045,
/// word F1 0.8065 → 0.8069. Run count changed (3285 → 3280) because words
/// containing 々 now merge into their surrounding Japanese runs.
///
/// The floors sit ~0.4–1.1pt below the measured values. That margin absorbs
/// float noise, not regressions — training is deterministic (a retrain
/// reproduces `ja_segmenter.bin` byte-for-byte), so any movement here is a
/// real change in behaviour. `EXACT_RUN_RATIO_FLOOR` is the tightest of the
/// three; if a legitimate improvement pushes these up, raise the floors.
const BOUNDARY_F1_FLOOR: f64 = 0.90;
const WORD_F1_FLOOR: f64 = 0.80;
const EXACT_RUN_RATIO_FLOOR: f64 = 0.53;

/// Both training corpora, concatenated — the segmenter is trained on both, so
/// scoring only the seed would leave the supplement's contexts unguarded.
fn corpus() -> String {
    let root = env!("CARGO_MANIFEST_DIR");
    let seed = std::fs::read_to_string(format!("{root}/training/seed_corpus.txt"))
        .expect("training/seed_corpus.txt is readable");
    let supplement = std::fs::read_to_string(format!("{root}/training/context_supplement.txt"))
        .expect("training/context_supplement.txt is readable");
    format!("{seed}\n{supplement}")
}

#[test]
fn boundary_f1_does_not_regress() {
    let report = evaluate_corpus(&corpus());
    assert!(
        report.boundary.f1 >= BOUNDARY_F1_FLOOR,
        "boundary F1 regressed to {:.4} (floor {BOUNDARY_F1_FLOOR:.2}); \
         P={:.4} R={:.4} over {} runs. Retrain or fix the corpus — \
         do not lower the floor.",
        report.boundary.f1,
        report.boundary.precision,
        report.boundary.recall,
        report.runs,
    );
}

#[test]
fn word_f1_does_not_regress() {
    let report = evaluate_corpus(&corpus());
    assert!(
        report.word.f1 >= WORD_F1_FLOOR,
        "word F1 regressed to {:.4} (floor {WORD_F1_FLOOR:.2}); P={:.4} R={:.4}",
        report.word.f1,
        report.word.precision,
        report.word.recall,
    );
}

#[test]
fn exactly_segmented_run_ratio_does_not_regress() {
    let report = evaluate_corpus(&corpus());
    let ratio = report.exact_runs as f64 / report.runs as f64;
    assert!(
        ratio >= EXACT_RUN_RATIO_FLOOR,
        "exact-run ratio regressed to {ratio:.4} (floor {EXACT_RUN_RATIO_FLOOR:.2}); \
         {}/{} runs segmented exactly",
        report.exact_runs,
        report.runs,
    );
}

#[test]
fn corpus_yields_a_meaningful_number_of_runs() {
    // Guards against the corpus file being emptied/moved and the accuracy
    // assertions passing vacuously on zero runs.
    let report = evaluate_corpus(&corpus());
    assert!(
        report.runs > 3000,
        "expected >3000 scored Japanese runs, got {}",
        report.runs
    );
}

/// Word tokens `tokenize()` produces, dropping the character-n-gram entries.
fn words(text: &str) -> Vec<String> {
    lexsim::tokenize(text)
        .into_iter()
        .filter(|t| !lexsim::is_cl_ngram(t))
        .collect()
}

#[test]
fn kanji_followed_by_desu_is_one_token() {
    // Aggregate F1 floors are too coarse to catch a specific high-frequency
    // pattern regressing, so pin the mis-splits that `context_supplement.txt`
    // was written to fix (x-metrics referral ref-20260710-060424-851178046).
    // Before the supplement, `記事です` segmented as ["記事", "で", "す"],
    // leaking a bare `す` into keyword extraction.
    for text in ["これは記事です", "便利です", "動物です"] {
        let got = words(text);
        assert!(
            got.contains(&"です".to_string()),
            "{text} should keep `です` whole, got {got:?}"
        );
        assert!(
            !got.contains(&"す".to_string()),
            "{text} must not emit a bare `す`, got {got:?}"
        );
    }
}

#[test]
fn kanji_followed_by_tachi_is_one_token() {
    // `子供たち` used to segment as ["子供", "た", "ち"]. The plural suffix
    // glues to its stem per the corpus's suffix rule (seed_corpus.txt:11).
    for text in ["子供たち", "学生たち", "私たち"] {
        let got = words(text);
        assert_eq!(got, vec![text.to_string()], "{text} should be one token");
    }
}

#[test]
fn iteration_mark_words_are_not_split() {
    // 々 (U+3005, ideographic iteration mark) repeats the preceding kanji and
    // never starts a word. It used to be excluded from both the tokenizer's
    // non-spacing script ranges and the segmenter's kanji class, so 人々
    // fractured into ["人", "々"] at the script-segmentation stage — before
    // the boundary model even ran.
    for text in ["人々", "島々", "日々", "時々", "我々", "佐々木"] {
        let got = words(text);
        assert_eq!(
            got,
            vec![text.to_string()],
            "{text} should be one token, got {got:?}"
        );
    }
}

#[test]
fn iteration_mark_words_survive_sentence_context() {
    // The same words must stay whole mid-sentence, next to particles — not
    // only in isolation. 山々/木々/隅々 do not appear in either training
    // corpus, so they also pin generalization to unseen 々 words.
    for (text, word) in [
        ("街の人々が集まった", "人々"),
        ("島々を巡る旅", "島々"),
        ("佐々木さんに会った", "佐々木"),
        ("我々の日々の記録", "我々"),
        ("我々の日々の記録", "日々"),
        ("山々が連なる", "山々"),
        ("木々の間から光が差す", "木々"),
        ("隅々まで掃除した", "隅々"),
    ] {
        let got = words(text);
        assert!(
            got.contains(&word.to_string()),
            "{text} should keep {word} whole, got {got:?}"
        );
        assert!(
            !got.contains(&"々".to_string()),
            "{text} must not emit a bare 々, got {got:?}"
        );
    }
}

#[test]
fn iteration_mark_stays_glued_even_when_kanji_compounds_merge() {
    // Known limitation: a kanji word directly after a 々 word can merge with
    // it (時々雨 → one token) — 々|漢字 boundaries are too rare in the corpus
    // for the learner to pick their features, and supplement sentences did
    // not fix it (6 targeted lines left 時々雨 merged and cost 0.9pt of
    // boundary F1; see training/context_supplement.txt). The contract pinned
    // here is what the fix guarantees: 々 never splits off its stem, in
    // either direction.
    for (text, stem) in [
        ("時々雨が降る", "時々"),
        ("年々増加している", "年々"),
        ("日々努力を続ける", "日々"),
    ] {
        let got = words(text);
        assert!(
            got.iter().any(|w| w.starts_with(stem)),
            "{text}: some token should start with {stem}, got {got:?}"
        );
        assert!(
            got.iter().all(|w| w != "々" && !w.starts_with('々')),
            "{text}: 々 must never be split off its stem, got {got:?}"
        );
    }
}

#[test]
fn hiragana_stem_before_desu_still_segments_correctly() {
    // Regression guard in the other direction: the contexts that already
    // worked before the supplement must keep working.
    assert_eq!(words("幸いです"), vec!["幸いです".to_string()]);
    assert_eq!(words("猫です"), vec!["猫".to_string(), "です".to_string()]);
}

#[test]
fn sahen_noun_keeps_shita_as_a_separate_token() {
    // The corpus deliberately splits 漢語サ変動詞 (noun + する/した), which is
    // why `した` is a stopword. If a retrain glued them, `実施した` would become
    // one content-bearing token and `is_stopword("した")` would stop mattering.
    assert_eq!(
        words("実施した結果"),
        vec!["実施".to_string(), "した".to_string(), "結果".to_string()]
    );
}

#[test]
fn inflected_verbs_remain_single_tokens() {
    // Commit 846c779's contract: 活用形は1トークン (助詞のみ分離).
    assert_eq!(
        words("昨日は雨が降りました"),
        vec!["昨日", "は", "雨", "が", "降りました"]
            .into_iter()
            .map(String::from)
            .collect::<Vec<_>>()
    );
}

#[test]
fn sentence_final_daro_is_a_standalone_token() {
    // Colloquial sentence-final `だろ` (truncated `だろう`) becomes a
    // standalone token whenever the Japanese run ends there — at punctuation,
    // ASCII, emoji, or end-of-text (x-metrics: leaks in 9/12 natural
    // colloquial tweets). This is why `is_stopword("だろ")` must be true.
    assert_eq!(
        words("これはバグだろ！"),
        vec!["これ", "は", "バグ", "だろ"]
    );
    assert_eq!(words("そうだろ？"), vec!["そう", "だろ"]);
    assert_eq!(words("無理だろw"), vec!["無理", "だろ", "w"]);
    assert!(words("これはバグだろ！")
        .iter()
        .any(|t| lexsim::is_stopword(t) && t == "だろ"));
}

#[test]
fn mid_run_darou_still_segments_as_darou() {
    // Regression guard: adding `だろ` to the stopword list must not change
    // tokenization (stopwords are extraction-stage only), and the full form
    // `だろう` keeps being emitted — and filtered — where the run continues.
    assert_eq!(words("そうだろう"), vec!["そう", "だろう"]);
    assert!(lexsim::is_stopword("だろう"));
}
