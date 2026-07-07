//! Training pipeline and runtime model for the Japanese boundary segmenter.
//!
//! The training-side sub-modules ([`features`], [`adaboost`], [`corpus`]) are
//! only consumed by the training tool (`examples/train_segmenter.rs`, gated
//! behind the `cli` feature) — they are compiled into the crate so they can be
//! unit tested with `cargo test`, but nothing in `src/tokenize.rs` depends on
//! them. [`binary`] and the embedded [`MODEL`] are the runtime side: the
//! trained model is encoded to a fixed-length binary format, baked into the
//! crate with `include_bytes!`, and exposed as a zero-copy [`binary::ModelView`].
//! The tokenizer itself does not call into the segmenter yet (that wiring
//! lands in a later phase per the segmenter design spec).
//!
//! Sub-modules:
//! - [`features`]: litsea-style 8-class character classification + the
//!   42-feature template extractor.
//! - [`adaboost`]: a from-scratch, deterministic AdaBoost binary classifier
//!   (no `rand` dependency — each iteration is a full deterministic scan).
//! - [`corpus`]: litsea-compatible wakachi (space-segmented) corpus loading,
//!   Japanese run extraction, and boundary-label generation.
//! - [`binary`]: fixed-length binary model encoder/decoder (no `serde`
//!   dependency), used to bake the trained model into the crate binary.
//! - [`inference`]: runtime Japanese-run segmentation using [`MODEL`], with a
//!   character-bigram fallback for non-Japanese non-spacing scripts (Hangul,
//!   etc.). Not yet wired into `tokenize()`.

pub mod adaboost;
pub mod binary;
pub mod corpus;
pub mod features;
pub mod inference;

pub use inference::push_segmented_ja;

/// The trained segmenter model, baked into the crate binary at compile time.
///
/// Built from `src/model_data/ja_segmenter.bin`, which is produced by
/// `examples/train_segmenter.rs --bin-output ...` (see that tool's docs for
/// the training command). Parsing happens once, lazily, on first access.
static MODEL_BYTES: &[u8] = include_bytes!("../model_data/ja_segmenter.bin");

/// Lazily-parsed zero-copy view over [`MODEL_BYTES`].
pub static MODEL: std::sync::LazyLock<binary::ModelView<'static>> =
    std::sync::LazyLock::new(|| {
        binary::ModelView::from_bytes(MODEL_BYTES).expect("embedded model is valid")
    });

#[cfg(test)]
mod tests {
    use super::MODEL;

    #[test]
    fn embedded_model_parses_and_is_non_empty() {
        // The include_bytes!-baked model (produced by `train_segmenter
        // --bin-output`) must parse successfully via the lazily-initialized
        // static and expose a non-trivial trained vocabulary.
        assert!(!MODEL.is_empty(), "embedded model has zero feature entries");
    }

    #[test]
    fn embedded_model_lookup_is_callable() {
        // Any feature key is a valid lookup argument; unseen keys resolve to
        // 0.0 rather than panicking.
        let score = MODEL.lookup("this-feature-key-almost-certainly-does-not-exist");
        assert_eq!(score, 0.0);
    }
}
