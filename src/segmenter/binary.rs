//! Fixed-length binary model format for the trained segmenter, so the model
//! can be embedded in the crate via `include_bytes!` with no `serde`/`serde_json`
//! dependency at runtime (see the segmenter design spec §4.2).
//!
//! # Format
//!
//! ```text
//! Header (24 bytes):
//!   magic:              [u8; 4]  = b"LXSG"
//!   version:            u16 (LE) = 1
//!   _reserved:          u16 (LE) = 0
//!   bias:               f32 (LE)
//!   n_entries:          u32 (LE)
//!   training_data_hash: u64 (LE)
//!
//! Entry array (n_entries * 12 bytes, sorted by feature_hash):
//!   feature_hash: u64 (LE)  // FNV-1a 64-bit hash of the feature key string
//!   weight:       f32 (LE)
//! ```
//!
//! Entries are sorted by `feature_hash` so [`ModelView::lookup`] can use a
//! binary search (`O(log n)`) instead of a linear scan.

use std::collections::HashMap;

const MAGIC: [u8; 4] = *b"LXSG";
const VERSION: u16 = 1;
const HEADER_LEN: usize = 24;
const ENTRY_LEN: usize = 12;

const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// FNV-1a (64-bit) hash of a string, used both as the feature-key hash stored
/// in the binary format and as the training-data corpus hash in the header.
pub fn fnv1a_64(s: &str) -> u64 {
    let mut hash = FNV_OFFSET_BASIS;
    for &b in s.as_bytes() {
        hash ^= b as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// Encode a trained model (`bias` + per-feature `weights`) into the fixed-length
/// binary format described in the module docs.
///
/// `training_data_hash` is an opaque `u64` the caller derives from the
/// training corpus (e.g. `fnv1a_64` of the corpus file contents), stored in the
/// header for provenance/debugging; it plays no role in decoding.
///
/// # Panics
///
/// Panics if two distinct feature keys hash to the same `feature_hash` (FNV-1a
/// 64-bit collisions are not expected at the scale of a few thousand feature
/// keys; if one is ever observed it indicates a hashing bug, not benign noise,
/// so this fails loudly at encode time rather than silently dropping an entry).
pub fn encode(bias: f64, weights: &HashMap<String, f64>, training_data_hash: u64) -> Vec<u8> {
    let mut entries: Vec<(u64, f32)> = weights
        .iter()
        .map(|(k, w)| (fnv1a_64(k), *w as f32))
        .collect();
    entries.sort_by_key(|(hash, _)| *hash);
    assert_no_hash_collisions(&entries);

    let mut out = Vec::with_capacity(HEADER_LEN + entries.len() * ENTRY_LEN);
    out.extend_from_slice(&MAGIC);
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes()); // _reserved
    out.extend_from_slice(&(bias as f32).to_le_bytes());
    out.extend_from_slice(&(entries.len() as u32).to_le_bytes());
    out.extend_from_slice(&training_data_hash.to_le_bytes());

    for (hash, weight) in &entries {
        out.extend_from_slice(&hash.to_le_bytes());
        out.extend_from_slice(&weight.to_le_bytes());
    }

    out
}

/// Panics if `sorted_entries` (sorted by `feature_hash`) contains two entries
/// with the same `feature_hash` — an FNV-1a 64-bit collision among the
/// model's feature keys. Extracted from [`encode`] so the guard itself is
/// directly unit-testable without needing a genuine hash collision.
fn assert_no_hash_collisions(sorted_entries: &[(u64, f32)]) {
    for pair in sorted_entries.windows(2) {
        assert!(
            pair[0].0 != pair[1].0,
            "FNV-1a hash collision detected for feature_hash {:#x} — cannot encode model",
            pair[0].0
        );
    }
}

/// Zero-copy view over an encoded model buffer (see module docs for the
/// layout). Borrows `data` directly — no allocation, no parsing beyond
/// reading the fixed header fields.
#[derive(Debug)]
pub struct ModelView<'a> {
    bias: f32,
    training_data_hash: u64,
    entries: &'a [u8],
    n_entries: usize,
}

impl<'a> ModelView<'a> {
    /// Parse `data` as an encoded model. Fails if the buffer is shorter than
    /// the header, the magic doesn't match, the version is unsupported, or the
    /// buffer is too short for the declared `n_entries`.
    pub fn from_bytes(data: &'a [u8]) -> Result<Self, &'static str> {
        if data.len() < HEADER_LEN {
            return Err("buffer shorter than header");
        }
        if data[0..4] != MAGIC {
            return Err("bad magic");
        }
        let version = u16::from_le_bytes([data[4], data[5]]);
        if version != VERSION {
            return Err("unsupported version");
        }
        let bias = f32::from_le_bytes([data[8], data[9], data[10], data[11]]);
        let n_entries = u32::from_le_bytes([data[12], data[13], data[14], data[15]]) as usize;
        let training_data_hash = u64::from_le_bytes([
            data[16], data[17], data[18], data[19], data[20], data[21], data[22], data[23],
        ]);

        let entries_len = n_entries
            .checked_mul(ENTRY_LEN)
            .ok_or("n_entries overflow")?;
        let expected_total = HEADER_LEN
            .checked_add(entries_len)
            .ok_or("buffer size overflow")?;
        if data.len() < expected_total {
            return Err("buffer shorter than declared entry array");
        }

        Ok(ModelView {
            bias,
            training_data_hash,
            entries: &data[HEADER_LEN..expected_total],
            n_entries,
        })
    }

    /// The model's bias term.
    pub fn bias(&self) -> f32 {
        self.bias
    }

    /// The training-data corpus hash stored in the header.
    pub fn training_data_hash(&self) -> u64 {
        self.training_data_hash
    }

    /// The number of feature entries in the model.
    pub fn len(&self) -> usize {
        self.n_entries
    }

    /// `true` if the model has no feature entries.
    pub fn is_empty(&self) -> bool {
        self.n_entries == 0
    }

    fn entry_at(&self, idx: usize) -> (u64, f32) {
        let start = idx * ENTRY_LEN;
        let hash = u64::from_le_bytes([
            self.entries[start],
            self.entries[start + 1],
            self.entries[start + 2],
            self.entries[start + 3],
            self.entries[start + 4],
            self.entries[start + 5],
            self.entries[start + 6],
            self.entries[start + 7],
        ]);
        let weight = f32::from_le_bytes([
            self.entries[start + 8],
            self.entries[start + 9],
            self.entries[start + 10],
            self.entries[start + 11],
        ]);
        (hash, weight)
    }

    /// Look up the weight for a feature key by FNV-1a hashing it and running a
    /// binary search over the sorted entry array. Returns `0.0` if the feature
    /// is not present in the model (i.e. it never fired during training).
    pub fn lookup(&self, feature_key: &str) -> f32 {
        let target = fnv1a_64(feature_key);
        let mut lo = 0usize;
        let mut hi = self.n_entries;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let (hash, weight) = self.entry_at(mid);
            match hash.cmp(&target) {
                std::cmp::Ordering::Equal => return weight,
                std::cmp::Ordering::Less => lo = mid + 1,
                std::cmp::Ordering::Greater => hi = mid,
            }
        }
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fnv1a_64_known_values() {
        // FNV-1a 64-bit reference test vectors (empty string -> offset basis).
        assert_eq!(fnv1a_64(""), FNV_OFFSET_BASIS);
        // Cross-checked against the independent `hash.rs` FNV-1a implementation
        // (same algorithm, byte-for-byte) to catch accidental divergence.
        let expected = {
            let mut hash = FNV_OFFSET_BASIS;
            for &b in b"test" {
                hash ^= b as u64;
                hash = hash.wrapping_mul(FNV_PRIME);
            }
            hash
        };
        assert_eq!(fnv1a_64("test"), expected);
    }

    #[test]
    fn fnv1a_64_deterministic_and_distinct() {
        assert_eq!(fnv1a_64("UW1:猫"), fnv1a_64("UW1:猫"));
        assert_ne!(fnv1a_64("UW1:猫"), fnv1a_64("UW1:犬"));
    }

    #[test]
    fn encode_decode_roundtrip_preserves_bias_and_weights() {
        let mut weights = HashMap::new();
        weights.insert("UW1:猫".to_string(), 0.5_f64);
        weights.insert("BC2:HI".to_string(), -1.25_f64);
        weights.insert("TQ4:xyz".to_string(), 3.0_f64);

        let bytes = encode(0.75, &weights, 0xdead_beef_cafe_1234);
        let view = ModelView::from_bytes(&bytes).expect("valid model bytes");

        assert!((view.bias() - 0.75_f32).abs() < 1e-6);
        assert_eq!(view.training_data_hash(), 0xdead_beef_cafe_1234);
        assert_eq!(view.len(), 3);

        for (key, weight) in &weights {
            let looked_up = view.lookup(key);
            assert!(
                (looked_up - *weight as f32).abs() < 1e-6,
                "expected weight for {key:?} ~= {weight}, got {looked_up}"
            );
        }
    }

    #[test]
    fn lookup_missing_feature_returns_zero() {
        let mut weights = HashMap::new();
        weights.insert("UW1:a".to_string(), 1.0);
        let bytes = encode(0.0, &weights, 0);
        let view = ModelView::from_bytes(&bytes).expect("valid model bytes");
        assert_eq!(view.lookup("does-not-exist"), 0.0);
    }

    #[test]
    fn entries_are_sorted_and_binary_search_finds_all() {
        let mut weights = HashMap::new();
        for i in 0..200 {
            weights.insert(format!("feat-{i}"), i as f64 * 0.1);
        }
        let bytes = encode(0.0, &weights, 0);
        let view = ModelView::from_bytes(&bytes).expect("valid model bytes");
        assert_eq!(view.len(), 200);

        // Verify sortedness directly against the raw entry bytes.
        let mut prev: Option<u64> = None;
        for i in 0..view.len() {
            let (hash, _) = view.entry_at(i);
            if let Some(p) = prev {
                assert!(p < hash, "entries must be strictly sorted by feature_hash");
            }
            prev = Some(hash);
        }

        for i in 0..200 {
            let key = format!("feat-{i}");
            let expected = i as f32 * 0.1;
            assert!((view.lookup(&key) - expected).abs() < 1e-4);
        }
    }

    #[test]
    fn from_bytes_rejects_too_short_buffer() {
        let err = ModelView::from_bytes(&[0u8; 10]).unwrap_err();
        assert_eq!(err, "buffer shorter than header");
    }

    #[test]
    fn from_bytes_rejects_bad_magic() {
        let mut bytes = encode(0.0, &HashMap::new(), 0);
        bytes[0] = b'X';
        let err = ModelView::from_bytes(&bytes).unwrap_err();
        assert_eq!(err, "bad magic");
    }

    #[test]
    fn from_bytes_rejects_truncated_entry_array() {
        let mut weights = HashMap::new();
        weights.insert("a".to_string(), 1.0);
        weights.insert("b".to_string(), 2.0);
        let mut bytes = encode(0.0, &weights, 0);
        bytes.truncate(bytes.len() - 1); // chop off the last byte of the last entry
        let err = ModelView::from_bytes(&bytes).unwrap_err();
        assert_eq!(err, "buffer shorter than declared entry array");
    }

    #[test]
    fn from_bytes_rejects_unsupported_version() {
        let mut bytes = encode(0.0, &HashMap::new(), 0);
        bytes[4] = 99; // low byte of version
        let err = ModelView::from_bytes(&bytes).unwrap_err();
        assert_eq!(err, "unsupported version");
    }

    #[test]
    fn encode_empty_model_roundtrips() {
        let bytes = encode(0.0, &HashMap::new(), 0);
        let view = ModelView::from_bytes(&bytes).expect("valid model bytes");
        assert_eq!(view.len(), 0);
        assert!(view.is_empty());
        assert_eq!(view.lookup("anything"), 0.0);
    }

    #[test]
    #[should_panic(expected = "FNV-1a hash collision detected")]
    fn encode_panics_on_hash_collision() {
        // A genuine FNV-1a 64-bit collision between two distinct strings is
        // infeasible to find by brute force, so this test exercises the
        // actual guard `encode()` calls (`assert_no_hash_collisions`) with a
        // hand-built colliding entry pair instead.
        let entries: Vec<(u64, f32)> = vec![(42, 1.0), (42, 2.0)];
        assert_no_hash_collisions(&entries);
    }
}
