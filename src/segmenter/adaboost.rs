//! Deterministic AdaBoost binary classifier, re-implemented from the
//! TinySegmenter/litsea design (not vendored code). No `rand` dependency: each
//! boosting iteration deterministically scans every candidate feature and
//! picks the one weak learner (a single feature's presence) that minimizes the
//! current weighted error.
//!
//! # Weak learner
//!
//! Each weak learner is "does feature `f` fire for this sample?" — if it
//! fires, predict `+1` (boundary), else predict `-1` (no boundary), or the
//! polarity-flipped version of that rule (whichever has lower weighted error).
//! This exactly mirrors the design's feature-key weak-learner scheme: the
//! trained model is a `HashMap<String, f64>` of feature weights plus a bias,
//! and inference is a linear sum of the weights of the features that fire.
//!
//! # Determinism
//!
//! There is no randomness anywhere in this module. Sample weights start
//! uniform, and each iteration's feature scan is over a fixed, sorted feature
//! list, so ties are broken deterministically (first feature in sorted order
//! wins). Training the same corpus twice yields bit-identical output.

use std::collections::{HashMap, HashSet};

/// One labeled training sample as boolean feature membership: the set of
/// feature keys that fire for this boundary candidate, and the gold label.
pub struct Sample {
    pub features: Vec<String>,
    /// `+1.0` for boundary, `-1.0` for no boundary.
    pub label: f64,
}

/// A trained model: per-feature additive weight plus a bias term. Inference
/// score = `bias + sum(weights[f] for f in active_features)`; boundary iff
/// score > 0.
#[derive(Debug, Clone, Default)]
pub struct Model {
    pub bias: f64,
    pub weights: HashMap<String, f64>,
}

impl Model {
    /// Linear decision score for a set of active feature keys.
    pub fn score(&self, features: &[String]) -> f64 {
        let mut s = self.bias;
        for f in features {
            if let Some(w) = self.weights.get(f) {
                s += w;
            }
        }
        s
    }

    /// `true` if the score is positive (boundary predicted).
    pub fn predict(&self, features: &[String]) -> bool {
        self.score(features) > 0.0
    }
}

/// Train a deterministic AdaBoost classifier.
///
/// `iterations` boosting rounds are run; each round selects the single
/// feature (and polarity) that minimizes the current sample-weighted error,
/// adds it to the model with weight `alpha = 0.5 * ln((1 - err) / err)`, and
/// reweights samples for the next round. Bias starts at 0 and is not updated
/// by boosting rounds (matches the design's "decision threshold at 0" scheme;
/// the bias field is kept in [`Model`] for forward compatibility with a
/// future calibration step).
pub fn train(samples: &[Sample], iterations: usize) -> Model {
    let n = samples.len();
    assert!(n > 0, "cannot train on an empty sample set");

    // Deterministic, sorted universe of candidate features.
    let mut feature_universe: Vec<String> = {
        let mut set: HashSet<&str> = HashSet::new();
        for s in samples {
            for f in &s.features {
                set.insert(f.as_str());
            }
        }
        set.into_iter().map(|s| s.to_string()).collect()
    };
    feature_universe.sort();

    // Precompute, for each feature, the sample indices where it fires — avoids
    // rescanning every sample's Vec<String> on every iteration. Features are
    // typically sparse (each fires on a small fraction of samples), so per
    // iteration we only touch the firing samples, not the full sample set.
    let mut fires_in: HashMap<&str, Vec<usize>> = HashMap::new();
    for (idx, s) in samples.iter().enumerate() {
        for f in &s.features {
            fires_in.entry(f.as_str()).or_default().push(idx);
        }
    }

    let mut sample_weights = vec![1.0 / n as f64; n];
    let mut weights: HashMap<String, f64> = HashMap::new();

    for _ in 0..iterations {
        // Total weight of positive-label samples, recomputed once per
        // iteration (O(n), not O(n * |features|)).
        let total_pos_weight: f64 = samples
            .iter()
            .zip(&sample_weights)
            .filter(|(s, _)| s.label > 0.0)
            .map(|(_, w)| w)
            .sum();

        let mut best_feature: Option<&str> = None;
        let mut best_polarity = 1.0_f64;
        let mut best_error = f64::INFINITY;

        for feature in &feature_universe {
            let idxs = fires_in
                .get(feature.as_str())
                .map(Vec::as_slice)
                .unwrap_or(&[]);

            // err_pos = weighted error of the rule "fires => +1, else => -1".
            // Derivation: samples NOT firing but labeled +1 contribute
            // (total_pos_weight - firing_pos_weight); samples firing but
            // labeled -1 contribute (firing_weight - firing_pos_weight).
            let mut firing_weight = 0.0;
            let mut firing_pos_weight = 0.0;
            for &idx in idxs {
                let w = sample_weights[idx];
                firing_weight += w;
                if samples[idx].label > 0.0 {
                    firing_pos_weight += w;
                }
            }
            let err_pos =
                (total_pos_weight - firing_pos_weight) + (firing_weight - firing_pos_weight);
            let err_neg = 1.0 - err_pos;

            let (err, polarity) = if err_pos <= err_neg {
                (err_pos, 1.0)
            } else {
                (err_neg, -1.0)
            };

            if err < best_error {
                best_error = err;
                best_feature = Some(feature.as_str());
                best_polarity = polarity;
            }
        }

        let Some(feature) = best_feature else {
            break;
        };

        // Clamp to avoid division by zero / infinite alpha on perfectly
        // separable toy data.
        let err = best_error.clamp(1e-6, 1.0 - 1e-6);
        let alpha = 0.5 * ((1.0 - err) / err).ln();
        let signed_alpha = alpha * best_polarity;

        *weights.entry(feature.to_string()).or_insert(0.0) += signed_alpha;

        // Reweight samples: only the firing samples' predictions differ from
        // `-best_polarity`, so we can update all samples in one pass and flip
        // just the firing subset without building a HashSet.
        let firing: HashSet<usize> = fires_in
            .get(feature)
            .map(|v| v.iter().copied().collect())
            .unwrap_or_default();
        let mut z = 0.0;
        for (idx, s) in samples.iter().enumerate() {
            let predicted = if firing.contains(&idx) {
                best_polarity
            } else {
                -best_polarity
            };
            let new_w = sample_weights[idx] * (-alpha * s.label * predicted).exp();
            sample_weights[idx] = new_w;
            z += new_w;
        }
        for w in sample_weights.iter_mut() {
            *w /= z;
        }
    }

    Model { bias: 0.0, weights }
}

/// Evaluation metrics for a trained model against a set of samples.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Metrics {
    pub accuracy: f64,
    pub precision: f64,
    pub recall: f64,
}

/// Compute accuracy/precision/recall of `model` on `samples`.
pub fn evaluate(model: &Model, samples: &[Sample]) -> Metrics {
    let mut tp = 0.0;
    let mut fp = 0.0;
    let mut fn_ = 0.0;
    let mut tn = 0.0;

    for s in samples {
        let predicted = model.predict(&s.features);
        let actual = s.label > 0.0;
        match (predicted, actual) {
            (true, true) => tp += 1.0,
            (true, false) => fp += 1.0,
            (false, true) => fn_ += 1.0,
            (false, false) => tn += 1.0,
        }
    }

    let total = tp + fp + fn_ + tn;
    let accuracy = if total > 0.0 { (tp + tn) / total } else { 0.0 };
    let precision = if tp + fp > 0.0 { tp / (tp + fp) } else { 0.0 };
    let recall = if tp + fn_ > 0.0 { tp / (tp + fn_) } else { 0.0 };

    Metrics {
        accuracy,
        precision,
        recall,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(features: &[&str], label: f64) -> Sample {
        Sample {
            features: features.iter().map(|s| s.to_string()).collect(),
            label,
        }
    }

    #[test]
    fn converges_on_linearly_separable_toy_data() {
        // "boundary_marker" perfectly predicts +1, "no_boundary_marker" -1.
        let mut samples = Vec::new();
        for _ in 0..20 {
            samples.push(sample(&["boundary_marker", "common"], 1.0));
            samples.push(sample(&["no_boundary_marker", "common"], -1.0));
        }

        let model = train(&samples, 20);
        let metrics = evaluate(&model, &samples);
        assert!(
            metrics.accuracy > 0.9,
            "expected accuracy > 0.9, got {}",
            metrics.accuracy
        );
    }

    #[test]
    fn deterministic_across_runs() {
        let mut samples = Vec::new();
        for _ in 0..10 {
            samples.push(sample(&["a", "common"], 1.0));
            samples.push(sample(&["b", "common"], -1.0));
        }

        let model1 = train(&samples, 10);
        let model2 = train(&samples, 10);

        assert_eq!(model1.bias, model2.bias);
        let mut w1: Vec<_> = model1.weights.iter().collect();
        let mut w2: Vec<_> = model2.weights.iter().collect();
        w1.sort_by(|a, b| a.0.cmp(b.0));
        w2.sort_by(|a, b| a.0.cmp(b.0));
        assert_eq!(w1, w2);
    }

    #[test]
    fn predict_uses_sign_of_linear_score() {
        let mut model = Model::default();
        model.weights.insert("f1".to_string(), 1.0);
        model.weights.insert("f2".to_string(), -2.0);

        assert!(model.predict(&["f1".to_string()]));
        assert!(!model.predict(&["f2".to_string()]));
        assert!(!model.predict(&["f1".to_string(), "f2".to_string()]));
    }

    #[test]
    fn evaluate_computes_expected_metrics() {
        let model = Model {
            bias: 0.0,
            weights: [("pos".to_string(), 1.0)].into_iter().collect(),
        };
        let samples = vec![
            sample(&["pos"], 1.0),  // TP
            sample(&["pos"], -1.0), // FP
            sample(&[], 1.0),       // FN
            sample(&[], -1.0),      // TN
        ];
        let metrics = evaluate(&model, &samples);
        assert!((metrics.accuracy - 0.5).abs() < 1e-9);
        assert!((metrics.precision - 0.5).abs() < 1e-9);
        assert!((metrics.recall - 0.5).abs() < 1e-9);
    }

    #[test]
    #[should_panic(expected = "empty sample set")]
    fn train_on_empty_samples_panics() {
        train(&[], 10);
    }
}
