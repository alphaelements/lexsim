//! Dictionary-based sentiment polarity classification.
//!
//! Uses built-in positive/negative word lists for English and Japanese.
//! No ML — purely lexical matching against the crate's tokenizer output.

use crate::tokenize::tokenize;
use std::collections::HashSet;

/// Sentiment polarity of a text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Polarity {
    Positive,
    Neutral,
    Negative,
}

impl Polarity {
    pub fn as_str(self) -> &'static str {
        match self {
            Polarity::Positive => "positive",
            Polarity::Neutral => "neutral",
            Polarity::Negative => "negative",
        }
    }
}

/// Result of sentiment analysis for a single text.
#[derive(Debug, Clone)]
pub struct SentimentResult {
    pub polarity: Polarity,
    pub confidence: f64,
    pub positive_count: u32,
    pub negative_count: u32,
}

// English positive words (common sentiment lexicon subset)
const EN_POSITIVE: &[&str] = &[
    "good",
    "great",
    "excellent",
    "amazing",
    "wonderful",
    "fantastic",
    "awesome",
    "best",
    "love",
    "loved",
    "happy",
    "beautiful",
    "perfect",
    "brilliant",
    "outstanding",
    "superb",
    "nice",
    "impressive",
    "enjoy",
    "enjoyed",
    "helpful",
    "useful",
    "recommend",
    "recommended",
    "easy",
    "fast",
    "powerful",
    "elegant",
    "clean",
    "reliable",
    "efficient",
    "innovative",
    "success",
    "successful",
    "win",
    "winning",
    "strong",
    "exciting",
    "fun",
    "cool",
    "better",
    "improved",
    "improvement",
    "advantage",
    "benefit",
    "safe",
    "secure",
    "stable",
    "simple",
    "clear",
    "intuitive",
    "smart",
    "clever",
];

// English negative words
const EN_NEGATIVE: &[&str] = &[
    "bad",
    "terrible",
    "horrible",
    "awful",
    "worst",
    "hate",
    "hated",
    "ugly",
    "poor",
    "broken",
    "fail",
    "failed",
    "failure",
    "error",
    "bug",
    "bugs",
    "crash",
    "crashed",
    "slow",
    "complex",
    "complicated",
    "confusing",
    "confused",
    "difficult",
    "hard",
    "problem",
    "problems",
    "issue",
    "issues",
    "annoying",
    "frustrating",
    "disappointed",
    "disappointing",
    "unfortunately",
    "wrong",
    "worse",
    "painful",
    "useless",
    "unstable",
    "insecure",
    "unsafe",
    "weak",
    "messy",
    "bloated",
    "lacking",
    "missing",
    "deprecated",
    "warning",
    "warnings",
    "risk",
    "risky",
    "danger",
    "dangerous",
];

// Japanese positive words
const JA_POSITIVE: &[&str] = &[
    "良い",
    "いい",
    "素晴らしい",
    "最高",
    "優れ",
    "便利",
    "快適",
    "嬉しい",
    "楽しい",
    "好き",
    "美しい",
    "綺麗",
    "安心",
    "安全",
    "簡単",
    "効率",
    "改善",
    "成功",
    "最適",
    "優秀",
    "感謝",
    "おすすめ",
    "推奨",
    "満足",
    "完璧",
    "強力",
    "高速",
    "安定",
    "シンプル",
    "スマート",
    "クリーン",
    "エレガント",
];

// Japanese negative words
const JA_NEGATIVE: &[&str] = &[
    "悪い",
    "ダメ",
    "駄目",
    "最悪",
    "問題",
    "不具合",
    "バグ",
    "エラー",
    "遅い",
    "難しい",
    "複雑",
    "面倒",
    "不便",
    "危険",
    "不安",
    "心配",
    "失敗",
    "残念",
    "困る",
    "困った",
    "辛い",
    "壊れ",
    "落ちる",
    "クラッシュ",
    "不安定",
    "非推奨",
    "警告",
    "リスク",
    "欠点",
    "弱い",
    "不足",
    "不満",
];

type SentimentDicts = (
    HashSet<String>,
    HashSet<String>,
    HashSet<String>,
    HashSet<String>,
);

fn build_set(words: &[&str]) -> HashSet<String> {
    words.iter().map(|w| w.to_string()).collect()
}

/// Analyze the sentiment polarity of a single text.
///
/// Tokenizes the text and counts matches against built-in positive/negative
/// dictionaries. Confidence is based on the proportion of sentiment-bearing
/// tokens relative to total word tokens.
pub fn analyze_sentiment(text: &str) -> SentimentResult {
    static DICTS: std::sync::LazyLock<SentimentDicts> = std::sync::LazyLock::new(|| {
        (
            build_set(EN_POSITIVE),
            build_set(EN_NEGATIVE),
            build_set(JA_POSITIVE),
            build_set(JA_NEGATIVE),
        )
    });
    let (pos_en, neg_en, pos_ja, neg_ja) = &*DICTS;

    let tokens = tokenize(text);

    let mut positive_count = 0u32;
    let mut negative_count = 0u32;
    let mut word_count = 0u32;

    for token in &tokens {
        if token.starts_with('\u{1}') {
            continue;
        }
        word_count += 1;

        let is_pos = pos_en.contains(token.as_str()) || contains_any(token, pos_ja);
        let is_neg = neg_en.contains(token.as_str()) || contains_any(token, neg_ja);

        if is_pos && is_neg {
            // When a token matches both polarities, prefer the longer
            // dictionary match — "不安定" (neg, 3 chars) beats "安定"
            // (pos, 2 chars) because the longer match is more specific.
            let pos_len = longest_match_len(token, pos_en, pos_ja);
            let neg_len = longest_match_len(token, neg_en, neg_ja);
            if neg_len > pos_len {
                negative_count += 1;
            } else if pos_len > neg_len {
                positive_count += 1;
            }
        } else if is_pos {
            positive_count += 1;
        } else if is_neg {
            negative_count += 1;
        }
    }

    let total_sentiment = positive_count + negative_count;

    let polarity = if total_sentiment == 0 {
        Polarity::Neutral
    } else if positive_count > negative_count {
        Polarity::Positive
    } else if negative_count > positive_count {
        Polarity::Negative
    } else {
        Polarity::Neutral
    };

    let confidence = if word_count == 0 {
        0.0
    } else if total_sentiment == 0 {
        0.5
    } else {
        let diff = (positive_count as f64 - negative_count as f64).abs();
        let ratio = diff / total_sentiment as f64;
        let coverage = total_sentiment as f64 / word_count as f64;
        // Blend: how decisive (ratio) and how much evidence (coverage)
        let raw = 0.5 + 0.5 * ratio * coverage.min(1.0);
        raw.min(1.0)
    };

    SentimentResult {
        polarity,
        confidence,
        positive_count,
        negative_count,
    }
}

/// Check if a token matches any dictionary entry. For CJK bigrams, the token
/// may be a substring of a longer dict word (e.g. "素晴" matches "素晴らしい"),
/// or vice versa.
fn contains_any(token: &str, dict: &HashSet<String>) -> bool {
    dict.iter()
        .any(|w| token.contains(w.as_str()) || w.contains(token))
}

fn longest_match_len(token: &str, en_dict: &HashSet<String>, ja_dict: &HashSet<String>) -> usize {
    let en_max = if en_dict.contains(token) {
        token.len()
    } else {
        0
    };
    let ja_max = ja_dict
        .iter()
        .filter(|w| token.contains(w.as_str()) || w.contains(token))
        .map(|w| w.len())
        .max()
        .unwrap_or(0);
    en_max.max(ja_max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn positive_english() {
        let r = analyze_sentiment("This is great and amazing work");
        assert_eq!(r.polarity, Polarity::Positive);
        assert!(r.positive_count >= 2);
        assert!(r.confidence > 0.5);
    }

    #[test]
    fn negative_english() {
        let r = analyze_sentiment("This is terrible and horrible");
        assert_eq!(r.polarity, Polarity::Negative);
        assert!(r.negative_count >= 2);
        assert!(r.confidence > 0.5);
    }

    #[test]
    fn neutral_english() {
        let r = analyze_sentiment("The function returns a value");
        assert_eq!(r.polarity, Polarity::Neutral);
    }

    #[test]
    fn positive_japanese() {
        let r = analyze_sentiment("素晴らしい機能で便利です");
        assert_eq!(r.polarity, Polarity::Positive);
        assert!(r.positive_count >= 1);
    }

    #[test]
    fn negative_japanese() {
        let r = analyze_sentiment("バグが多くて不安定です");
        assert_eq!(r.polarity, Polarity::Negative);
        assert!(r.negative_count >= 1);
    }

    #[test]
    fn mixed_sentiment_balanced() {
        let r = analyze_sentiment("good but also bad");
        assert_eq!(r.polarity, Polarity::Neutral);
    }

    #[test]
    fn empty_text() {
        let r = analyze_sentiment("");
        assert_eq!(r.polarity, Polarity::Neutral);
        assert_eq!(r.positive_count, 0);
        assert_eq!(r.negative_count, 0);
    }

    #[test]
    fn mixed_language() {
        let r = analyze_sentiment("This is a great 素晴らしい tool");
        assert_eq!(r.polarity, Polarity::Positive);
        assert!(r.positive_count >= 2);
    }

    #[test]
    fn polarity_as_str() {
        assert_eq!(Polarity::Positive.as_str(), "positive");
        assert_eq!(Polarity::Neutral.as_str(), "neutral");
        assert_eq!(Polarity::Negative.as_str(), "negative");
    }

    #[test]
    fn confidence_is_bounded() {
        let r = analyze_sentiment("great great great great great");
        assert!(r.confidence >= 0.0 && r.confidence <= 1.0);
    }
}
