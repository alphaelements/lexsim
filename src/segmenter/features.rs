//! Character classification and the 42-feature boundary-candidate template,
//! re-implemented from the litsea/TinySegmenter design (not vendored code —
//! see the segmenter design spec §3.2-3.3 for the template catalogue).
//!
//! A "boundary candidate" is the gap between character `i` and character
//! `i + 1` in a run of text. For each candidate we look at a fixed window of
//! surrounding characters/char-classes/prior boundary decisions and emit a set
//! of string feature keys (e.g. `"UW1:猫"`, `"UC3:H"`). The AdaBoost learner
//! treats each distinct feature key as a weak learner.

/// litsea-style 8-way character classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CharClass {
    /// Kanji numerals (一二三...十百千万億兆).
    KanjiNumeral,
    /// Han ideographs (CJK Unified Ideographs block), excluding kanji numerals.
    Kanji,
    /// Hiragana (U+3040-309F).
    Hiragana,
    /// Katakana (U+30A0-30FF, U+31F0-31FF).
    Katakana,
    /// Punctuation (。、！？「」（）etc).
    Punctuation,
    /// ASCII letters.
    Ascii,
    /// ASCII digits.
    Digit,
    /// Anything else.
    Other,
}

impl CharClass {
    /// Single-letter code used in feature keys (`UC1:I` etc), matching the
    /// litsea-style 8-class scheme: M/H/I/K/P/A/N/O.
    pub fn code(self) -> char {
        match self {
            CharClass::KanjiNumeral => 'M',
            CharClass::Kanji => 'H',
            CharClass::Hiragana => 'I',
            CharClass::Katakana => 'K',
            CharClass::Punctuation => 'P',
            CharClass::Ascii => 'A',
            CharClass::Digit => 'N',
            CharClass::Other => 'O',
        }
    }
}

/// Kanji numerals recognized as the `M` class (excluded from plain `H` kanji).
const KANJI_NUMERALS: &[char] = &[
    '一', '二', '三', '四', '五', '六', '七', '八', '九', '十', '百', '千', '万', '億', '兆',
];

/// Punctuation recognized as the `P` class (ASCII + common Japanese
/// punctuation/bracket marks).
fn is_punctuation(c: char) -> bool {
    matches!(
        c,
        '。' | '、'
            | '！'
            | '？'
            | '「'
            | '」'
            | '『'
            | '』'
            | '（'
            | '）'
            | '・'
            | '…'
            | '　'
            | '\u{FF0C}' // fullwidth comma
            | '.' | ',' | '!' | '?' | '(' | ')' | ':' | ';' | '"' | '\''
    ) || c.is_whitespace()
        || c.is_ascii_punctuation()
}

/// Classify a single character into the litsea-style 8-way scheme.
pub fn classify_char(c: char) -> CharClass {
    if KANJI_NUMERALS.contains(&c) {
        return CharClass::KanjiNumeral;
    }
    if is_punctuation(c) {
        return CharClass::Punctuation;
    }
    if is_hiragana(c) {
        return CharClass::Hiragana;
    }
    if is_katakana(c) {
        return CharClass::Katakana;
    }
    if is_kanji(c) {
        return CharClass::Kanji;
    }
    if c.is_ascii_alphabetic() {
        return CharClass::Ascii;
    }
    if c.is_ascii_digit() {
        return CharClass::Digit;
    }
    CharClass::Other
}

fn is_hiragana(c: char) -> bool {
    matches!(c as u32, 0x3040..=0x309F)
}

fn is_katakana(c: char) -> bool {
    matches!(c as u32, 0x30A0..=0x30FF | 0x31F0..=0x31FF)
}

fn is_kanji(c: char) -> bool {
    matches!(c as u32, 0x3400..=0x4DBF | 0x4E00..=0x9FFF | 0xF900..=0xFAFF)
}

/// Placeholder character used to pad the context window past the start/end
/// of a run.
const PAD_CHAR: char = 'B';
/// Placeholder character used to pad the context window past the end of a
/// run. litsea/TinySegmenter conventionally use distinct begin/end markers;
/// lexsim uses `B` before the run and `E` after it.
const PAD_CHAR_END: char = 'E';
/// Placeholder char-class code for padding positions (outside the run).
const PAD_CLASS: char = 'O';

/// A resolved boundary decision at a prior candidate position, expressed as
/// litsea does: `B` (boundary) or `O` (no boundary / "other").
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PriorDecision {
    Boundary,
    NoBoundary,
}

impl PriorDecision {
    fn code(self) -> char {
        match self {
            PriorDecision::Boundary => 'B',
            PriorDecision::NoBoundary => 'O',
        }
    }
}

/// Extract the 42 feature-template keys for the boundary candidate that sits
/// between `chars[i]` and `chars[i + 1]`.
///
/// `prior` gives the last up-to-3 boundary decisions for candidates before
/// this one (nearest first, i.e. `prior[0]` is the decision immediately to the
/// left of this candidate). Missing entries (near the start of the run) are
/// treated as `O` via [`PAD_CLASS`]-equivalent padding, matching litsea's
/// begin-of-run convention.
pub fn extract_features(chars: &[char], i: usize, prior: &[PriorDecision]) -> Vec<String> {
    debug_assert!(
        i + 1 < chars.len(),
        "candidate must have a char on each side"
    );

    let char_at = |offset: isize| -> char {
        let idx = i as isize + offset;
        if idx < 0 {
            PAD_CHAR
        } else if idx as usize >= chars.len() {
            PAD_CHAR_END
        } else {
            chars[idx as usize]
        }
    };
    let class_at = |offset: isize| -> char {
        let idx = i as isize + offset;
        if idx < 0 || idx as usize >= chars.len() {
            PAD_CLASS
        } else {
            classify_char(chars[idx as usize]).code()
        }
    };
    let prior_at = |back: usize| -> char { prior.get(back).map(|p| p.code()).unwrap_or(PAD_CLASS) };

    // Window offsets relative to the candidate gap (between i and i+1):
    // p3 p2 p1 | n1 n2 n3  (p1 = chars[i], n1 = chars[i+1])
    let p3 = char_at(-2);
    let p2 = char_at(-1);
    let p1 = char_at(0);
    let n1 = char_at(1);
    let n2 = char_at(2);
    let n3 = char_at(3);

    let cp3 = class_at(-2);
    let cp2 = class_at(-1);
    let cp1 = class_at(0);
    let cn1 = class_at(1);
    let cn2 = class_at(2);
    let cn3 = class_at(3);

    let up1 = prior_at(0);
    let up2 = prior_at(1);
    let up3 = prior_at(2);

    let mut out = Vec::with_capacity(42);

    // UW1-6: unigram window of surrounding characters.
    out.push(format!("UW1:{p3}"));
    out.push(format!("UW2:{p2}"));
    out.push(format!("UW3:{p1}"));
    out.push(format!("UW4:{n1}"));
    out.push(format!("UW5:{n2}"));
    out.push(format!("UW6:{n3}"));

    // BW1-3: adjacent character bigrams.
    out.push(format!("BW1:{p2}{p1}"));
    out.push(format!("BW2:{p1}{n1}"));
    out.push(format!("BW3:{n1}{n2}"));

    // UC1-6: unigram window of char classes.
    out.push(format!("UC1:{cp3}"));
    out.push(format!("UC2:{cp2}"));
    out.push(format!("UC3:{cp1}"));
    out.push(format!("UC4:{cn1}"));
    out.push(format!("UC5:{cn2}"));
    out.push(format!("UC6:{cn3}"));

    // BC1-3: adjacent char-class bigrams.
    out.push(format!("BC1:{cp2}{cp1}"));
    out.push(format!("BC2:{cp1}{cn1}"));
    out.push(format!("BC3:{cn1}{cn2}"));

    // TC1-4: adjacent char-class trigrams.
    out.push(format!("TC1:{cp3}{cp2}{cp1}"));
    out.push(format!("TC2:{cp2}{cp1}{cn1}"));
    out.push(format!("TC3:{cp1}{cn1}{cn2}"));
    out.push(format!("TC4:{cn1}{cn2}{cn3}"));

    // UP1-3: prior boundary decisions (unigram).
    out.push(format!("UP1:{up1}"));
    out.push(format!("UP2:{up2}"));
    out.push(format!("UP3:{up3}"));

    // BP1-2: prior boundary decisions (bigram).
    out.push(format!("BP1:{up2}{up1}"));
    out.push(format!("BP2:{up1}"));

    // UQ1-3: char-class + prior decision composite (unigram window x UP1).
    out.push(format!("UQ1:{cp3}{up1}"));
    out.push(format!("UQ2:{cp2}{up1}"));
    out.push(format!("UQ3:{cp1}{up1}"));

    // BQ1-4: char-class bigram + prior decision composite.
    out.push(format!("BQ1:{cp2}{cp1}{up1}"));
    out.push(format!("BQ2:{cp1}{cn1}{up1}"));
    out.push(format!("BQ3:{cn1}{cn2}{up1}"));
    out.push(format!("BQ4:{cp1}{up2}{up1}"));

    // TQ1-4: char-class trigram + prior decision composite.
    out.push(format!("TQ1:{cp3}{cp2}{cp1}{up1}"));
    out.push(format!("TQ2:{cp2}{cp1}{cn1}{up1}"));
    out.push(format!("TQ3:{cp1}{cn1}{cn2}{up1}"));
    out.push(format!("TQ4:{cn1}{cn2}{cn3}{up1}"));

    // WC1-4: character + char-class mixed features (Japanese-specific).
    out.push(format!("WC1:{p1}{cp1}"));
    out.push(format!("WC2:{n1}{cn1}"));
    out.push(format!("WC3:{p1}{cn1}"));
    out.push(format!("WC4:{n1}{cp1}"));

    debug_assert_eq!(out.len(), 42, "feature template must emit exactly 42 keys");
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_kanji_numeral() {
        assert_eq!(classify_char('三'), CharClass::KanjiNumeral);
        assert_eq!(classify_char('三').code(), 'M');
    }

    #[test]
    fn classify_kanji() {
        assert_eq!(classify_char('猫'), CharClass::Kanji);
        assert_eq!(classify_char('猫').code(), 'H');
    }

    #[test]
    fn classify_hiragana() {
        assert_eq!(classify_char('が'), CharClass::Hiragana);
        assert_eq!(classify_char('が').code(), 'I');
    }

    #[test]
    fn classify_katakana() {
        assert_eq!(classify_char('メ'), CharClass::Katakana);
        assert_eq!(classify_char('メ').code(), 'K');
    }

    #[test]
    fn classify_punctuation() {
        assert_eq!(classify_char('。'), CharClass::Punctuation);
        assert_eq!(classify_char('、').code(), 'P');
    }

    #[test]
    fn classify_ascii_letter() {
        assert_eq!(classify_char('a'), CharClass::Ascii);
        assert_eq!(classify_char('Z').code(), 'A');
    }

    #[test]
    fn classify_digit() {
        assert_eq!(classify_char('5'), CharClass::Digit);
        assert_eq!(classify_char('0').code(), 'N');
    }

    #[test]
    fn classify_other() {
        assert_eq!(classify_char('★'), CharClass::Other);
        assert_eq!(classify_char('★').code(), 'O');
    }

    #[test]
    fn extract_features_returns_exactly_42_keys() {
        let chars: Vec<char> = "猫が好き".chars().collect();
        let feats = extract_features(&chars, 0, &[]);
        assert_eq!(feats.len(), 42);
    }

    #[test]
    fn extract_features_known_unigram_window() {
        // "猫が好き", boundary candidate at i=1 (between 'が' and '好').
        let chars: Vec<char> = "猫が好き".chars().collect();
        let feats = extract_features(&chars, 1, &[]);
        assert!(feats.contains(&"UW3:が".to_string()));
        assert!(feats.contains(&"UW4:好".to_string()));
        assert!(feats.contains(&"UW2:猫".to_string()));
        // n3 is past the end of the 4-char run at i=1 (n3 = chars[4], out of range) -> E
        assert!(feats.contains(&"UW6:E".to_string()));
    }

    #[test]
    fn extract_features_known_class_window() {
        let chars: Vec<char> = "猫が好き".chars().collect();
        let feats = extract_features(&chars, 0, &[]);
        // p1 = '猫' (H), n1 = 'が' (I)
        assert!(feats.contains(&"UC3:H".to_string()));
        assert!(feats.contains(&"UC4:I".to_string()));
        assert!(feats.contains(&"BC2:HI".to_string()));
    }

    #[test]
    fn extract_features_begin_padding() {
        let chars: Vec<char> = "猫が".chars().collect();
        let feats = extract_features(&chars, 0, &[]);
        // p3/p2 are before the start of the run -> pad char 'B', pad class 'O'.
        assert!(feats.contains(&"UW1:B".to_string()));
        assert!(feats.contains(&"UW2:B".to_string()));
        assert!(feats.contains(&"UC1:O".to_string()));
    }

    #[test]
    fn extract_features_end_padding() {
        let chars: Vec<char> = "猫が".chars().collect();
        let feats = extract_features(&chars, 0, &[]);
        // n2/n3 are past the end of the 2-char run -> pad char 'E', pad class 'O'.
        assert!(feats.contains(&"UW5:E".to_string()));
        assert!(feats.contains(&"UW6:E".to_string()));
        assert!(feats.contains(&"UC6:O".to_string()));
    }

    #[test]
    fn extract_features_prior_decisions() {
        let chars: Vec<char> = "猫が好き".chars().collect();
        let prior = vec![PriorDecision::Boundary, PriorDecision::NoBoundary];
        let feats = extract_features(&chars, 2, &prior);
        assert!(feats.contains(&"UP1:B".to_string()));
        assert!(feats.contains(&"UP2:O".to_string()));
        // No 3rd prior decision recorded -> padded with 'O'.
        assert!(feats.contains(&"UP3:O".to_string()));
    }

    #[test]
    fn extract_features_deterministic() {
        let chars: Vec<char> = "猫が好き".chars().collect();
        let a = extract_features(&chars, 1, &[]);
        let b = extract_features(&chars, 1, &[]);
        assert_eq!(a, b);
    }
}
