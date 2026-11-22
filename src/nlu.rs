//! Core data types shared by the classifier, the tagger and the benchmark
//! harness, plus tokenisation, BIO conversion, inline annotation markup and
//! typo noise.

use crate::rng::SplitMix64;

/// A labelled run of tokens. `start` is inclusive, `end` exclusive.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Span {
    /// Index of the first token.
    pub start: usize,
    /// Index one past the last token.
    pub end: usize,
    /// Slot type, for example `"city"`.
    pub label: String,
}

impl Span {
    /// Creates a span covering tokens `start..end`.
    pub fn new(start: usize, end: usize, label: impl Into<String>) -> Self {
        Self {
            start,
            end,
            label: label.into(),
        }
    }
}

/// A tokenised sentence with an intent label and slot spans.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Utterance {
    /// Lower-cased word tokens.
    pub tokens: Vec<String>,
    /// Intent label; empty when the utterance carries none.
    pub intent: String,
    /// Slot spans over `tokens`.
    pub slots: Vec<Span>,
}

/// Lower-cases `text` and splits it into word tokens. Letters, digits and
/// apostrophes stay inside tokens; everything else separates them.
///
/// ```
/// assert_eq!(rslu::tokenize("What's the weather, Paris?"), ["what's", "the", "weather", "paris"]);
/// ```
pub fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !(c.is_alphabetic() || c.is_numeric() || c == '\''))
        .filter(|w| !w.is_empty())
        .map(str::to_owned)
        .collect()
}

/// Converts BIO tags (`B-x`, `I-x`, `O`) to spans. An `I-` tag whose label
/// differs from the open span, or that has no open span, starts a new one.
pub fn spans_from_bio<S: AsRef<str>>(tags: &[S]) -> Vec<Span> {
    let mut spans = Vec::new();
    let mut open: Option<(usize, &str)> = None;
    let mut flush = |open: &mut Option<(usize, &str)>, end: usize| {
        if let Some((start, label)) = open.take() {
            spans.push(Span::new(start, end, label));
        }
    };
    for (i, tag) in tags.iter().enumerate() {
        let tag = tag.as_ref();
        match tag.strip_prefix("B-") {
            Some(label) => {
                flush(&mut open, i);
                open = Some((i, label));
            }
            None => match tag.strip_prefix("I-") {
                Some(label) if open.is_some_and(|(_, l)| l == label) => {}
                Some(label) => {
                    flush(&mut open, i);
                    open = Some((i, label));
                }
                None => flush(&mut open, i),
            },
        }
    }
    flush(&mut open, tags.len());
    spans
}

/// The inverse of [`spans_from_bio`] for `n` tokens. Overlapping or
/// out-of-range spans are ignored.
pub fn bio_from_spans(n: usize, spans: &[Span]) -> Vec<String> {
    let mut tags = vec!["O".to_owned(); n];
    for s in spans {
        if s.start >= s.end || s.end > n || tags[s.start..s.end].iter().any(|t| t != "O") {
            continue;
        }
        tags[s.start] = format!("B-{}", s.label);
        for tag in &mut tags[s.start + 1..s.end] {
            *tag = format!("I-{}", s.label);
        }
    }
    tags
}

/// Turns text with inline slot markup into a training [`Utterance`]. Slots are
/// written `[label : value]`:
///
/// ```
/// let u = rslu::parse_annotated("wake me up at [time : nine am]", "alarm_set");
/// assert_eq!(u.tokens, ["wake", "me", "up", "at", "nine", "am"]);
/// assert_eq!(u.slots[0].label, "time");
/// assert_eq!((u.slots[0].start, u.slots[0].end), (4, 6));
/// ```
///
/// The text is tokenised with [`tokenize`], so spans index into the tokens.
pub fn parse_annotated(text: &str, intent: &str) -> Utterance {
    let mut u = Utterance {
        intent: intent.to_owned(),
        ..Utterance::default()
    };
    let mut last = 0;
    let mut from = 0;
    while let Some(m) = next_markup(text, from) {
        u.tokens.extend(tokenize(&text[last..m.open]));
        let value = tokenize(m.value);
        if !value.is_empty() {
            let start = u.tokens.len();
            u.slots.push(Span::new(start, start + value.len(), m.label));
        }
        u.tokens.extend(value);
        last = m.close + 1;
        from = last;
    }
    u.tokens.extend(tokenize(&text[last..]));
    u
}

struct Markup<'a> {
    open: usize,
    close: usize,
    label: &'a str,
    value: &'a str,
}

/// Finds the next `[label : value]` at or after byte offset `from`. Brackets
/// may not nest and the label may not contain `:`.
fn next_markup(text: &str, from: usize) -> Option<Markup<'_>> {
    let mut search = from;
    while let Some(rel) = text[search..].find('[') {
        let open = search + rel;
        let rest = &text[open + 1..];
        let end = rest.find(['[', ']'])?;
        if rest.as_bytes()[end] == b']'
            && let Some((label, value)) = rest[..end].split_once(':')
        {
            let (label, value) = (label.trim(), value.trim());
            if !label.is_empty() && !value.is_empty() {
                return Some(Markup {
                    open,
                    close: open + 1 + end,
                    label,
                    value,
                });
            }
        }
        search = open + 1;
    }
    None
}

/// Returns a copy of `data` in which each token longer than three characters
/// is, with probability `rate`, corrupted by a deletion or an adjacent
/// transposition. Intents, slot spans and token counts are preserved, so gold
/// labels stay valid. The same `seed` always yields the same output. It
/// measures robustness to typos and augments training data.
pub fn add_typos(data: &[Utterance], rate: f64, seed: u64) -> Vec<Utterance> {
    let mut rng = SplitMix64::new(seed);
    data.iter()
        .map(|u| {
            let tokens = u
                .tokens
                .iter()
                .map(|t| corrupt(t, rate, &mut rng))
                .collect();
            Utterance {
                tokens,
                intent: u.intent.clone(),
                slots: u.slots.clone(),
            }
        })
        .collect()
}

fn corrupt(token: &str, rate: f64, rng: &mut SplitMix64) -> String {
    let mut chars: Vec<char> = token.chars().collect();
    if chars.len() <= 3 || rng.next_f64() >= rate {
        return token.to_owned();
    }
    let p = 1 + rng.below(chars.len() - 2); // interior position
    if rng.below(2) == 0 {
        chars.remove(p);
    } else {
        chars.swap(p, p - 1);
    }
    chars.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenize_keeps_apostrophes_and_digits() {
        assert_eq!(tokenize("It's 5pm, OK?"), ["it's", "5pm", "ok"]);
        assert_eq!(tokenize("  ,;  "), Vec::<String>::new());
    }

    #[test]
    fn bio_round_trips() {
        let spans = vec![Span::new(1, 3, "city"), Span::new(4, 5, "time")];
        let tags = bio_from_spans(6, &spans);
        assert_eq!(tags, ["O", "B-city", "I-city", "O", "B-time", "O"]);
        assert_eq!(spans_from_bio(&tags), spans);
    }

    #[test]
    fn stray_i_tag_opens_a_span_and_label_change_splits() {
        let spans = spans_from_bio(&["I-a", "I-a", "I-b", "O"]);
        assert_eq!(spans, [Span::new(0, 2, "a"), Span::new(2, 3, "b")]);
    }

    #[test]
    fn overlapping_and_out_of_range_spans_are_ignored() {
        let spans = [
            Span::new(0, 2, "a"),
            Span::new(1, 3, "b"),
            Span::new(2, 9, "c"),
            Span::new(2, 2, "d"),
        ];
        assert_eq!(bio_from_spans(3, &spans), ["B-a", "I-a", "O"]);
    }

    #[test]
    fn annotated_markup_yields_token_spans() {
        let u = parse_annotated("book in [city : new york] for [party:two] now", "book");
        assert_eq!(u.tokens, ["book", "in", "new", "york", "for", "two", "now"]);
        assert_eq!(u.slots, [Span::new(2, 4, "city"), Span::new(5, 6, "party")]);
        assert_eq!(u.intent, "book");
    }

    #[test]
    fn malformed_markup_is_plain_text() {
        let u = parse_annotated("a [no colon] b [x : ] c [open", "i");
        assert_eq!(u.slots, []);
        assert_eq!(u.tokens, ["a", "no", "colon", "b", "x", "c", "open"]);
    }

    #[test]
    fn typos_preserve_structure_and_are_deterministic() {
        let data = vec![parse_annotated(
            "show flights from [city : boston]",
            "flight",
        )];
        let a = add_typos(&data, 1.0, 7);
        assert_eq!(a, add_typos(&data, 1.0, 7));
        assert_eq!(a[0].tokens.len(), data[0].tokens.len());
        assert_eq!(a[0].slots, data[0].slots);
        assert_ne!(a[0].tokens, data[0].tokens);
    }

    #[test]
    fn typos_never_touch_short_tokens() {
        let data = vec![Utterance {
            tokens: vec!["to".into(), "the".into()],
            ..Utterance::default()
        }];
        assert_eq!(add_typos(&data, 1.0, 1), data);
    }
}
