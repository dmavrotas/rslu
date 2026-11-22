//! Slot filling: a greedy left-to-right averaged-perceptron BIO tagger. Small,
//! dependency-free, deterministic, and trains in seconds on ATIS-sized corpora.

use std::collections::HashMap;

use crate::gazetteer::{Lexicon, jackknife};
use crate::nlu::{Span, Utterance, add_typos, bio_from_spans, spans_from_bio};
use crate::rng::SplitMix64;

/// Tunes tagger training. [`TaggerOptions::default`] matches the plain setup.
#[derive(Debug, Clone, PartialEq)]
pub struct TaggerOptions {
    /// Training passes over the data (default 10).
    pub epochs: usize,
    /// Add prefix/suffix (1-4) and neighbour-affix features.
    pub char_features: bool,
    /// When non-zero, also train on a copy of the data in which this fraction
    /// of words carries a typo. Trades a little clean-text accuracy on
    /// open-vocabulary data for much better robustness to misspellings.
    pub typo_augment: f64,
    /// Adds typo-tolerant lexicon features learned from the training slot
    /// values (jackknifed, so training never sees its own values).
    pub gazetteer: bool,
    /// External known values per slot type, for example
    /// `{"driver": ["Lewis Hamilton"]}`. Setting it also enables the gazetteer.
    pub entries: HashMap<String, Vec<String>>,
}

impl Default for TaggerOptions {
    fn default() -> Self {
        Self {
            epochs: 10,
            char_features: false,
            typo_augment: 0.0,
            gazetteer: false,
            entries: HashMap::new(),
        }
    }
}

/// One feature's weight for one tag, with the bookkeeping for averaging.
#[derive(Default, Clone, Copy)]
struct Cell {
    w: f64,
    total: f64,
    stamp: usize,
}

#[derive(Default)]
struct Trainer {
    cells: HashMap<String, HashMap<usize, Cell>>,
    step: usize,
}

impl Trainer {
    fn update(&mut self, feat: &str, tag: usize, delta: f64) {
        let step = self.step;
        let cell = self
            .cells
            .entry(feat.to_owned())
            .or_default()
            .entry(tag)
            .or_default();
        // lazily accumulate the weight over the steps it stayed unchanged
        cell.total += (step - cell.stamp) as f64 * cell.w;
        cell.stamp = step;
        cell.w += delta;
    }

    fn average(self) -> Weights {
        let step = self.step as f64;
        self.cells
            .into_iter()
            .filter_map(|(feat, tags)| {
                let mut avg: Vec<(usize, f64)> = tags
                    .into_iter()
                    .filter_map(|(tag, c)| {
                        let total = c.total + (self.step - c.stamp) as f64 * c.w;
                        let a = total / step;
                        (a != 0.0).then_some((tag, a))
                    })
                    .collect();
                avg.sort_unstable_by_key(|&(tag, _)| tag);
                (!avg.is_empty()).then_some((feat, avg))
            })
            .collect()
    }
}

type Weights = HashMap<String, Vec<(usize, f64)>>;

/// A trained slot tagger.
#[derive(Debug, Clone)]
pub struct Tagger {
    char_features: bool,
    lex: Option<Lexicon>,
    tags: Vec<String>,
    weights: Weights,
}

fn shape(w: &str) -> &'static str {
    let digit = w.chars().any(char::is_numeric);
    let alpha = w.chars().any(char::is_alphabetic);
    match (digit, alpha) {
        (true, true) => "mixed",
        (true, false) => "digit",
        _ => "alpha",
    }
}

/// The token `off` positions from `i`, or a boundary marker.
fn at(toks: &[String], i: usize, off: isize) -> &str {
    match i.checked_add_signed(off) {
        None => "<s>",
        Some(j) => toks.get(j).map_or("</s>", String::as_str),
    }
}

/// The first `n` characters of `w`.
fn prefix(w: &str, n: usize) -> &str {
    w.char_indices().nth(n).map_or(w, |(i, _)| &w[..i])
}

/// The last `n` characters of `w`.
fn suffix(w: &str, n: usize) -> &str {
    let len = w.chars().count();
    if len <= n {
        return w;
    }
    w.char_indices().nth(len - n).map_or(w, |(i, _)| &w[i..])
}

/// Sparse features for position `i` given the previous two tags.
fn features(
    toks: &[String],
    i: usize,
    prev: &str,
    prev2: &str,
    char_features: bool,
    gaz: Option<&[Vec<String>]>,
) -> Vec<String> {
    let w = toks[i].as_str();
    let (p1, p2, n1, n2) = (
        at(toks, i, -1),
        at(toks, i, -2),
        at(toks, i, 1),
        at(toks, i, 2),
    );
    let mut f = vec![
        "b".to_owned(),
        format!("w={w}"),
        format!("w-1={p1}"),
        format!("w-2={p2}"),
        format!("w+1={n1}"),
        format!("w+2={n2}"),
        format!("p3={}", prefix(w, 3)),
        format!("s3={}", suffix(w, 3)),
        format!("s2={}", suffix(w, 2)),
        format!("shape={}", shape(w)),
        format!("t-1={prev}"),
        format!("t-1,t-2={prev},{prev2}"),
        format!("t-1,w={prev},{w}"),
        format!("w-1,w={p1},{w}"),
        format!("w,w+1={w},{n1}"),
    ];
    if char_features {
        f.extend([
            format!("p1={}", prefix(w, 1)),
            format!("p2={}", prefix(w, 2)),
            format!("p4={}", prefix(w, 4)),
            format!("s1={}", suffix(w, 1)),
            format!("s4={}", suffix(w, 4)),
            format!("s3-1={}", suffix(p1, 3)),
            format!("s3+1={}", suffix(n1, 3)),
            format!("p3-1={}", prefix(p1, 3)),
            format!("p3+1={}", prefix(n1, 3)),
        ]);
    }
    if let Some(gaz) = gaz {
        for g in &gaz[i] {
            f.push(g.clone());
            f.push(format!("{g}|t-1={prev}"));
        }
        if let Some(left) = i.checked_sub(1).and_then(|j| gaz.get(j)) {
            f.extend(left.iter().map(|g| format!("L:{g}")));
        }
        if let Some(right) = gaz.get(i + 1) {
            f.extend(right.iter().map(|g| format!("R:{g}")));
        }
    }
    f
}

/// The best-scoring tag index. Ties go to the lowest index, and tags are
/// sorted, so decoding is deterministic.
fn best_tag<'w>(
    weights: impl Fn(&str) -> Option<&'w [(usize, f64)]>,
    feats: &[String],
    n_tags: usize,
) -> usize {
    let mut sums = vec![0.0; n_tags];
    for f in feats {
        for &(tag, v) in weights(f).into_iter().flatten() {
            sums[tag] += v;
        }
    }
    sums.iter()
        .enumerate()
        .fold(0, |best, (t, &s)| if s > sums[best] { t } else { best })
}

impl Tagger {
    /// Fits a tagger. Utterance order is shuffled each epoch with a fixed seed,
    /// so training is reproducible. Returns `None` when `data` is empty.
    pub fn train(data: &[Utterance], opts: &TaggerOptions) -> Option<Self> {
        if data.is_empty() {
            return None;
        }
        let epochs = if opts.epochs == 0 { 10 } else { opts.epochs };
        let orig_n = data.len();
        let mut all: Vec<Utterance> = data.to_vec();
        if opts.typo_augment > 0.0 {
            all.extend(add_typos(data, opts.typo_augment, 3));
        }
        let gold: Vec<Vec<String>> = all
            .iter()
            .map(|u| bio_from_spans(u.tokens.len(), &u.slots))
            .collect();
        let mut tags: Vec<String> = gold.iter().flatten().cloned().collect();
        tags.push("O".to_owned());
        tags.sort_unstable();
        tags.dedup();
        let tag_idx: HashMap<&str, usize> = tags
            .iter()
            .enumerate()
            .map(|(i, t)| (t.as_str(), i))
            .collect();

        let use_gaz = opts.gazetteer || !opts.entries.is_empty();
        let lex = use_gaz.then(|| Lexicon::build(&all[..orig_n], &opts.entries));
        let gaz = use_gaz.then(|| jackknife(&all, orig_n, &opts.entries));

        let mut tr = Trainer::default();
        let mut rng = SplitMix64::new(1);
        let mut order = rng.permutation(all.len());
        for _ in 0..epochs {
            rng.shuffle(&mut order);
            for &idx in &order {
                let toks = &all[idx].tokens;
                let (mut prev, mut prev2) = ("O", "O");
                for i in 0..toks.len() {
                    let g = gaz.as_ref().map(|g| g[idx].as_slice());
                    let feats = features(toks, i, prev, prev2, opts.char_features, g);
                    let want = tag_idx[gold[idx][i].as_str()];
                    let guess = train_guess(&tr, &feats, tags.len());
                    tr.step += 1;
                    if guess != want {
                        for f in &feats {
                            tr.update(f, want, 1.0);
                            tr.update(f, guess, -1.0);
                        }
                    }
                    // teacher forcing: condition on the gold history while training
                    prev2 = prev;
                    prev = gold[idx][i].as_str();
                }
            }
        }
        tr.step = tr.step.max(1);
        Some(Self {
            char_features: opts.char_features,
            lex,
            tags,
            weights: tr.average(),
        })
    }

    /// Labels `tokens` and returns the predicted slot spans.
    pub fn tag<S: AsRef<str>>(&self, tokens: &[S]) -> Vec<Span> {
        if tokens.is_empty() {
            return Vec::new();
        }
        let toks: Vec<String> = tokens.iter().map(|t| t.as_ref().to_owned()).collect();
        let gaz = self.lex.as_ref().map(|lx| lx.token_features(&toks));
        let mut out: Vec<&str> = Vec::with_capacity(toks.len());
        let (mut prev, mut prev2) = ("O", "O");
        for i in 0..toks.len() {
            let feats = features(&toks, i, prev, prev2, self.char_features, gaz.as_deref());
            let t = best_tag(
                |f| self.weights.get(f).map(Vec::as_slice),
                &feats,
                self.tags.len(),
            );
            prev2 = prev;
            prev = self.tags[t].as_str();
            out.push(prev);
        }
        spans_from_bio(&out)
    }
}

/// Scores against the live (un-averaged) training weights.
fn train_guess(tr: &Trainer, feats: &[String], n_tags: usize) -> usize {
    let mut sums = vec![0.0; n_tags];
    for f in feats {
        if let Some(tags) = tr.cells.get(f) {
            for (&tag, c) in tags {
                sums[tag] += c.w;
            }
        }
    }
    sums.iter()
        .enumerate()
        .fold(0, |best, (t, &s)| if s > sums[best] { t } else { best })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{parse_annotated, tokenize};

    fn cities() -> Vec<Utterance> {
        let mut d = Vec::new();
        // A tiny corpus makes the perceptron sensitive to shuffle order, so use a
        // dozen cities and assert only on clear-cut cases.
        let names = [
            "rome", "paris", "berlin", "madrid", "oslo", "lima", "cairo", "delhi", "quito",
            "perth", "dublin", "athens",
        ];
        for city in names {
            d.push(parse_annotated(
                &format!("book a table in [city : {city}] for [party : two]"),
                "b",
            ));
            d.push(parse_annotated(
                &format!("what is the weather in [city : {city}]"),
                "w",
            ));
        }
        d
    }

    #[test]
    fn affix_helpers_are_char_safe() {
        assert_eq!(prefix("école", 3), "éco");
        assert_eq!(suffix("école", 3), "ole");
        assert_eq!(prefix("ab", 5), "ab");
        assert_eq!(suffix("ab", 5), "ab");
        assert_eq!(prefix("", 1), "");
    }

    #[test]
    fn boundaries_read_as_markers() {
        let t = tokenize("a b");
        assert_eq!(at(&t, 0, -1), "<s>");
        assert_eq!(at(&t, 1, -2), "<s>");
        assert_eq!(at(&t, 1, 1), "</s>");
        assert_eq!(at(&t, 0, 1), "b");
    }

    #[test]
    fn shapes() {
        assert_eq!(
            (shape("abc"), shape("42"), shape("5pm")),
            ("alpha", "digit", "mixed")
        );
    }

    #[test]
    fn learns_slots_and_generalises_by_context() {
        let tagger = Tagger::train(&cities(), &TaggerOptions::default()).unwrap();
        let toks = tokenize("what is the weather in vienna");
        assert_eq!(tagger.tag(&toks), [Span::new(5, 6, "city")]);
    }

    #[test]
    fn gazetteer_finds_values_unseen_in_context() {
        let entries = std::collections::HashMap::from([(
            "city".to_owned(),
            vec!["Lisbon".to_owned(), "Vienna".to_owned()],
        )]);
        let opts = TaggerOptions {
            entries,
            ..TaggerOptions::default()
        };
        let tagger = Tagger::train(&cities(), &opts).unwrap();
        // a frame never seen in training, with a known city never seen as a slot value
        let toks = tokenize("reserve for two people at vienna");
        let spans = tagger.tag(&toks);
        assert!(spans.contains(&Span::new(5, 6, "city")), "{spans:?}");
    }

    #[test]
    fn training_is_deterministic() {
        let opts = TaggerOptions {
            char_features: true,
            typo_augment: 0.3,
            ..TaggerOptions::default()
        };
        let (a, b) = (
            Tagger::train(&cities(), &opts).unwrap(),
            Tagger::train(&cities(), &opts).unwrap(),
        );
        let q = tokenize("book a table in lisbon for four");
        assert_eq!(a.tag(&q), b.tag(&q));
    }

    #[test]
    fn edge_inputs() {
        assert!(Tagger::train(&[], &TaggerOptions::default()).is_none());
        let t = Tagger::train(
            &cities(),
            &TaggerOptions {
                char_features: true,
                ..TaggerOptions::default()
            },
        )
        .unwrap();
        assert_eq!(t.tag::<&str>(&[]), []);
        let _ = t.tag(&["école", "ñandú", "日本語"]); // must not panic on non-ASCII
    }

    #[test]
    fn utterances_without_slots_teach_nothing_to_extract() {
        let data = vec![parse_annotated("hello there friend", "greet"); 4];
        let t = Tagger::train(&data, &TaggerOptions::default()).unwrap();
        assert_eq!(t.tag(&tokenize("hello there friend")), []);
    }
}
