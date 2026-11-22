//! Intent classification: multinomial logistic regression over word n-grams and
//! character n-grams, trained with Adagrad. Training is deterministic.

use std::collections::{HashMap, HashSet};

use crate::nlu::{Utterance, add_typos};
use crate::rng::SplitMix64;

/// One labelled training sentence, borrowed.
#[derive(Debug, Clone, Copy)]
pub struct Example<'a> {
    /// Word tokens.
    pub tokens: &'a [String],
    /// Intent label.
    pub label: &'a str,
}

/// Tunes logistic-regression training. [`LogRegOptions::default`] is the
/// recommended setup; [`LogRegOptions::binary`] is the plain baseline.
#[derive(Debug, Clone, PartialEq)]
pub struct LogRegOptions {
    /// Training passes over the data (default 20).
    pub epochs: usize,
    /// Adagrad base rate (default 0.25).
    pub learning_rate: f64,
    /// Weight decay on active features (default 1e-4).
    pub l2: f64,
    /// Longest word n-gram (default 2).
    pub max_n: usize,
    /// If non-zero, also use character n-grams of this length (default 3).
    pub char_n: usize,
    /// When non-zero, also train on a copy of the data in which this fraction
    /// of words carries a typo.
    pub typo_augment: f64,
    /// Weight features by inverse document frequency and normalise each
    /// sentence's vector, instead of binary features. Usually helps on
    /// many-class problems.
    pub tfidf: bool,
    /// Exponent alpha of class-balanced loss weighting: an example of class
    /// `c` is weighted `(N / (K * n_c))^alpha`. `0` is off, `1` fully
    /// balanced. Counters a dominant class swallowing rare ones.
    pub class_balance: f64,
    /// Each TF-IDF row is divided by `||x||^norm_power`. `0` selects the
    /// default of `1` (full L2); a negative value disables normalisation.
    /// Ignored unless `tfidf` is set.
    pub norm_power: f64,
}

impl Default for LogRegOptions {
    /// TF-IDF weighting, word uni/bigrams plus character trigrams, learning
    /// rate 0.25 and square-root class balancing.
    fn default() -> Self {
        Self {
            tfidf: true,
            learning_rate: 0.25,
            char_n: 3,
            class_balance: 0.5,
            ..Self::binary()
        }
    }
}

impl LogRegOptions {
    /// The plain baseline: binary word n-gram features, no balancing.
    pub fn binary() -> Self {
        Self {
            epochs: 20,
            learning_rate: 0.5,
            l2: 1e-4,
            max_n: 2,
            char_n: 0,
            typo_augment: 0.0,
            tfidf: false,
            class_balance: 0.0,
            norm_power: 0.0,
        }
    }

    /// [`LogRegOptions::default`] plus typo-augmented training data, for
    /// noisy, human-typed input.
    pub fn robust() -> Self {
        Self {
            typo_augment: 0.3,
            ..Self::default()
        }
    }
}

/// A sparse feature vector: columns and their values.
struct Row {
    ids: Vec<usize>,
    vals: Vec<f64>,
}

/// A trained multinomial logistic-regression classifier.
#[derive(Debug, Clone)]
pub struct LogReg {
    labels: Vec<String>,
    opts: LogRegOptions,
    feat: HashMap<String, usize>,
    idf: Option<Vec<f64>>,
    w: Vec<Vec<f64>>,
    bias: Vec<f64>,
}

impl LogReg {
    /// Fits a classifier on `data`, using the utterances that have an intent.
    /// Returns `None` when none do.
    pub fn train(data: &[Utterance], opts: &LogRegOptions) -> Option<Self> {
        let augmented = (opts.typo_augment > 0.0).then(|| add_typos(data, opts.typo_augment, 3));
        let examples: Vec<Example<'_>> = data
            .iter()
            .chain(augmented.iter().flatten())
            .filter(|u| !u.intent.is_empty())
            .map(|u| Example {
                tokens: &u.tokens,
                label: &u.intent,
            })
            .collect();
        Self::fit(&examples, opts)
    }

    /// Fits a classifier on explicit examples (no typo augmentation).
    /// Returns `None` for empty input.
    pub fn fit(examples: &[Example<'_>], opts: &LogRegOptions) -> Option<Self> {
        if examples.is_empty() {
            return None;
        }
        let opts = opts.clone();
        let mut labels: Vec<String> = examples.iter().map(|e| e.label.to_owned()).collect();
        labels.sort_unstable();
        labels.dedup();
        let label_idx: HashMap<&str, usize> = labels
            .iter()
            .enumerate()
            .map(|(i, l)| (l.as_str(), i))
            .collect();

        // Vocabulary, in first-seen order, and each example's distinct columns.
        let mut feat: HashMap<String, usize> = HashMap::new();
        let mut columns: Vec<Vec<usize>> = Vec::with_capacity(examples.len());
        for e in examples {
            let mut seen = HashSet::new();
            let mut ids = Vec::new();
            for f in ngram_features(e.tokens, opts.max_n, opts.char_n) {
                let next = feat.len();
                let id = *feat.entry(f).or_insert(next);
                if seen.insert(id) {
                    ids.push(id);
                }
            }
            columns.push(ids);
        }

        let nf = feat.len();
        let idf = opts.tfidf.then(|| idf_weights(&columns, nf));
        let rows: Vec<Row> = columns
            .into_iter()
            .map(|ids| make_row(ids, idf.as_deref(), opts.norm_power))
            .collect();
        let gold: Vec<usize> = examples.iter().map(|e| label_idx[e.label]).collect();
        let class_weight = class_weights(&gold, labels.len(), opts.class_balance);

        let mut m = Self {
            w: vec![vec![0.0; nf]; labels.len()],
            bias: vec![0.0; labels.len()],
            labels,
            opts,
            feat,
            idf,
        };
        m.descend(&rows, &gold, &class_weight);
        Some(m)
    }

    /// Adagrad over shuffled epochs.
    fn descend(&mut self, rows: &[Row], gold: &[usize], class_weight: &[f64]) {
        const EPS: f64 = 1e-8;
        let (lr, l2) = (self.opts.learning_rate, self.opts.l2);
        let mut acc = vec![vec![0.0; self.feat.len()]; self.labels.len()];
        let mut bias_acc = vec![0.0; self.labels.len()];
        let mut rng = SplitMix64::new(1);
        let mut order = rng.permutation(rows.len());
        for _ in 0..self.opts.epochs {
            rng.shuffle(&mut order);
            for &i in &order {
                let row = &rows[i];
                let p = self.probs(row);
                for (l, &pl) in p.iter().enumerate() {
                    let mut g = if l == gold[i] { pl - 1.0 } else { pl };
                    g *= class_weight[gold[i]];
                    bias_acc[l] += g * g;
                    self.bias[l] -= lr * g / (bias_acc[l].sqrt() + EPS);
                    for (&id, &v) in row.ids.iter().zip(&row.vals) {
                        let gw = g * v + l2 * self.w[l][id];
                        acc[l][id] += gw * gw;
                        self.w[l][id] -= lr * gw / (acc[l][id].sqrt() + EPS);
                    }
                }
            }
        }
    }

    /// Maps tokens to their known feature columns (unknown ones are dropped).
    fn row(&self, tokens: &[String]) -> Row {
        let mut seen = HashSet::new();
        let ids = ngram_features(tokens, self.opts.max_n, self.opts.char_n)
            .iter()
            .filter_map(|f| self.feat.get(f).copied())
            .filter(|id| seen.insert(*id))
            .collect();
        make_row(ids, self.idf.as_deref(), self.opts.norm_power)
    }

    fn probs(&self, row: &Row) -> Vec<f64> {
        let mut z: Vec<f64> = self
            .w
            .iter()
            .zip(&self.bias)
            .map(|(w, b)| {
                b + row
                    .ids
                    .iter()
                    .zip(&row.vals)
                    .map(|(&id, &v)| w[id] * v)
                    .sum::<f64>()
            })
            .collect();
        let max = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut sum = 0.0;
        for v in &mut z {
            *v = (*v - max).exp();
            sum += *v;
        }
        for v in &mut z {
            *v /= sum;
        }
        z
    }

    /// Returns the most likely label and its probability in `(0, 1]`.
    pub fn predict(&self, tokens: &[String]) -> (&str, f64) {
        let p = self.probs(&self.row(tokens));
        let best = p
            .iter()
            .enumerate()
            .fold(0, |best, (i, &v)| if v > p[best] { i } else { best });
        (&self.labels[best], p[best])
    }

    /// The labels this classifier can emit, sorted.
    pub fn labels(&self) -> &[String] {
        &self.labels
    }
}

/// Binary or TF-IDF values for `ids`, with optional length normalisation.
fn make_row(ids: Vec<usize>, idf: Option<&[f64]>, norm_power: f64) -> Row {
    let mut vals: Vec<f64> = ids.iter().map(|&id| idf.map_or(1.0, |w| w[id])).collect();
    if idf.is_some() && norm_power >= 0.0 {
        let norm = vals.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm > 0.0 {
            let pow = if norm_power == 0.0 { 1.0 } else { norm_power };
            let div = norm.powf(pow);
            for v in &mut vals {
                *v /= div;
            }
        }
    }
    Row { ids, vals }
}

/// Smoothed inverse document frequency per feature column.
fn idf_weights(columns: &[Vec<usize>], nf: usize) -> Vec<f64> {
    let mut df = vec![0.0; nf];
    for ids in columns {
        for &id in ids {
            df[id] += 1.0;
        }
    }
    let n = columns.len() as f64;
    df.into_iter()
        .map(|d| ((1.0 + n) / (1.0 + d)).ln() + 1.0)
        .collect()
}

fn class_weights(gold: &[usize], classes: usize, alpha: f64) -> Vec<f64> {
    if alpha <= 0.0 {
        return vec![1.0; classes];
    }
    let mut counts = vec![0.0; classes];
    for &g in gold {
        counts[g] += 1.0;
    }
    counts
        .into_iter()
        .map(|c| (gold.len() as f64 / (classes as f64 * c)).powf(alpha))
        .collect()
}

/// Word n-grams up to `max_n` (with sentence boundary markers) and, when
/// `char_n > 0`, character n-grams of each word.
fn ngram_features(tokens: &[String], max_n: usize, char_n: usize) -> Vec<String> {
    let mut padded: Vec<&str> = Vec::with_capacity(tokens.len() + 2);
    padded.push("<s>");
    padded.extend(tokens.iter().map(String::as_str));
    padded.push("</s>");
    let mut out = Vec::new();
    for n in 1..=max_n {
        for i in 0..(padded.len() + 1).saturating_sub(n) {
            if n == 1 && (i == 0 || i == padded.len() - 1) {
                continue; // boundary markers only appear inside longer n-grams
            }
            out.push(padded[i..i + n].join(" "));
        }
    }
    if char_n > 0 {
        for t in tokens {
            let w: Vec<char> = std::iter::once('^')
                .chain(t.chars())
                .chain(std::iter::once('$'))
                .collect();
            for win in w.windows(char_n) {
                out.push(format!("c:{}", win.iter().collect::<String>()));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nlu::parse_annotated;

    fn toks(s: &str) -> Vec<String> {
        crate::tokenize(s)
    }

    fn weather_vs_book() -> Vec<Utterance> {
        let mut d = Vec::new();
        for city in ["rome", "paris", "berlin", "oslo"] {
            d.push(parse_annotated(
                &format!("will it rain in {city} today"),
                "weather",
            ));
            d.push(parse_annotated(
                &format!("what is the weather in {city}"),
                "weather",
            ));
            d.push(parse_annotated(&format!("book a table in {city}"), "book"));
            d.push(parse_annotated(
                &format!("reserve a table for two in {city}"),
                "book",
            ));
        }
        d
    }

    #[test]
    fn ngrams_include_boundaries_and_chars() {
        let f = ngram_features(&toks("hi you"), 2, 0);
        assert_eq!(f, ["hi", "you", "<s> hi", "hi you", "you </s>"]);
        let c = ngram_features(&toks("ab"), 1, 3);
        assert_eq!(c, ["ab", "c:^ab", "c:ab$"]);
    }

    #[test]
    fn learns_two_intents_and_generalises_to_new_words() {
        let m = LogReg::train(&weather_vs_book(), &LogRegOptions::default()).unwrap();
        assert_eq!(m.predict(&toks("will it rain in vienna")).0, "weather");
        assert_eq!(m.predict(&toks("reserve a table in vienna")).0, "book");
        let (_, conf) = m.predict(&toks("book a table"));
        assert!(conf > 0.5 && conf <= 1.0);
    }

    #[test]
    fn robust_profile_survives_a_typo() {
        let m = LogReg::train(&weather_vs_book(), &LogRegOptions::robust()).unwrap();
        assert_eq!(m.predict(&toks("what is the wether in paris")).0, "weather");
    }

    #[test]
    fn binary_baseline_trains_too() {
        let m = LogReg::train(&weather_vs_book(), &LogRegOptions::binary()).unwrap();
        assert_eq!(m.labels(), ["book", "weather"]);
        assert_eq!(m.predict(&toks("book a table in rome")).0, "book");
    }

    #[test]
    fn training_is_deterministic() {
        let d = weather_vs_book();
        let a = LogReg::train(&d, &LogRegOptions::default()).unwrap();
        let b = LogReg::train(&d, &LogRegOptions::default()).unwrap();
        let q = toks("table for two in madrid");
        assert_eq!(a.predict(&q), b.predict(&q));
    }

    #[test]
    fn no_intents_means_no_classifier() {
        let d = vec![parse_annotated("hello [x : there]", "")];
        assert!(LogReg::train(&d, &LogRegOptions::default()).is_none());
        assert!(LogReg::train(&[], &LogRegOptions::default()).is_none());
    }

    #[test]
    fn unknown_input_still_yields_a_label() {
        let m = LogReg::train(&weather_vs_book(), &LogRegOptions::default()).unwrap();
        let (label, conf) = m.predict(&toks("zzzz qqqq"));
        assert!(m.labels().iter().any(|l| l == label));
        assert!(conf > 0.0);
    }
}
