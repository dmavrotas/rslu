//! A small, deterministic, offline natural-language-understanding toolkit.
//!
//! It classifies the **intent** of a sentence and extracts its **slots**
//! (entities), with no network calls, no model files and no dependencies.
//! It is the Rust counterpart of [gonlu](https://github.com/dmavrotas/gonlu).
//!
//! Train a [`Model`] from annotated examples and parse new text:
//!
//! ```
//! use rslu::{Model, Options, parse_annotated};
//!
//! let mut data = Vec::new();
//! for city in ["rome", "paris", "berlin", "madrid", "oslo", "lima"] {
//!     data.push(parse_annotated(&format!("book a table in [city : {city}] for [party : two]"), "book_table"));
//!     data.push(parse_annotated(&format!("what is the weather in [city : {city}]"), "get_weather"));
//! }
//!
//! let model = Model::train(&data, &Options::default())?;
//! let parse = model.parse("book a table in vienna for two");
//! assert_eq!(parse.intent.as_ref().map(|i| i.label.as_str()), Some("book_table"));
//! assert_eq!(parse.slots[0].label, "city");
//! # Ok::<(), rslu::Error>(())
//! ```

mod codec;
mod gazetteer;
pub mod intent;
mod link;
pub mod nlu;
mod rng;
pub mod slot;

use std::collections::HashMap;
use std::error::Error as StdError;
use std::fmt;

pub use codec::DecodeError;
pub use intent::{Classifier, LogReg, LogRegOptions, NaiveBayes};
pub use link::Link;
pub use nlu::{
    Span, Utterance, add_typos, bio_from_spans, parse_annotated, spans_from_bio, tokenize,
};
pub use slot::{Tagger, TaggerOptions};

/// What can go wrong when training a [`Model`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Error {
    /// [`Model::train`] was given no utterances.
    NoData,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoData => f.write_str("rslu: no training data"),
        }
    }
}

impl StdError for Error {}

/// Configures [`Model::train`]. The default is the benchmark-default model.
#[derive(Debug, Clone, Default)]
pub struct Options {
    /// Train for noisy, human-typed input by adding typo-augmented data. Costs
    /// little on clean text and recovers a lot on text with misspellings.
    pub robust: bool,
    /// Known entity values per slot type, for example
    /// `{"driver": ["Lewis Hamilton", "Max Verstappen"]}`. They are matched
    /// typo-tolerantly, help label values never seen in training, and link each
    /// extracted slot to its canonical entry ([`Slot::entry`]). List more
    /// important entries first: they win ties.
    pub entries: HashMap<String, Vec<String>>,
    /// Overrides the slot tagger's training epochs (`0` keeps the default, 10).
    pub epochs: usize,
}

/// One extracted entity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Slot {
    /// Slot type, for example `"city"`.
    pub label: String,
    /// The matched tokens joined by spaces (lower-cased).
    pub text: String,
    /// Token offset of the first matched token.
    pub start: usize,
    /// Token offset one past the last matched token.
    pub end: usize,
    /// The closest value from [`Options::entries`] for this slot's type, in its
    /// original spelling (for example `"Max Verstappen"` for `"verstapen"`).
    /// `None` when no entry is close enough or the type has none.
    pub entry: Option<String>,
}

/// A predicted intent and its probability.
#[derive(Debug, Clone, PartialEq)]
pub struct Intent {
    /// The intent label.
    pub label: String,
    /// Probability of the label, in `(0, 1]`.
    pub confidence: f64,
}

/// The parse of one sentence.
#[derive(Debug, Clone, PartialEq)]
pub struct Parse {
    /// `None` when the model was trained without intents.
    pub intent: Option<Intent>,
    /// The tokenised input.
    pub tokens: Vec<String>,
    /// Extracted slots, in order.
    pub slots: Vec<Slot>,
}

/// A trained intent classifier and slot tagger. Immutable once trained, so it
/// can be shared across threads.
#[derive(Debug, Clone)]
pub struct Model {
    classifier: Option<LogReg>,
    tagger: Tagger,
    linker: link::Linker,
}

impl Model {
    /// Fits a model. Utterances with an empty intent are ignored by the intent
    /// classifier, and utterances without slots teach the tagger that nothing
    /// should be extracted.
    ///
    /// # Errors
    /// [`Error::NoData`] when `data` is empty.
    pub fn train(data: &[Utterance], opts: &Options) -> Result<Self, Error> {
        if data.is_empty() {
            return Err(Error::NoData);
        }
        let (intent_opts, mut tagger_opts) = if opts.robust {
            (
                LogRegOptions::robust(),
                TaggerOptions {
                    char_features: true,
                    typo_augment: 0.3,
                    ..TaggerOptions::default()
                },
            )
        } else {
            (LogRegOptions::default(), TaggerOptions::default())
        };
        tagger_opts.entries.clone_from(&opts.entries);
        if opts.epochs > 0 {
            tagger_opts.epochs = opts.epochs;
        }
        Ok(Self {
            classifier: LogReg::train(data, &intent_opts),
            tagger: Tagger::train(data, &tagger_opts).ok_or(Error::NoData)?,
            linker: link::Linker::new(&opts.entries),
        })
    }

    /// Classifies `text` and extracts its slots.
    pub fn parse(&self, text: &str) -> Parse {
        let tokens = tokenize(text);
        if tokens.is_empty() {
            return Parse {
                intent: None,
                tokens,
                slots: Vec::new(),
            };
        }
        let intent = self.classifier.as_ref().map(|c| {
            let (label, confidence) = c.predict(&tokens);
            Intent {
                label: label.to_owned(),
                confidence,
            }
        });
        let slots = self
            .tagger
            .tag(&tokens)
            .into_iter()
            .map(|s| {
                let text = tokens[s.start..s.end].join(" ");
                let entry = self
                    .linker
                    .best(&s.label, &text)
                    .map(|l| l.entry.to_owned());
                Slot {
                    label: s.label,
                    text,
                    start: s.start,
                    end: s.end,
                    entry,
                }
            })
            .collect();
        Parse {
            intent,
            tokens,
            slots,
        }
    }

    /// Resolves `text` to the closest value in [`Options::entries`] for
    /// `slot_type`: the same lookup [`Model::parse`] uses to fill
    /// [`Slot::entry`], with the match cost (see [`Link`]).
    ///
    /// `parse` links a slot only within the type the tagger chose. `link` lets
    /// a caller compare several types and let the name itself decide, which is
    /// more reliable than the tagger's guess when a sentence frame fits more
    /// than one type.
    pub fn link(&self, slot_type: &str, text: &str) -> Option<Link<'_>> {
        self.linker
            .best(slot_type, text.trim().to_lowercase().as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_data_is_an_error() {
        assert_eq!(
            Model::train(&[], &Options::default()).unwrap_err(),
            Error::NoData
        );
        assert_eq!(Error::NoData.to_string(), "rslu: no training data");
    }

    #[test]
    fn empty_text_parses_to_nothing() {
        let data = [parse_annotated("hello [x : there]", "greet")];
        let p = Model::train(&data, &Options::default())
            .unwrap()
            .parse("  ?! ");
        assert!(p.intent.is_none() && p.tokens.is_empty() && p.slots.is_empty());
    }

    #[test]
    fn model_is_send_and_sync() {
        fn assert_traits<T: Send + Sync>() {}
        assert_traits::<Model>();
    }
}
