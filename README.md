# rslu

A small, deterministic, **offline, pure-Rust** natural-language-understanding toolkit.
It classifies the **intent** of a sentence and extracts its **slots** (entities), with no
network calls, no model files, no unsafe code and **no dependencies**.

It is the Rust counterpart of [gonlu](https://github.com/dmavrotas/gonlu) and uses the same
algorithms: a logistic-regression intent classifier (TF-IDF word n-grams plus character
n-grams), an averaged-perceptron slot tagger with an optional typo-tolerant gazetteer, and
entity linking against your own vocabulary.

> Status: early (v0). The API may change. Accuracy is that of a linear model: strong on small,
> domain-specific vocabularies, clearly behind fine-tuned transformers on large open ones.

I found no other Rust crate that trains intent and slots together, offline.
[snips-nlu-rs](https://github.com/talkative-ai/snips-nlu-rs) is inference-only (you train in
Python), and the other crates cover one piece (POS tagging, TF-IDF, intent only).

## Install

```toml
[dependencies]
rslu = { git = "https://github.com/dmavrotas/rslu" }
```

Requires Rust 1.88 or newer (the code uses let-chains).

## Example: train on a handful of sentences

Write training data with inline `[label : value]` markup, train, parse
([`examples/quickstart.rs`](examples/quickstart.rs)):

```rust
use rslu::{Model, Options, parse_annotated};

let mut data = Vec::new();
for city in ["rome", "paris", "berlin", "madrid", "oslo", "lima"] {
    data.push(parse_annotated(&format!("book a table in [city : {city}] for [party : two]"), "book_table"));
    data.push(parse_annotated(&format!("what is the weather in [city : {city}]"), "get_weather"));
}
let model = Model::train(&data, &Options::default())?;
let parse = model.parse("Book a table in Vienna for two");
```

Real output of `cargo run --example quickstart` (24 training sentences):

```
"Book a table in Vienna for two"
  intent: book_table (100%)
  slot:   city = "vienna"
  slot:   party = "two"
"is it going to rain in Oslo"
  intent: get_weather (98%)
  slot:   city = "oslo"
"reserve a table for four people in Lisbon"
  intent: book_table (100%)
  slot:   party = "four"
  slot:   city = "lisbon"
```

With so little data results depend on training order: gonlu misses `lisbon` on the same
sentences and rslu finds it, and on a 12-sentence corpus changing only the shuffle seed
changed whether an unseen city was tagged. Use more examples or supply known values (below).

## Your own vocabulary, typos, entity linking

`Options::entries` gives the model known entity values it never saw in training. They are
matched typo-tolerantly, and every extracted slot is linked to its canonical entry in
`Slot::entry`. `Options::robust` trains for human-typed input.

```rust
let opts = Options {
    robust: true,
    entries: HashMap::from([("driver".into(), vec!["Lewis Hamilton".into(), "Max Verstappen".into()])]),
    ..Options::default()
};
let model = Model::train(&data, &opts)?;
let p = model.parse("how many poles did hamliton take");
// p.slots[0].text == "hamliton", p.slots[0].entry == Some("Lewis Hamilton")
```

Linking prefers the whole entry, then its last word (a surname beats a forename), then earlier
words, each exact before typo-tolerant. Ties go to the entry listed first. `Model::link`
exposes the lookup with a match cost so a caller can compare several slot types and let the
name decide.

## Benchmarks

`cargo run --release -p rslu-bench -- fetch` downloads the six public corpora (ATIS, SNIPS,
MASSIVE, CLINC150, BANKING77, MIT-Restaurant), and `cargo run --release -p rslu-bench -- run`
scores three profiles on the standard test splits (`--only a,b` selects, `--typo 0.3` corrupts
the test set to measure robustness).

Side by side with [gonlu](https://github.com/dmavrotas/gonlu), same machine, same data. Each
cell is intent accuracy / slot F1 / frame accuracy (`-` = not applicable):

| dataset | profile | gonlu | rslu |
|---|---|---|---|
| ATIS | default | 93.73 / 93.27 / 79.62 | 92.95 / 93.52 / 79.40 |
| ATIS | robust | 92.95 / 93.35 / 79.51 | 93.28 / 93.16 / 79.17 |
| ATIS | robust+gaz | 92.95 / 93.75 / 79.62 | 93.28 / 93.77 / 80.52 |
| SNIPS | default | 98.57 / 91.49 / 80.29 | 98.29 / 91.09 / 80.00 |
| SNIPS | robust | 98.43 / 91.17 / 79.29 | 98.43 / 90.85 / 78.71 |
| SNIPS | robust+gaz | 98.43 / 92.32 / 80.71 | 98.43 / 92.72 / 81.43 |
| MASSIVE | default | 85.44 / 76.40 / 63.95 | 85.24 / 75.71 / 63.21 |
| MASSIVE | robust | 85.31 / 75.99 / 63.55 | 85.44 / 75.82 / 63.45 |
| MASSIVE | robust+gaz | 85.31 / 76.29 / 63.58 | 85.44 / 75.60 / 63.48 |
| CLINC150 | default | 91.67 / - / - | 91.42 / - / - |
| CLINC150 | robust | 91.42 / - / - | 91.51 / - / - |
| BANKING77 | default | 91.04 / - / - | 91.04 / - / - |
| BANKING77 | robust | 90.75 / - / - | 90.62 / - / - |
| MIT-Rest | default | - / 76.28 / - | - / 75.86 / - |
| MIT-Rest | robust | - / 76.13 / - | - / 75.91 / - |
| MIT-Rest | robust+gaz | - / 77.23 / - | - / 76.95 / - |

Every cell is within 0.9 points of gonlu, and the differences go both ways. The two libraries
are not bit-identical: both shuffle training data with a fixed seed, but the generators
differ, and a perceptron is order-sensitive. The accuracy is equivalent, not equal.
Like gonlu, the linear models are strong on small domain vocabularies, in the low 90s on
CLINC150 and BANKING77, and clearly behind fine-tuned transformers on those.

Speed is not equivalent: rslu is faster. ATIS and SNIPS (all profiles) take about 30 s in
rslu (`--release`) against 4 min 1 s in gonlu; the other four corpora take about 75 s against
4 min 32 s. Over all six that is roughly 5x.

CI runs `rslu-bench check`, which scores the default profile on ATIS, SNIPS, CLINC150 and
BANKING77 and fails if any metric drops more than 0.5 points below
[`bench/baseline.json`](bench/baseline.json) (or if a baselined benchmark produces no result).
The baseline file has the same format as gonlu's.

## Example: a real benchmark corpus

Train on the 13,784-utterance [SNIPS](https://github.com/snipsco/nlu-benchmark) training set
([`bench/examples/snips.rs`](bench/examples/snips.rs); fetch it first). Real output,
including the mistakes:

```
"Play some Beatles on Spotify"
  intent: PlayMusic (99%)
  slot:   genre                    "beatles"            <- should be artist
  slot:   service                  "spotify"
"What's the weather like in Paris tomorrow"
  intent: GetWeather (99%)
  slot:   city                     "paris"
  slot:   timeRange                "tomorrow"
"Book a table for 4 at a pizza place in Boston"
  intent: BookRestaurant (100%)
  slot:   party_size_number        "4"
  slot:   served_dish              "pizza"
  slot:   city                     "boston"
"Add Bohemian Rhapsody to my road trip playlist"
  intent: AddToPlaylist (97%)
  slot:   artist                   "bohemian rhapsody"
  slot:   playlist_owner           "my"
  slot:   playlist                 "road trip"
"Rate this book 5 out of 6 stars"
  intent: RateBook (100%)
  slot:   rating_value             "5"
  slot:   best_rating              "6"
  slot:   rating_unit              "stars"
```

(The `<- should be artist` annotation is added by hand; the `Rate this book` output also lists
`object_select` and `object_type` slots, omitted here.)

## Design notes

- Deterministic: a fixed-seed PRNG (SplitMix64) drives all shuffling and typo noise.
- Inputs are tokenised by lower-casing and splitting on anything but letters, digits and
  apostrophes. Affixes and character n-grams are character-based, so non-ASCII text is safe.
- A trained `Model` is immutable and `Send + Sync`.
- `NaiveBayes` (unigram+bigram, Laplace smoothing) ships as the baseline every other classifier
  must beat; both classifiers implement the `Classifier` trait.
- `unsafe` is forbidden; clippy pedantic is enabled and clean.

## License

Apache-2.0.
