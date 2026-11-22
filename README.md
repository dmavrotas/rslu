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

`cargo run --release -p rslu-bench -- fetch` downloads the public corpora, and
`cargo run --release -p rslu-bench -- run` scores three profiles on the standard test splits.
Only ATIS and SNIPS are ported so far; CLINC150, BANKING77, MASSIVE and MIT-Restaurant are next.

Side by side with gonlu, same machine, same data:

| dataset | profile | metric | gonlu | rslu |
|---|---|---|---|---|
| ATIS | default | intent / slot F1 / frame | 93.73 / 93.27 / 79.62 | 92.95 / 93.52 / 79.40 |
| ATIS | robust | intent / slot F1 / frame | 92.95 / 93.35 / 79.51 | 93.28 / 93.16 / 79.17 |
| ATIS | robust+gaz | intent / slot F1 / frame | 92.95 / 93.75 / 79.62 | 93.28 / 93.77 / 80.52 |
| SNIPS | default | intent / slot F1 / frame | 98.57 / 91.49 / 80.29 | 98.29 / 91.09 / 80.00 |
| SNIPS | robust | intent / slot F1 / frame | 98.43 / 91.17 / 79.29 | 98.43 / 90.85 / 78.71 |
| SNIPS | robust+gaz | intent / slot F1 / frame | 98.43 / 92.32 / 80.71 | 98.43 / 92.72 / 81.43 |

Every cell is within 0.8 points of gonlu except one (ATIS robust+gaz frame, +0.9), and the
differences go both ways. The two libraries are not bit-identical: both shuffle training data
with a fixed seed, but the generators differ, and a perceptron is order-sensitive. The
accuracy numbers are therefore equivalent, not equal.

Speed is not equivalent: rslu is much faster. Scoring all three profiles on ATIS and SNIPS
takes about 30 s in rslu (`--release`) against about 4 min in gonlu on the same machine.

CI runs `rslu-bench check`, which scores the default profile and fails if any metric drops
more than 0.5 points below [`bench/baseline.json`](bench/baseline.json).

## Design notes

- Deterministic: a fixed-seed PRNG (SplitMix64) drives all shuffling and typo noise.
- Inputs are tokenised by lower-casing and splitting on anything but letters, digits and
  apostrophes. Affixes and character n-grams are character-based, so non-ASCII text is safe.
- A trained `Model` is immutable and `Send + Sync`.
- `unsafe` is forbidden; clippy pedantic is enabled and clean.

## License

Apache-2.0.
