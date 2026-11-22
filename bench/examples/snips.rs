//! SNIPS: train on the real SNIPS benchmark (run `cargo run --release -p
//! rslu-bench -- fetch --only SNIPS` first) and parse voice-assistant style
//! requests: `cargo run --release -p rslu-bench --example snips`.

use std::fs::File;
use std::path::Path;

use anyhow::{Context, Result};
use rslu::{Model, Options};
use rslu_bench::dataset::parse_snips;

const INTENTS: [&str; 7] = [
    "AddToPlaylist",
    "BookRestaurant",
    "GetWeather",
    "PlayMusic",
    "RateBook",
    "SearchCreativeWork",
    "SearchScreeningEvent",
];

fn load(dir: &Path) -> Result<Vec<rslu::Utterance>> {
    let mut all = Vec::new();
    for intent in INTENTS {
        let path = dir.join(format!("snips/train_{intent}_full.json"));
        let file = File::open(&path).with_context(|| {
            format!(
                "{} (run `rslu-bench fetch --only SNIPS` first)",
                path.display()
            )
        })?;
        all.extend(parse_snips(file)?);
    }
    Ok(all)
}

fn main() -> Result<()> {
    let dir = std::env::args().nth(1).unwrap_or_else(|| "data".to_owned());
    let data = load(Path::new(&dir))?;
    let model = Model::train(&data, &Options::default())?;
    println!("trained on {} SNIPS utterances\n", data.len());
    for text in [
        "Play some Beatles on Spotify",
        "What's the weather like in Paris tomorrow",
        "Book a table for 4 at a pizza place in Boston",
        "Add Bohemian Rhapsody to my road trip playlist",
        "Rate this book 5 out of 6 stars",
        "When is Black Panther playing at the Rialto cinema",
    ] {
        let parse = model.parse(text);
        println!("{text:?}");
        if let Some(i) = &parse.intent {
            println!("  intent: {} ({:.0}%)", i.label, 100.0 * i.confidence);
        }
        for s in &parse.slots {
            println!("  slot:   {:<24} {:?}", s.label, s.text);
        }
    }
    Ok(())
}
