//! Entities: teach the model a domain vocabulary it never saw in training, see
//! it survive typos, and link each extracted name to its canonical entry. This
//! is how a query engine can plug in its own driver/team/circuit names:
//! `cargo run --example entities`.

use std::collections::HashMap;

use rslu::{Model, Options, parse_annotated};

fn main() -> Result<(), rslu::Error> {
    let mut data = Vec::new();
    for d in ["senna", "prost", "alonso", "vettel", "raikkonen", "button"] {
        for (text, intent) in [
            (format!("how many wins does [driver : {d}] have"), "wins"),
            (format!("how many races did [driver : {d}] win"), "wins"),
            (format!("how many poles did [driver : {d}] take"), "poles"),
            (format!("poles for [driver : {d}]"), "poles"),
        ] {
            data.push(parse_annotated(&text, intent));
        }
    }

    // Names that appear nowhere in the training data:
    let entries = HashMap::from([(
        "driver".to_owned(),
        [
            "Lewis Hamilton",
            "Max Verstappen",
            "Charles Leclerc",
            "Fernando Alonso",
        ]
        .map(String::from)
        .to_vec(),
    )]);
    let model = Model::train(
        &data,
        &Options {
            robust: true,
            entries,
            ..Options::default()
        },
    )?;

    for text in [
        "how many wins does verstappen have",
        "how many poles did hamilton take",
        "how many poles did hamliton take", // typo in an entity
        "how many wnis does leclerc have",  // typo in a non-entity word
    ] {
        let parse = model.parse(text);
        let intent = parse.intent.as_ref().map_or("-", |i| i.label.as_str());
        print!("{text:<40} -> {intent:<6}");
        for s in &parse.slots {
            print!("  {}={:?}", s.label, s.text);
            if let Some(entry) = &s.entry {
                print!(" -> {entry:?}"); // linked to the canonical entry
            }
        }
        println!();
    }
    Ok(())
}
