//! Train on 24 sentences and parse a few new ones: `cargo run --example quickstart`.

use rslu::{Model, Options, parse_annotated};

fn main() -> Result<(), rslu::Error> {
    let mut data = Vec::new();
    for city in ["rome", "paris", "berlin", "madrid", "oslo", "lima"] {
        data.push(parse_annotated(
            &format!("book a table in [city : {city}] for [party : two]"),
            "book_table",
        ));
        data.push(parse_annotated(
            &format!("reserve a table in [city : {city}] for [party : four] people"),
            "book_table",
        ));
        data.push(parse_annotated(
            &format!("what is the weather in [city : {city}]"),
            "get_weather",
        ));
        data.push(parse_annotated(
            &format!("will it rain in [city : {city}] tomorrow"),
            "get_weather",
        ));
    }
    let model = Model::train(&data, &Options::default())?;
    for text in [
        "Book a table in Vienna for two",
        "is it going to rain in Oslo",
        "reserve a table for four people in Lisbon",
    ] {
        let p = model.parse(text);
        println!("{text:?}");
        if let Some(i) = &p.intent {
            println!("  intent: {} ({:.0}%)", i.label, 100.0 * i.confidence);
        }
        for s in &p.slots {
            println!("  slot:   {} = {:?}", s.label, s.text);
        }
    }
    Ok(())
}
