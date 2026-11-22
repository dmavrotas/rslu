//! End-to-end behaviour of the public API.

use std::collections::HashMap;

use rslu::{Error, Model, Options, parse_annotated};

fn training_data() -> Vec<rslu::Utterance> {
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
    data
}

#[test]
fn parses_intent_and_slots() {
    let model = Model::train(&training_data(), &Options::default()).unwrap();
    let p = model.parse("Book a table in Vienna for two");
    let intent = p.intent.unwrap();
    assert_eq!(intent.label, "book_table");
    assert!(intent.confidence > 0.9);
    let slots: Vec<_> = p
        .slots
        .iter()
        .map(|s| (s.label.as_str(), s.text.as_str()))
        .collect();
    assert_eq!(slots, [("city", "vienna"), ("party", "two")]);
}

fn drivers() -> Options {
    Options {
        robust: true,
        entries: HashMap::from([(
            "driver".to_owned(),
            ["Lewis Hamilton", "Max Verstappen", "Charles Leclerc"]
                .map(String::from)
                .into(),
        )]),
        ..Options::default()
    }
}

fn driver_data() -> Vec<rslu::Utterance> {
    let mut data = Vec::new();
    for d in [
        "senna",
        "prost",
        "alonso",
        "vettel",
        "schumacher",
        "raikkonen",
    ] {
        data.push(parse_annotated(
            &format!("how many wins does [driver : {d}] have"),
            "wins",
        ));
        data.push(parse_annotated(
            &format!("how many poles did [driver : {d}] take"),
            "poles",
        ));
    }
    data
}

#[test]
fn links_slots_to_entries_even_with_typos() {
    let model = Model::train(&driver_data(), &drivers()).unwrap();
    let p = model.parse("how many poles did hamliton take");
    assert_eq!(p.intent.unwrap().label, "poles");
    let slot = &p.slots[0];
    assert_eq!(
        (slot.label.as_str(), slot.text.as_str()),
        ("driver", "hamliton")
    );
    assert_eq!(slot.entry.as_deref(), Some("Lewis Hamilton"));
}

#[test]
fn link_exposes_the_cost_across_types() {
    let model = Model::train(&driver_data(), &drivers()).unwrap();
    let hit = model.link("driver", "  Leclerc ").unwrap();
    assert_eq!((hit.entry, hit.cost), ("Charles Leclerc", 1));
    assert!(model.link("team", "leclerc").is_none());
}

#[test]
fn training_twice_gives_identical_parses() {
    let (a, b) = (
        Model::train(&training_data(), &Options::default()).unwrap(),
        Model::train(&training_data(), &Options::default()).unwrap(),
    );
    let q = "will it rain in lisbon";
    assert_eq!(a.parse(q), b.parse(q));
}

#[test]
fn no_data_is_an_error() {
    assert_eq!(
        Model::train(&[], &Options::default()).unwrap_err(),
        Error::NoData
    );
}
