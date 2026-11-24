//! Saving and loading trained classifiers.

use std::io::{self, Cursor, Read};

use rslu::{DecodeError, LogReg, LogRegOptions, Utterance, parse_annotated, tokenize};

fn corpus() -> Vec<Utterance> {
    let mut data = Vec::new();
    for city in ["rome", "paris", "berlin", "madrid", "oslo", "lima", "quito"] {
        data.push(parse_annotated(
            &format!("book a table in {city} for two"),
            "book",
        ));
        data.push(parse_annotated(
            &format!("what is the weather in {city}"),
            "weather",
        ));
        data.push(parse_annotated(
            &format!("play some music from {city}"),
            "music",
        ));
    }
    data
}

fn profiles() -> [(&'static str, LogRegOptions); 3] {
    [
        ("default", LogRegOptions::default()),
        ("binary", LogRegOptions::binary()),
        ("robust", LogRegOptions::robust()),
    ]
}

const PROBES: [&str; 6] = [
    "book a table in vienna",
    "weather in tokyo tomorrow",
    "play jazz",
    "",
    "completely unrelated words",
    "bokk a tabel",
];

#[test]
fn a_loaded_model_predicts_exactly_like_the_trained_one() {
    for (name, opts) in profiles() {
        let model = LogReg::train(&corpus(), &opts).unwrap();
        let loaded = LogReg::from_bytes(&model.to_bytes()).unwrap();
        assert_eq!(loaded.labels(), model.labels(), "{name}");
        for probe in PROBES {
            let tokens = tokenize(probe);
            let (want_label, want_p) = model.predict(&tokens);
            let (got_label, got_p) = loaded.predict(&tokens);
            assert_eq!(got_label, want_label, "{name}: {probe:?}");
            assert_eq!(got_p.to_bits(), want_p.to_bits(), "{name}: {probe:?}");
        }
    }
}

#[test]
fn encoding_is_stable() {
    let model = LogReg::train(&corpus(), &LogRegOptions::default()).unwrap();
    let bytes = model.to_bytes();
    assert_eq!(LogReg::from_bytes(&bytes).unwrap().to_bytes(), bytes);
    // Training is deterministic, so retraining gives the same file.
    let again = LogReg::train(&corpus(), &LogRegOptions::default()).unwrap();
    assert_eq!(again.to_bytes(), bytes);
}

#[test]
fn writer_and_reader_round_trip() {
    let model = LogReg::train(&corpus(), &LogRegOptions::binary()).unwrap();
    let mut file = Vec::new();
    model.write_to(&mut file).unwrap();
    let loaded = LogReg::read_from(Cursor::new(file)).unwrap();
    assert_eq!(loaded.to_bytes(), model.to_bytes());
}

#[test]
fn a_failing_reader_is_an_io_error() {
    struct Broken;
    impl Read for Broken {
        fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
            Err(io::Error::other("disk on fire"))
        }
    }
    assert_eq!(
        LogReg::read_from(Broken).err(),
        Some(DecodeError::Io(io::ErrorKind::Other))
    );
}

#[test]
fn foreign_and_damaged_input_is_rejected() {
    let bytes = LogReg::train(&corpus(), &LogRegOptions::binary())
        .unwrap()
        .to_bytes();
    assert_eq!(
        LogReg::from_bytes(b"{\"json\": true}").err(),
        Some(DecodeError::NotAModel)
    );
    let mut trailing = bytes.clone();
    trailing.push(0);
    assert!(matches!(
        LogReg::from_bytes(&trailing),
        Err(DecodeError::Corrupt(_))
    ));
    for cut in 0..bytes.len() {
        assert!(
            LogReg::from_bytes(&bytes[..cut]).is_err(),
            "prefix of {cut} bytes"
        );
    }
}

#[test]
fn flipped_bytes_never_panic() {
    let data = &corpus()[..6];
    let bytes = LogReg::train(data, &LogRegOptions::default())
        .unwrap()
        .to_bytes();
    for i in 0..bytes.len() {
        let mut damaged = bytes.clone();
        damaged[i] ^= 0xff;
        // Either outcome is fine (a flipped weight is still a valid model);
        // what matters is that decoding and predicting do not panic.
        if let Ok(model) = LogReg::from_bytes(&damaged) {
            let _ = model.predict(&tokenize("book a table"));
        }
    }
}
