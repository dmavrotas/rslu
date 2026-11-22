//! Evaluation profiles and scoring.

use rslu::{LogReg, LogRegOptions, Span, Tagger, TaggerOptions, Utterance};
use serde::{Deserialize, Serialize};

use crate::metrics;
use crate::sets::Benchmark;

/// Intent and slot settings evaluated together.
pub struct Profile {
    pub name: &'static str,
    pub intent: LogRegOptions,
    pub slot: TaggerOptions,
}

pub fn profiles() -> Vec<Profile> {
    let slot = |char_features, typo_augment, gazetteer| TaggerOptions {
        char_features,
        typo_augment,
        gazetteer,
        ..TaggerOptions::default()
    };
    vec![
        Profile {
            name: "default",
            intent: LogRegOptions::default(),
            slot: slot(false, 0.0, false),
        },
        Profile {
            name: "robust",
            intent: LogRegOptions::robust(),
            slot: slot(true, 0.3, false),
        },
        Profile {
            name: "robust+gaz",
            intent: LogRegOptions::robust(),
            slot: slot(true, 0.3, true),
        },
    ]
}

/// Metrics of one benchmark/profile, in percent. `None` means not applicable
/// (for example no slots in CLINC150).
#[derive(Debug, Default, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Scores {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub intent: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub slot_f1: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frame: Option<f64>,
}

/// Trains profile `p` on `train` and scores it on `test`.
pub fn evaluate(b: &Benchmark, p: &Profile, train: &[Utterance], test: &[Utterance]) -> Scores {
    let clf = b.intents.then(|| LogReg::train(train, &p.intent)).flatten();
    let tagger = b.slots.then(|| Tagger::train(train, &p.slot)).flatten();
    let want_intent: Vec<String> = test.iter().map(|u| u.intent.clone()).collect();
    let want_slots: Vec<Vec<Span>> = test.iter().map(|u| u.slots.clone()).collect();
    let got_intent: Option<Vec<String>> = clf.as_ref().map(|c| {
        test.iter()
            .map(|u| c.predict(&u.tokens).0.to_owned())
            .collect()
    });
    let got_slots: Option<Vec<Vec<Span>>> = tagger
        .as_ref()
        .map(|t| test.iter().map(|u| t.tag(&u.tokens)).collect());
    Scores {
        intent: got_intent
            .as_ref()
            .map(|g| 100.0 * metrics::accuracy(g, &want_intent)),
        slot_f1: got_slots
            .as_ref()
            .map(|g| 100.0 * metrics::slot_f1(g, &want_slots)),
        frame: got_intent
            .zip(got_slots)
            .map(|(gi, gs)| 100.0 * metrics::frame_accuracy(&gi, &want_intent, &gs, &want_slots)),
    }
}
