//! Downloads public NLU benchmarks and scores rslu on them.
//!
//! ```text
//! rslu-bench fetch [--dir data]                          download the benchmarks
//! rslu-bench run   [--dir data] [--only a,b] [--typo 0.3] evaluate and print a report
//! rslu-bench check [--dir data] [--baseline f] [--tolerance 0.5] [--update]
//!                                                         fail if the default profile regressed
//! ```

mod dataset;
mod metrics;

use std::collections::BTreeMap;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use rslu::{LogReg, LogRegOptions, Span, Tagger, TaggerOptions, Utterance, add_typos};
use serde::{Deserialize, Serialize};

type Split = (Vec<Utterance>, Vec<Utterance>);

struct Benchmark {
    name: &'static str,
    intents: bool,
    slots: bool,
    files: Vec<(String, String)>, // (url, path relative to the data dir)
    load: fn(&Path) -> Result<Split>,
}

const ATIS_BASE: &str = "https://raw.githubusercontent.com/yvchen/JointSLU/master/data/";
const SNIPS_BASE: &str =
    "https://raw.githubusercontent.com/snipsco/nlu-benchmark/master/2017-06-custom-intent-engines/";
const SNIPS_INTENTS: [&str; 7] = [
    "AddToPlaylist",
    "BookRestaurant",
    "GetWeather",
    "PlayMusic",
    "RateBook",
    "SearchCreativeWork",
    "SearchScreeningEvent",
];

fn benchmarks() -> Vec<Benchmark> {
    let snips_files = SNIPS_INTENTS
        .iter()
        .flat_map(|i| {
            [format!("train_{i}_full.json"), format!("validate_{i}.json")]
                .into_iter()
                .map(move |f| (format!("{SNIPS_BASE}{i}/{f}"), format!("snips/{f}")))
        })
        .collect();
    vec![
        Benchmark {
            name: "ATIS",
            intents: true,
            slots: true,
            files: ["atis.train.w-intent.iob", "atis.test.w-intent.iob"]
                .map(|f| (format!("{ATIS_BASE}{f}"), format!("atis/{f}")))
                .into(),
            load: |dir| {
                Ok((
                    load_file(
                        &dir.join("atis/atis.train.w-intent.iob"),
                        dataset::parse_atis,
                    )?,
                    load_file(
                        &dir.join("atis/atis.test.w-intent.iob"),
                        dataset::parse_atis,
                    )?,
                ))
            },
        },
        Benchmark {
            name: "SNIPS",
            intents: true,
            slots: true,
            files: snips_files,
            load: |dir| {
                let load = |prefix: &str, suffix: &str| -> Result<Vec<Utterance>> {
                    let mut all = Vec::new();
                    for i in SNIPS_INTENTS {
                        let p = dir.join(format!("snips/{prefix}{i}{suffix}"));
                        all.extend(load_file(&p, dataset::parse_snips)?);
                    }
                    Ok(all)
                };
                Ok((load("train_", "_full.json")?, load("validate_", ".json")?))
            },
        },
    ]
}

fn load_file(path: &Path, parse: fn(fs::File) -> Result<Vec<Utterance>>) -> Result<Vec<Utterance>> {
    let f = fs::File::open(path)
        .with_context(|| format!("{} (run `rslu-bench fetch` first)", path.display()))?;
    parse(f).with_context(|| path.display().to_string())
}

struct Profile {
    name: &'static str,
    intent: LogRegOptions,
    slot: TaggerOptions,
}

fn profiles() -> Vec<Profile> {
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

/// Metrics of one benchmark/profile, in percent. Absent means not applicable.
#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize)]
struct Scores {
    #[serde(skip_serializing_if = "Option::is_none")]
    intent: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    slot_f1: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    frame: Option<f64>,
}

fn evaluate(b: &Benchmark, p: &Profile, train: &[Utterance], test: &[Utterance]) -> Scores {
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

struct Options {
    dir: PathBuf,
    only: Vec<String>,
    typo: f64,
    baseline: PathBuf,
    tolerance: f64,
    update: bool,
}

fn parse_flags(args: &[String]) -> Result<Options> {
    let mut o = Options {
        dir: "data".into(),
        only: Vec::new(),
        typo: 0.0,
        baseline: "bench/baseline.json".into(),
        tolerance: 0.5,
        update: false,
    };
    let mut it = args.iter();
    while let Some(flag) = it.next() {
        let mut value = || it.next().with_context(|| format!("{flag} needs a value"));
        match flag.trim_start_matches('-') {
            "dir" => o.dir = value()?.into(),
            "only" => o.only = value()?.split(',').map(str::to_owned).collect(),
            "typo" => o.typo = value()?.parse()?,
            "baseline" => o.baseline = value()?.into(),
            "tolerance" => o.tolerance = value()?.parse()?,
            "update" => o.update = true,
            _ => bail!("unknown flag {flag:?}"),
        }
    }
    Ok(o)
}

fn selected(only: &[String]) -> Vec<Benchmark> {
    benchmarks()
        .into_iter()
        .filter(|b| only.is_empty() || only.iter().any(|n| n.eq_ignore_ascii_case(b.name)))
        .collect()
}

fn fetch(o: &Options) -> Result<()> {
    for b in selected(&o.only) {
        for (url, rel) in &b.files {
            let dst = o.dir.join(rel);
            if dst.exists() {
                continue;
            }
            fs::create_dir_all(dst.parent().context("no parent")?)?;
            let mut body = Vec::new();
            ureq::get(url)
                .call()?
                .into_body()
                .into_reader()
                .read_to_end(&mut body)?;
            fs::write(&dst, body)?;
            println!("fetched {rel}");
        }
    }
    Ok(())
}

fn cell(v: Option<f64>) -> String {
    v.map_or_else(|| "-".to_owned(), |v| format!("{v:.2}%"))
}

fn run(o: &Options) -> Result<()> {
    println!(
        "{:<10} {:<10} {:>6} {:>6} {:>11} {:>8} {:>9} {:>7}",
        "dataset", "profile", "train", "test", "intent-acc", "slot-F1", "frame-acc", "secs"
    );
    for b in selected(&o.only) {
        let (train, mut test) = (b.load)(&o.dir)?;
        if o.typo > 0.0 {
            test = add_typos(&test, o.typo, 7);
        }
        for p in profiles() {
            if p.slot.gazetteer && !b.slots {
                continue; // gazetteer features only affect slot filling
            }
            let t = Instant::now();
            let s = evaluate(&b, &p, &train, &test);
            println!(
                "{:<10} {:<10} {:>6} {:>6} {:>11} {:>8} {:>9} {:>7.1}",
                b.name,
                p.name,
                train.len(),
                test.len(),
                cell(s.intent),
                cell(s.slot_f1),
                cell(s.frame),
                t.elapsed().as_secs_f64()
            );
        }
    }
    Ok(())
}

/// Scores the default profile and fails if any metric dropped more than
/// `tolerance` points below the baseline; `--update` rewrites the baseline.
fn check(o: &Options) -> Result<()> {
    let default = profiles().remove(0);
    let mut now = BTreeMap::new();
    for b in selected(&o.only) {
        let (train, test) = (b.load)(&o.dir)?;
        now.insert(b.name.to_owned(), evaluate(&b, &default, &train, &test));
    }
    if o.update {
        fs::write(&o.baseline, serde_json::to_string_pretty(&now)? + "\n")?;
        println!("baseline written to {}", o.baseline.display());
        return Ok(());
    }
    let base: BTreeMap<String, Scores> = serde_json::from_slice(
        &fs::read(&o.baseline).with_context(|| format!("reading {}", o.baseline.display()))?,
    )?;
    let mut failed = false;
    for (name, want) in &base {
        let Some(got) = now.get(name) else { continue };
        for (metric, w, g) in [
            ("intent", want.intent, got.intent),
            ("slot_f1", want.slot_f1, got.slot_f1),
            ("frame", want.frame, got.frame),
        ] {
            let (Some(w), Some(g)) = (w, g) else { continue };
            let bad = g < w - o.tolerance;
            failed |= bad;
            println!(
                "{} {name:<10} {metric:<8} baseline {w:6.2}  now {g:6.2}  ({:+.2})",
                if bad { "FAIL" } else { "ok  " },
                g - w
            );
        }
    }
    if failed {
        bail!("accuracy regressed more than {} points", o.tolerance);
    }
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some((cmd, rest)) = args.split_first() else {
        bail!("usage: rslu-bench fetch|run|check [--dir data] [--only names] [--typo rate]");
    };
    let o = parse_flags(rest)?;
    match cmd.as_str() {
        "fetch" => fetch(&o),
        "run" => run(&o),
        "check" => check(&o),
        other => bail!("unknown command {other:?}"),
    }
}
