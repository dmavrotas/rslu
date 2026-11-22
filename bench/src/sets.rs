//! The benchmark registry: what each public corpus measures, where its files
//! come from and how to load its train and test splits.

use std::fs::File;
use std::path::Path;

use anyhow::{Context, Result};
use rslu::Utterance;

use crate::dataset;

pub type Split = (Vec<Utterance>, Vec<Utterance>);

pub struct Benchmark {
    pub name: &'static str,
    /// Has intent labels.
    pub intents: bool,
    /// Has slot labels.
    pub slots: bool,
    /// `(url, path relative to the data dir)`; MASSIVE is fetched separately.
    pub files: Vec<(String, String)>,
    pub load: fn(&Path) -> Result<Split>,
}

const ATIS_BASE: &str = "https://raw.githubusercontent.com/yvchen/JointSLU/master/data/";
const SNIPS_BASE: &str =
    "https://raw.githubusercontent.com/snipsco/nlu-benchmark/master/2017-06-custom-intent-engines/";
const CLINC_URL: &str =
    "https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_full.json";
const BANKING_BASE: &str =
    "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/";
const MIT_BASE: &str = "https://groups.csail.mit.edu/sls/downloads/restaurant/";
pub const MASSIVE_URL: &str =
    "https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.0.tar.gz";

const SNIPS_INTENTS: [&str; 7] = [
    "AddToPlaylist",
    "BookRestaurant",
    "GetWeather",
    "PlayMusic",
    "RateBook",
    "SearchCreativeWork",
    "SearchScreeningEvent",
];

/// The subset used as a CI gate: small enough to run in a few minutes, broad
/// enough to cover joint, slot-heavy and many-class tasks.
pub const CHECK_SET: [&str; 4] = ["ATIS", "SNIPS", "CLINC150", "BANKING77"];

fn open(dir: &Path, rel: &str) -> Result<File> {
    let path = dir.join(rel);
    File::open(&path).with_context(|| format!("{} (run `rslu-bench fetch` first)", path.display()))
}

fn pair(
    dir: &Path,
    train: &str,
    test: &str,
    parse: fn(File) -> Result<Vec<Utterance>>,
) -> Result<Split> {
    let load = |rel: &str| parse(open(dir, rel)?).with_context(|| rel.to_owned());
    Ok((load(train)?, load(test)?))
}

fn load_snips(dir: &Path, prefix: &str, suffix: &str) -> Result<Vec<Utterance>> {
    let mut all = Vec::new();
    for intent in SNIPS_INTENTS {
        let rel = format!("snips/{prefix}{intent}{suffix}");
        all.extend(dataset::parse_snips(open(dir, &rel)?).with_context(|| rel.clone())?);
    }
    Ok(all)
}

/// Loads two partitions/splits of one file.
fn two_splits(
    dir: &Path,
    rel: &str,
    parse: fn(File, &str) -> Result<Vec<Utterance>>,
    names: [&str; 2],
) -> Result<Split> {
    let load = |name: &str| parse(open(dir, rel)?, name).with_context(|| rel.to_owned());
    Ok((load(names[0])?, load(names[1])?))
}

fn url_files(base: &str, dir: &str, names: &[&str]) -> Vec<(String, String)> {
    names
        .iter()
        .map(|f| (format!("{base}{f}"), format!("{dir}/{f}")))
        .collect()
}

pub fn benchmarks() -> Vec<Benchmark> {
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
            files: url_files(
                ATIS_BASE,
                "atis",
                &["atis.train.w-intent.iob", "atis.test.w-intent.iob"],
            ),
            load: |dir| {
                pair(
                    dir,
                    "atis/atis.train.w-intent.iob",
                    "atis/atis.test.w-intent.iob",
                    dataset::parse_atis,
                )
            },
        },
        Benchmark {
            name: "SNIPS",
            intents: true,
            slots: true,
            files: snips_files,
            load: |dir| {
                Ok((
                    load_snips(dir, "train_", "_full.json")?,
                    load_snips(dir, "validate_", ".json")?,
                ))
            },
        },
        Benchmark {
            name: "MASSIVE",
            intents: true,
            slots: true,
            files: Vec::new(), // fetched by `fetch::massive` (tar.gz)
            load: |dir| {
                two_splits(
                    dir,
                    "massive/en-US.jsonl",
                    dataset::parse_massive,
                    ["train", "test"],
                )
            },
        },
        Benchmark {
            name: "CLINC150",
            intents: true,
            slots: false,
            files: vec![(CLINC_URL.to_owned(), "clinc/data_full.json".to_owned())],
            load: |dir| {
                two_splits(
                    dir,
                    "clinc/data_full.json",
                    dataset::parse_clinc,
                    ["train", "test"],
                )
            },
        },
        Benchmark {
            name: "BANKING77",
            intents: true,
            slots: false,
            files: url_files(BANKING_BASE, "banking77", &["train.csv", "test.csv"]),
            load: |dir| {
                pair(
                    dir,
                    "banking77/train.csv",
                    "banking77/test.csv",
                    dataset::parse_banking,
                )
            },
        },
        Benchmark {
            name: "MIT-Rest",
            intents: false,
            slots: true,
            files: url_files(
                MIT_BASE,
                "mit",
                &["restauranttrain.bio", "restauranttest.bio"],
            ),
            load: |dir| {
                pair(
                    dir,
                    "mit/restauranttrain.bio",
                    "mit/restauranttest.bio",
                    dataset::parse_mit_bio,
                )
            },
        },
    ]
}

/// The benchmarks named in `only` (case-insensitive), or all when empty.
pub fn selected<S: AsRef<str>>(only: &[S]) -> Vec<Benchmark> {
    benchmarks()
        .into_iter()
        .filter(|b| only.is_empty() || only.iter().any(|n| n.as_ref().eq_ignore_ascii_case(b.name)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_is_consistent() {
        let all = benchmarks();
        assert_eq!(all.len(), 6);
        for name in CHECK_SET {
            assert!(all.iter().any(|b| b.name == name), "{name} missing");
        }
        // every benchmark scores something, and MASSIVE is the only one without URLs
        assert!(all.iter().all(|b| b.intents || b.slots));
        assert!(
            all.iter()
                .filter(|b| b.files.is_empty())
                .all(|b| b.name == "MASSIVE")
        );
    }

    #[test]
    fn selection_is_case_insensitive() {
        assert_eq!(selected(&["atis", "mit-rest"]).len(), 2);
        assert_eq!(selected::<&str>(&[]).len(), 6);
        assert!(selected(&["nope"]).is_empty());
    }
}
