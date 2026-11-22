//! The CI regression gate: score the default profile and compare with a stored
//! baseline.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use anyhow::{Context, Result, bail};

use crate::eval::{Scores, evaluate, profiles};
use crate::sets::{CHECK_SET, selected};

/// Benchmark name to the scores of the default profile.
pub type Baseline = BTreeMap<String, Scores>;

#[derive(Debug, Clone, PartialEq)]
pub struct Violation {
    pub bench: String,
    /// `intent`, `slot-F1`, `frame`, or `missing`.
    pub metric: &'static str,
    pub base: f64,
    pub cur: f64,
}

impl Violation {
    #[cfg(test)]
    pub fn key(&self) -> String {
        format!("{} {}", self.bench, self.metric)
    }
}

impl std::fmt::Display for Violation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.metric == "missing" {
            return write!(f, "{}: no result produced", self.bench);
        }
        write!(
            f,
            "{} {}: {:.2} -> {:.2} ({:+.2} points)",
            self.bench,
            self.metric,
            self.base,
            self.cur,
            self.cur - self.base
        )
    }
}

/// Every metric in `cur` that fell more than `tol` points below `base`, plus
/// benchmarks that have a baseline but no result. Improvements never fail, and
/// a metric the baseline does not have is not gated.
pub fn compare(base: &Baseline, cur: &Baseline, tol: f64) -> Vec<Violation> {
    let mut out = Vec::new();
    for (name, b) in base {
        let Some(c) = cur.get(name) else {
            out.push(Violation {
                bench: name.clone(),
                metric: "missing",
                base: 0.0,
                cur: 0.0,
            });
            continue;
        };
        for (metric, base, cur) in [
            ("intent", b.intent, c.intent),
            ("slot-F1", b.slot_f1, c.slot_f1),
            ("frame", b.frame, c.frame),
        ] {
            let Some(base) = base else { continue };
            let cur = cur.unwrap_or(0.0);
            if cur < base - tol - 1e-9 {
                out.push(Violation {
                    bench: name.clone(),
                    metric,
                    base,
                    cur,
                });
            }
        }
    }
    out
}

pub fn read_baseline(path: &Path) -> Result<Baseline> {
    let bytes = fs::read(path).with_context(|| format!("reading baseline {}", path.display()))?;
    serde_json::from_slice(&bytes).with_context(|| format!("parsing baseline {}", path.display()))
}

pub fn write_baseline(path: &Path, baseline: &Baseline) -> Result<()> {
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir)?;
    }
    fs::write(path, serde_json::to_string_pretty(baseline)? + "\n")?;
    Ok(())
}

pub struct CheckOptions<'a> {
    pub dir: &'a Path,
    pub only: &'a [String],
    pub baseline: &'a Path,
    pub tolerance: f64,
    pub update: bool,
}

/// Scores the default profile on [`CHECK_SET`] (or `only`) and compares with the
/// baseline, or rewrites the baseline with `update`.
pub fn check(o: &CheckOptions<'_>) -> Result<()> {
    let only: Vec<String> = if o.only.is_empty() {
        CHECK_SET.map(str::to_owned).to_vec()
    } else {
        o.only.to_vec()
    };
    let default = profiles().remove(0);
    let mut cur = Baseline::new();
    for b in selected(&only) {
        let (train, test) = (b.load)(o.dir).with_context(|| {
            format!(
                "{} (run `rslu-bench fetch --only {}` first)",
                b.name, b.name
            )
        })?;
        let s = evaluate(&b, &default, &train, &test);
        println!(
            "{:<10} intent={:.2} slot-F1={:.2} frame={:.2}",
            b.name,
            s.intent.unwrap_or(0.0),
            s.slot_f1.unwrap_or(0.0),
            s.frame.unwrap_or(0.0)
        );
        cur.insert(b.name.to_owned(), s);
    }
    if o.update {
        write_baseline(o.baseline, &cur)?;
        println!("baseline written to {}", o.baseline.display());
        return Ok(());
    }
    let mut base = read_baseline(o.baseline)?;
    // Only gate on what was asked for.
    base.retain(|n, _| cur.contains_key(n) || only.iter().any(|x| x.eq_ignore_ascii_case(n)));
    let violations = compare(&base, &cur, o.tolerance);
    for v in &violations {
        println!("REGRESSION: {v}");
    }
    if !violations.is_empty() {
        bail!(
            "{} metric(s) regressed by more than {:.2} points",
            violations.len(),
            o.tolerance
        );
    }
    println!(
        "OK: no metric dropped more than {:.2} points below the baseline",
        o.tolerance
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(intent: Option<f64>, slot: Option<f64>, frame: Option<f64>) -> Scores {
        Scores {
            intent,
            slot_f1: slot,
            frame,
        }
    }

    fn base() -> Baseline {
        Baseline::from([
            ("ATIS".into(), s(Some(93.7), Some(93.3), Some(79.6))),
            ("BANKING77".into(), s(Some(91.0), None, None)),
        ])
    }

    fn keys(cur: &Baseline, tol: f64) -> Vec<String> {
        compare(&base(), cur, tol)
            .iter()
            .map(Violation::key)
            .collect()
    }

    fn with(atis: Scores, banking: Scores) -> Baseline {
        Baseline::from([("ATIS".into(), atis), ("BANKING77".into(), banking)])
    }

    #[test]
    fn identical_improved_and_within_tolerance_pass() {
        assert!(keys(&base(), 0.5).is_empty());
        let better = with(
            s(Some(95.0), Some(94.0), Some(81.0)),
            s(Some(92.0), None, None),
        );
        assert!(keys(&better, 0.5).is_empty());
        let small = with(
            s(Some(93.4), Some(93.3), Some(79.6)),
            s(Some(91.0), None, None),
        );
        assert!(keys(&small, 0.5).is_empty());
        let edge = with(
            s(Some(93.2), Some(93.3), Some(79.6)),
            s(Some(91.0), None, None),
        );
        assert!(
            keys(&edge, 0.5).is_empty(),
            "exactly the tolerance is allowed"
        );
    }

    #[test]
    fn regressions_are_named_per_metric() {
        let intent = with(
            s(Some(92.0), Some(93.3), Some(79.6)),
            s(Some(91.0), None, None),
        );
        assert_eq!(keys(&intent, 0.5), ["ATIS intent"]);
        let several = with(
            s(Some(93.7), Some(90.0), Some(70.0)),
            s(Some(85.0), None, None),
        );
        assert_eq!(
            keys(&several, 0.5),
            ["ATIS slot-F1", "ATIS frame", "BANKING77 intent"]
        );
    }

    #[test]
    fn a_missing_benchmark_or_metric_is_a_violation() {
        let only_atis = Baseline::from([("ATIS".into(), base()["ATIS"])]);
        assert_eq!(keys(&only_atis, 0.5), ["BANKING77 missing"]);
        let lost_metric = with(s(Some(93.7), None, Some(79.6)), s(Some(91.0), None, None));
        assert_eq!(keys(&lost_metric, 0.5), ["ATIS slot-F1"]);
    }

    #[test]
    fn violations_describe_themselves() {
        let v = compare(&base(), &Baseline::new(), 0.5);
        assert_eq!(v[0].to_string(), "ATIS: no result produced");
        let drop = Violation {
            bench: "ATIS".into(),
            metric: "intent",
            base: 93.7,
            cur: 92.0,
        };
        assert_eq!(
            drop.to_string(),
            "ATIS intent: 93.70 -> 92.00 (-1.70 points)"
        );
    }

    #[test]
    fn baseline_round_trips_and_missing_file_mentions_the_baseline() {
        let dir = std::env::temp_dir().join(format!("rslu-bench-{}", std::process::id()));
        let path = dir.join("sub/baseline.json");
        write_baseline(&path, &base()).unwrap();
        assert_eq!(read_baseline(&path).unwrap(), base());
        let err = read_baseline(&dir.join("nope.json")).unwrap_err();
        assert!(err.to_string().contains("baseline"), "{err}");
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn baseline_json_matches_gonlu_s_format() {
        let json =
            r#"{"ATIS":{"intent":93.7,"slot_f1":93.3,"frame":79.6},"BANKING77":{"intent":91.0}}"#;
        assert_eq!(serde_json::from_str::<Baseline>(json).unwrap(), base());
    }
}
