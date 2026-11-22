//! Downloads public NLU benchmarks and scores rslu on them.
//!
//! ```text
//! rslu-bench fetch [--dir data] [--only a,b]               download the benchmarks
//! rslu-bench run   [--dir data] [--only a,b] [--typo 0.3]  evaluate and print a report
//! rslu-bench check [--dir data] [--baseline f] [--tolerance 0.5] [--update]
//!                                                          fail if the default profile regressed
//! ```

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use rslu::add_typos;
use rslu_bench::eval::{evaluate, profiles};
use rslu_bench::{check, fetch, sets};

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

fn cell(v: Option<f64>) -> String {
    v.map_or_else(|| "-".to_owned(), |v| format!("{v:.2}%"))
}

fn run(o: &Options) -> Result<()> {
    println!(
        "{:<10} {:<10} {:>6} {:>6} {:>11} {:>8} {:>9} {:>7}",
        "dataset", "profile", "train", "test", "intent-acc", "slot-F1", "frame-acc", "secs"
    );
    for b in sets::selected(&o.only) {
        let (train, mut test) = (b.load)(&o.dir).with_context(|| {
            format!(
                "{} (run `rslu-bench fetch --only {}` first)",
                b.name, b.name
            )
        })?;
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
                cell((b.intents && b.slots).then_some(s.frame).flatten()),
                t.elapsed().as_secs_f64()
            );
        }
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
        "fetch" => fetch::fetch(&o.dir, &o.only),
        "run" => run(&o),
        "check" => check::check(&check::CheckOptions {
            dir: &o.dir,
            only: &o.only,
            baseline: &o.baseline,
            tolerance: o.tolerance,
            update: o.update,
        }),
        other => bail!("unknown command {other:?}"),
    }
}
