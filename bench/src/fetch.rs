//! Downloads the benchmark corpora.

use std::fs::{self, File};
use std::io::{self, Read};
use std::path::Path;

use anyhow::{Context, Result, bail};
use flate2::read::GzDecoder;

use crate::sets::{MASSIVE_URL, selected};

/// Bounds any single downloaded or extracted file (a guard against
/// decompression bombs and runaway responses).
const MAX_BYTES: u64 = 512 << 20;

/// Writes `r` to `dst` through a temporary file, so an interrupted download is
/// never mistaken for a finished one.
fn write_file(dst: &Path, r: impl Read) -> Result<()> {
    fs::create_dir_all(dst.parent().context("destination has no parent")?)?;
    let part = dst.with_extension("part");
    let mut f = File::create(&part)?;
    io::copy(&mut r.take(MAX_BYTES), &mut f)?;
    drop(f);
    fs::rename(&part, dst)?;
    Ok(())
}

fn get(url: &str) -> Result<impl Read + use<>> {
    Ok(ureq::get(url)
        .call()
        .with_context(|| url.to_owned())?
        .into_body()
        .into_reader())
}

fn download(url: &str, dst: &Path) -> Result<()> {
    if dst.exists() {
        return Ok(());
    }
    write_file(dst, get(url)?)?;
    println!("fetched {}", dst.display());
    Ok(())
}

/// Streams the MASSIVE release and keeps only the English file.
fn massive(dir: &Path) -> Result<()> {
    let dst = dir.join("massive/en-US.jsonl");
    if dst.exists() {
        return Ok(());
    }
    let mut archive = tar::Archive::new(GzDecoder::new(get(MASSIVE_URL)?));
    for entry in archive.entries()? {
        let entry = entry?;
        if entry
            .path()?
            .file_name()
            .is_some_and(|n| n == "en-US.jsonl")
        {
            write_file(&dst, entry)?;
            println!("fetched {}", dst.display());
            return Ok(());
        }
    }
    bail!("massive: en-US.jsonl not found in archive")
}

pub fn fetch(dir: &Path, only: &[String]) -> Result<()> {
    for b in selected(only) {
        for (url, rel) in &b.files {
            download(url, &dir.join(rel))?;
        }
        if b.name == "MASSIVE" {
            massive(dir)?;
        }
    }
    println!("datasets ready in {}", dir.display());
    Ok(())
}
