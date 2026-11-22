//! Parsers for public intent/slot benchmark corpora.

use std::io::{BufRead, BufReader, Read};

use anyhow::{Context, Result, bail};
use rslu::{Span, Utterance, spans_from_bio, tokenize};
use serde::Deserialize;

/// Reads the ATIS `.w-intent.iob` format: tab-separated, tokens (wrapped in
/// BOS/EOS) on the left, one IOB tag per token on the right. The tag aligned
/// with EOS is the intent label. Multi-intent labels such as `a#b` are kept
/// verbatim as one class.
pub fn parse_atis(r: impl Read) -> Result<Vec<Utterance>> {
    let mut out = Vec::new();
    for (n, line) in BufReader::new(r).lines().enumerate() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let n = n + 1;
        let Some((left, right)) = line.split_once('\t') else {
            bail!("line {n}: missing tab separator");
        };
        let words: Vec<&str> = left.split_whitespace().collect();
        let fields: Vec<&str> = right.split_whitespace().collect();
        if words.len() < 3 || fields.len() != words.len() {
            bail!("line {n}: {} tokens but {} tags", words.len(), fields.len());
        }
        let intent = fields[fields.len() - 1].to_owned();
        // Drop the BOS/EOS sentinels; the tag at EOS was the intent.
        let inner = 1..words.len() - 1;
        out.push(Utterance {
            tokens: words[inner.clone()]
                .iter()
                .map(|w| (*w).to_owned())
                .collect(),
            intent,
            slots: spans_from_bio(&fields[inner]),
        });
    }
    Ok(out)
}

#[derive(Deserialize)]
struct SnipsChunk {
    text: String,
    #[serde(default)]
    entity: Option<String>,
}

#[derive(Deserialize)]
struct SnipsItem {
    data: Vec<SnipsChunk>,
}

/// Reads one Snips NLU benchmark JSON file (`{"Intent": [{"data": [...]}]}`).
///
/// Some upstream files contain invalid UTF-8; like Go's decoder, invalid bytes
/// become U+FFFD instead of failing the whole file.
pub fn parse_snips(mut r: impl Read) -> Result<Vec<Utterance>> {
    let mut bytes = Vec::new();
    r.read_to_end(&mut bytes)?;
    let file: std::collections::BTreeMap<String, Vec<SnipsItem>> =
        serde_json::from_str(&String::from_utf8_lossy(&bytes)).context("invalid snips json")?;
    let mut out = Vec::new();
    for (intent, items) in file {
        for item in items {
            let mut u = Utterance {
                intent: intent.clone(),
                ..Utterance::default()
            };
            for chunk in item.data {
                let toks = tokenize(&chunk.text);
                if let Some(entity) = chunk.entity.filter(|e| !e.is_empty())
                    && !toks.is_empty()
                {
                    let start = u.tokens.len();
                    u.slots.push(Span::new(start, start + toks.len(), entity));
                }
                u.tokens.extend(toks);
            }
            out.push(u);
        }
    }
    Ok(out)
}

/// Reads all of `r`, replacing invalid UTF-8 with U+FFFD like Go's decoder.
fn read_lossy(mut r: impl Read) -> Result<String> {
    let mut bytes = Vec::new();
    r.read_to_end(&mut bytes)?;
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

/// Reads CLINC150's `data_full.json` and returns one split (`train`, `val`,
/// `test`, or the out-of-scope `oos_*` splits) as intent-only utterances.
pub fn parse_clinc(r: impl Read, split: &str) -> Result<Vec<Utterance>> {
    let mut raw: std::collections::HashMap<String, Vec<Vec<String>>> =
        serde_json::from_str(&read_lossy(r)?).context("invalid clinc json")?;
    let Some(rows) = raw.remove(split) else {
        bail!("clinc: unknown split {split:?}");
    };
    rows.into_iter()
        .enumerate()
        .map(|(i, row)| match <[String; 2]>::try_from(row) {
            Ok([text, intent]) => Ok(Utterance {
                tokens: tokenize(&text),
                intent,
                slots: Vec::new(),
            }),
            Err(row) => bail!("clinc: {split} row {i} has {} fields", row.len()),
        })
        .collect()
}

/// Reads a BANKING77 CSV file (header `text,category`).
pub fn parse_banking(r: impl Read) -> Result<Vec<Utterance>> {
    let text = read_lossy(r)?;
    let mut out = Vec::new();
    for (i, rec) in csv::Reader::from_reader(text.as_bytes())
        .records()
        .enumerate()
    {
        let rec = rec?;
        if rec.len() != 2 {
            bail!("banking: row {} has {} fields", i + 1, rec.len());
        }
        out.push(Utterance {
            tokens: tokenize(&rec[0]),
            intent: rec[1].to_owned(),
            slots: Vec::new(),
        });
    }
    Ok(out)
}

#[derive(Deserialize)]
struct MassiveRecord {
    partition: String,
    intent: String,
    annot_utt: String,
}

/// Reads a MASSIVE JSONL file and keeps one partition (`train`, `dev` or
/// `test`). Slots come from the inline `[type : value]` markup.
pub fn parse_massive(r: impl Read, partition: &str) -> Result<Vec<Utterance>> {
    let mut out = Vec::new();
    for (n, line) in read_lossy(r)?.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let rec: MassiveRecord =
            serde_json::from_str(line).with_context(|| format!("massive line {}", n + 1))?;
        if rec.partition == partition {
            out.push(rslu::parse_annotated(&rec.annot_utt, &rec.intent));
        }
    }
    Ok(out)
}

/// Reads the MIT Restaurant/Movie corpora: one `tag<TAB>word` per line, blank
/// lines between sentences. These are slot-only (no intent).
pub fn parse_mit_bio(r: impl Read) -> Result<Vec<Utterance>> {
    fn flush(out: &mut Vec<Utterance>, toks: &mut Vec<String>, tags: &mut Vec<String>) {
        if !toks.is_empty() {
            out.push(Utterance {
                tokens: std::mem::take(toks),
                intent: String::new(),
                slots: spans_from_bio(tags),
            });
        }
        tags.clear();
    }
    let (mut out, mut toks, mut tags) = (Vec::new(), Vec::new(), Vec::new());
    for (n, line) in read_lossy(r)?.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            flush(&mut out, &mut toks, &mut tags);
            continue;
        }
        let Some((tag, word)) = line.split_once('\t') else {
            bail!("mit-bio line {}: expected tag<TAB>word", n + 1);
        };
        tags.push(tag.to_owned());
        toks.push(word.to_owned());
    }
    flush(&mut out, &mut toks, &mut tags);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn atis_drops_sentinels_and_reads_the_intent() {
        let line = "BOS show flights to boston EOS\tO O O O B-toloc.city_name atis_flight\n";
        let u = parse_atis(line.as_bytes()).unwrap();
        assert_eq!(u[0].tokens, ["show", "flights", "to", "boston"]);
        assert_eq!(u[0].intent, "atis_flight");
        assert_eq!(u[0].slots, [Span::new(3, 4, "toloc.city_name")]);
    }

    #[test]
    fn atis_rejects_misaligned_lines() {
        assert!(parse_atis("BOS a EOS\tO O\n".as_bytes()).is_err());
        assert!(parse_atis("no tab here\n".as_bytes()).is_err());
    }

    #[test]
    fn snips_tolerates_invalid_utf8() {
        let mut json = br#"{"X":[{"data":[{"text":"pop "#.to_vec();
        json.extend([0xff, 0xfe]);
        json.extend(br#" punk"}]}]}"#);
        assert_eq!(parse_snips(json.as_slice()).unwrap().len(), 1);
    }

    #[test]
    fn snips_chunks_become_slots() {
        let json = r#"{"GetWeather":[{"data":[{"text":"weather in "},{"text":"New York","entity":"city"},{"text":" today"}]}]}"#;
        let u = parse_snips(json.as_bytes()).unwrap();
        assert_eq!(u[0].tokens, ["weather", "in", "new", "york", "today"]);
        assert_eq!(u[0].slots, [Span::new(2, 4, "city")]);
        assert_eq!(u[0].intent, "GetWeather");
    }

    #[test]
    fn clinc_returns_the_requested_split() {
        let json = r#"{"train":[["what is my balance","balance"]],"test":[["hi there","greet"]]}"#;
        let u = parse_clinc(json.as_bytes(), "test").unwrap();
        assert_eq!((u[0].tokens.len(), u[0].intent.as_str()), (2, "greet"));
        assert!(parse_clinc(json.as_bytes(), "nope").is_err());
        assert!(parse_clinc(r#"{"train":[["only one"]]}"#.as_bytes(), "train").is_err());
    }

    #[test]
    fn banking_handles_quoted_commas() {
        let csv = "text,category\n\"Hello, I am waiting?\",card_arrival\nplain,other\n";
        let u = parse_banking(csv.as_bytes()).unwrap();
        assert_eq!(u.len(), 2);
        assert_eq!(u[0].tokens, ["hello", "i", "am", "waiting"]);
        assert_eq!(u[1].intent, "other");
    }

    #[test]
    fn massive_filters_partition_and_reads_markup() {
        let jsonl = concat!(
            r#"{"partition":"train","intent":"alarm_set","annot_utt":"wake me at [time : nine am]"}"#,
            "\n",
            r#"{"partition":"test","intent":"x","annot_utt":"hello"}"#,
            "\n\n",
        );
        let u = parse_massive(jsonl.as_bytes(), "train").unwrap();
        assert_eq!(u.len(), 1);
        assert_eq!(u[0].slots, [Span::new(3, 5, "time")]);
        assert!(parse_massive("not json".as_bytes(), "train").is_err());
    }

    #[test]
    fn mit_bio_splits_sentences_on_blank_lines() {
        let bio = "B-Rating\t2\nI-Rating\tstar\nO\trestaurants\n\nO\thi\n";
        let u = parse_mit_bio(bio.as_bytes()).unwrap();
        assert_eq!(u.len(), 2);
        assert_eq!(u[0].tokens, ["2", "star", "restaurants"]);
        assert_eq!(u[0].slots, [Span::new(0, 2, "Rating")]);
        assert_eq!(u[0].intent, "");
        assert!(parse_mit_bio("no-tab-here\n".as_bytes()).is_err());
    }
}
