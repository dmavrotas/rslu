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
}
