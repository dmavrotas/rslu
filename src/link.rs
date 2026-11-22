//! Entity linking: resolves extracted slot text to the closest known entry of
//! the same slot type, tolerating typos. Entries earlier in a list win ties, so
//! callers can rank entries by importance (for example drivers by career).

use std::collections::HashMap;

use crate::gazetteer::slot_type;
use crate::nlu::tokenize;

/// A linked entry and how well it matched.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Link<'a> {
    /// The entry in its original spelling.
    pub entry: &'a str,
    /// Match cost, lower is better: 0 for the whole entry, 1 for its last word
    /// (a surname), 2 for any earlier word (a forename), plus 3 per typo edit.
    pub cost: u32,
}

#[derive(Debug, Clone)]
struct Entry {
    value: String,
    norm: String,
    tokens: Vec<String>,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct Linker {
    by_type: HashMap<String, Vec<Entry>>,
}

impl Linker {
    pub(crate) fn new(entries: &HashMap<String, Vec<String>>) -> Self {
        let mut types: Vec<&String> = entries.keys().collect();
        types.sort(); // deterministic when two labels share a last segment
        let mut by_type: HashMap<String, Vec<Entry>> = HashMap::new();
        for typ in types {
            for value in &entries[typ] {
                let tokens = tokenize(value);
                if tokens.is_empty() {
                    continue;
                }
                by_type
                    .entry(slot_type(typ).to_owned())
                    .or_default()
                    .push(Entry {
                        value: value.clone(),
                        norm: tokens.join(" "),
                        tokens,
                    });
            }
        }
        Self { by_type }
    }

    /// The best entry for `text` (already lower-cased), or `None` when nothing
    /// is close enough. Ties go to the earlier entry.
    pub(crate) fn best(&self, label: &str, text: &str) -> Option<Link<'_>> {
        let cands = self.by_type.get(slot_type(label))?;
        if text.is_empty() {
            return None;
        }
        let budget = max_edits(text.chars().count());
        let mut best: Option<Link<'_>> = None;
        for e in cands {
            if let Some(cost) = entry_cost(e, text, budget)
                && best.is_none_or(|b| cost < b.cost)
            {
                best = Some(Link {
                    entry: &e.value,
                    cost,
                });
            }
        }
        best
    }
}

/// The typo budget for a word of `n` characters: none for very short words (too
/// many false matches), one edit up to 7 letters, two beyond.
fn max_edits(n: usize) -> u32 {
    match n {
        0..=3 => 0,
        4..=7 => 1,
        _ => 2,
    }
}

/// Scores how well `text` matches an entry (lower is better). The base cost
/// depends on what matched: the whole entry 0, the last word 1, any earlier
/// word 2. The last word wins over earlier ones because it is the identifying
/// part of a name. Every edit adds 3, so the three base levels never overlap:
/// any exact match beats any typo.
fn entry_cost(e: &Entry, text: &str, budget: u32) -> Option<u32> {
    let whole = Some(edit_distance(text, &e.norm))
        .filter(|&d| d <= budget)
        .map(|d| 3 * d);
    let last = e.tokens.len() - 1;
    let words = e.tokens.iter().enumerate().filter_map(|(i, w)| {
        let d = edit_distance(text, w);
        (d <= budget).then(|| 3 * d + if i == last { 1 } else { 2 })
    });
    whole.into_iter().chain(words).min()
}

/// Optimal-string-alignment distance: insertions, deletions, substitutions and
/// adjacent transpositions each cost 1.
pub(crate) fn edit_distance(a: &str, b: &str) -> u32 {
    let (a, b): (Vec<char>, Vec<char>) = (a.chars().collect(), b.chars().collect());
    let n = b.len();
    let mut prev2 = vec![0u32; n + 1];
    let mut prev: Vec<u32> = (0..=n as u32).collect();
    let mut cur = vec![0u32; n + 1];
    for i in 1..=a.len() {
        cur[0] = i as u32;
        for j in 1..=n {
            let sub = u32::from(a[i - 1] != b[j - 1]);
            cur[j] = (prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + sub);
            if i > 1 && j > 1 && a[i - 1] == b[j - 2] && a[i - 2] == b[j - 1] {
                cur[j] = cur[j].min(prev2[j - 2] + 1);
            }
        }
        std::mem::swap(&mut prev2, &mut prev);
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[n]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn linker() -> Linker {
        let entries = HashMap::from([(
            "driver".to_owned(),
            vec![
                "Lewis Hamilton".to_owned(),
                "Max Verstappen".to_owned(),
                "Wilson Fittipaldi".to_owned(),
                "Justin Wilson".to_owned(),
            ],
        )]);
        Linker::new(&entries)
    }

    #[test]
    fn edit_distance_counts_transpositions_once() {
        assert_eq!(edit_distance("hamilton", "hamliton"), 1);
        assert_eq!(edit_distance("kitten", "sitting"), 3);
        assert_eq!(edit_distance("", "abc"), 3);
        assert_eq!(edit_distance("same", "same"), 0);
        assert_eq!(edit_distance("ab", "ba"), 1);
    }

    #[test]
    fn surname_links_to_the_full_entry() {
        let l = linker();
        let hit = l.best("driver", "hamilton").unwrap();
        assert_eq!((hit.entry, hit.cost), ("Lewis Hamilton", 1));
    }

    #[test]
    fn whole_entry_costs_zero_and_typos_still_link() {
        let l = linker();
        assert_eq!(l.best("driver", "max verstappen").unwrap().cost, 0);
        let typo = l.best("driver", "verstapen").unwrap();
        assert_eq!((typo.entry, typo.cost), ("Max Verstappen", 4));
    }

    #[test]
    fn surname_beats_forename() {
        let l = linker();
        let hit = l.best("driver", "wilson").unwrap();
        assert_eq!(hit.entry, "Justin Wilson");
    }

    #[test]
    fn short_words_get_no_typo_budget_and_unknowns_do_not_link() {
        let l = linker();
        assert!(l.best("driver", "max").is_some());
        assert!(l.best("driver", "mxa").is_none());
        assert!(l.best("driver", "nobody").is_none());
        assert!(l.best("driver", "").is_none());
        assert!(l.best("team", "hamilton").is_none());
    }

    #[test]
    fn dotted_labels_share_entries_and_ties_go_to_the_first() {
        let entries = HashMap::from([(
            "fromloc.city".to_owned(),
            vec![
                "Springfield Illinois".to_owned(),
                "Springfield Missouri".to_owned(),
            ],
        )]);
        let l = Linker::new(&entries);
        assert_eq!(
            l.best("toloc.city", "springfield").unwrap().entry,
            "Springfield Illinois"
        );
    }
}
