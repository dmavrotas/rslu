//! A typo-tolerant lexicon mapping words seen inside slot values to slot types.
//! A symmetric-delete index lets words within one edit (insertion, deletion,
//! substitution or adjacent transposition) of a known word match too.

use std::collections::{BTreeSet, HashMap};

use crate::nlu::{Utterance, tokenize};

/// Shortest word that may match approximately; shorter words collide too often.
const MIN_FUZZY_LEN: usize = 4;
/// Folds used so a training utterance never sees a lexicon containing its own
/// slot values.
const JACKKNIFE_FOLDS: usize = 5;

type TypeIndex = HashMap<String, BTreeSet<String>>;

/// Reduces a label like `fromloc.city_name` to `city_name`, so related labels
/// share vocabulary.
pub(crate) fn slot_type(label: &str) -> &str {
    label.rsplit_once('.').map_or(label, |(_, last)| last)
}

#[derive(Debug, Clone, Default)]
pub(crate) struct Lexicon {
    exact: TypeIndex,
    fuzzy: TypeIndex,
}

/// The word itself followed by every single-character deletion of it.
fn deletions(w: &str) -> Vec<String> {
    let chars: Vec<char> = w.chars().collect();
    let mut out = Vec::with_capacity(chars.len() + 1);
    out.push(w.to_owned());
    for skip in 0..chars.len() {
        out.push(
            chars
                .iter()
                .enumerate()
                .filter_map(|(i, c)| (i != skip).then_some(*c))
                .collect(),
        );
    }
    out
}

fn add(index: &mut TypeIndex, key: String, typ: &str) {
    index.entry(key).or_default().insert(typ.to_owned());
}

impl Lexicon {
    fn add_word(&mut self, w: &str, typ: &str) {
        add(&mut self.exact, w.to_owned(), typ);
        if w.chars().count() >= MIN_FUZZY_LEN {
            for k in deletions(w) {
                add(&mut self.fuzzy, k, typ);
            }
        }
    }

    /// Indexes the slot values of `data` plus any external `entries`
    /// (slot type to known values, each tokenised like input text).
    pub(crate) fn build<'a>(
        data: impl IntoIterator<Item = &'a Utterance>,
        entries: &HashMap<String, Vec<String>>,
    ) -> Self {
        let mut lx = Self::default();
        for u in data {
            for s in &u.slots {
                for w in &u.tokens[s.start..s.end] {
                    lx.add_word(w, slot_type(&s.label));
                }
            }
        }
        for (typ, values) in entries {
            for w in values.iter().flat_map(|v| tokenize(v)) {
                lx.add_word(&w, slot_type(typ));
            }
        }
        lx
    }

    /// `gz:<type>` for exact matches and `gzf:<type>` for matches that are only
    /// approximate, in a deterministic order.
    fn features(&self, w: &str) -> Vec<String> {
        let empty = BTreeSet::new();
        let exact = self.exact.get(w).unwrap_or(&empty);
        let mut out: Vec<String> = exact.iter().map(|t| format!("gz:{t}")).collect();
        if w.chars().count() >= MIN_FUZZY_LEN {
            let approx: BTreeSet<&String> = deletions(w)
                .iter()
                .filter_map(|k| self.fuzzy.get(k))
                .flatten()
                .filter(|t| !exact.contains(*t))
                .collect();
            out.extend(approx.into_iter().map(|t| format!("gzf:{t}")));
        }
        out
    }

    pub(crate) fn token_features<S: AsRef<str>>(&self, tokens: &[S]) -> Vec<Vec<String>> {
        tokens.iter().map(|w| self.features(w.as_ref())).collect()
    }
}

/// Per-utterance gazetteer features where utterance `i` only sees a lexicon
/// built from the other folds (plus external entries). `data` holds `n`
/// originals followed by perturbed copies; copy `i % n` shares its original's
/// fold so a copy never sees its own original's slot values.
pub(crate) fn jackknife(
    data: &[Utterance],
    n: usize,
    entries: &HashMap<String, Vec<String>>,
) -> Vec<Vec<Vec<String>>> {
    let fold_of = |i: usize| (i % n) % JACKKNIFE_FOLDS;
    let mut out = vec![Vec::new(); data.len()];
    for fold in 0..JACKKNIFE_FOLDS {
        let rest = data
            .iter()
            .enumerate()
            .filter(|(i, _)| fold_of(*i) != fold)
            .map(|(_, u)| u);
        let lx = Lexicon::build(rest, entries);
        for (i, u) in data.iter().enumerate() {
            if fold_of(i) == fold {
                out[i] = lx.token_features(&u.tokens);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse_annotated;

    fn lexicon() -> Lexicon {
        let data = [parse_annotated("fly to [city : boston] now", "x")];
        Lexicon::build(&data, &HashMap::new())
    }

    #[test]
    fn slot_type_keeps_the_last_segment() {
        assert_eq!(slot_type("fromloc.city_name"), "city_name");
        assert_eq!(slot_type("city"), "city");
    }

    #[test]
    fn exact_and_approximate_matches_are_distinguished() {
        let lx = lexicon();
        assert_eq!(lx.features("boston"), ["gz:city"]);
        assert_eq!(lx.features("bostn"), ["gzf:city"]); // deletion
        assert_eq!(lx.features("zzston"), Vec::<String>::new()); // two edits away
        assert_eq!(lx.features("fly"), Vec::<String>::new());
    }

    #[test]
    fn transposition_and_substitution_are_one_edit() {
        let lx = lexicon();
        assert_eq!(lx.features("bostno"), ["gzf:city"]);
        assert_eq!(lx.features("boston"), ["gz:city"]);
        assert_eq!(lx.features("bastan"), Vec::<String>::new());
        assert_eq!(lx.features("bostan"), ["gzf:city"]);
    }

    #[test]
    fn external_entries_are_tokenised_and_indexed() {
        let entries = HashMap::from([("driver".to_owned(), vec!["Lewis Hamilton".to_owned()])]);
        let lx = Lexicon::build(&[], &entries);
        assert_eq!(lx.features("hamilton"), ["gz:driver"]);
        assert_eq!(lx.features("hamliton"), ["gzf:driver"]); // adjacent transposition
    }

    #[test]
    fn jackknife_hides_an_utterance_s_own_values() {
        let cities = [
            "boston", "denver", "austin", "dallas", "tucson", "miami", "fresno", "omaha", "tulsa",
            "madison",
        ];
        let data: Vec<Utterance> = cities
            .iter()
            .map(|c| parse_annotated(&format!("go to [city : {c}]"), "x"))
            .collect();
        let gaz = jackknife(&data, data.len(), &HashMap::new());
        for (i, g) in gaz.iter().enumerate() {
            assert!(g[2].is_empty(), "utterance {i} saw its own value: {g:?}");
        }
    }
}
