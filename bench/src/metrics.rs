//! The standard NLU evaluation measures: intent accuracy, exact-match slot-span
//! F1 (conlleval-style) and frame accuracy.

use std::collections::HashSet;

use rslu::Span;

/// Fraction of positions where `got` equals `want`.
pub fn accuracy<S: AsRef<str>>(got: &[S], want: &[S]) -> f64 {
    if want.is_empty() {
        return 0.0;
    }
    let hit = got
        .iter()
        .zip(want)
        .filter(|(g, w)| g.as_ref() == w.as_ref())
        .count();
    hit as f64 / want.len() as f64
}

/// Micro-averaged span F1. A predicted span counts only if start, end and label
/// all match a gold span exactly. Returns F1 in `[0, 1]`.
pub fn slot_f1(got: &[Vec<Span>], want: &[Vec<Span>]) -> f64 {
    let (mut tp, mut n_got, mut n_want) = (0usize, 0usize, 0usize);
    for (i, gold) in want.iter().enumerate() {
        n_want += gold.len();
        let Some(pred) = got.get(i) else { continue };
        n_got += pred.len();
        let gold: HashSet<&Span> = gold.iter().collect();
        tp += pred.iter().filter(|s| gold.contains(s)).count();
    }
    let precision = if n_got > 0 {
        tp as f64 / n_got as f64
    } else {
        0.0
    };
    let recall = if n_want > 0 {
        tp as f64 / n_want as f64
    } else {
        0.0
    };
    if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    }
}

fn same_spans(a: &[Span], b: &[Span]) -> bool {
    let sorted = |s: &[Span]| {
        let mut v = s.to_vec();
        v.sort();
        v
    };
    a.len() == b.len() && sorted(a) == sorted(b)
}

/// Fraction of utterances where the intent and the full set of slots are both
/// exactly right.
pub fn frame_accuracy(
    got_intent: &[String],
    want_intent: &[String],
    got: &[Vec<Span>],
    want: &[Vec<Span>],
) -> f64 {
    if want_intent.is_empty() {
        return 0.0;
    }
    let hit = (0..want_intent.len())
        .filter(|&i| {
            got_intent.get(i) == want_intent.get(i)
                && got
                    .get(i)
                    .zip(want.get(i))
                    .is_some_and(|(g, w)| same_spans(g, w))
        })
        .count();
    hit as f64 / want_intent.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(a: usize, b: usize, l: &str) -> Span {
        Span::new(a, b, l)
    }

    #[test]
    fn accuracy_counts_matches() {
        assert!((accuracy(&["a", "b"], &["a", "c"]) - 0.5).abs() < 1e-12);
        assert_eq!(accuracy::<&str>(&[], &[]), 0.0);
    }

    #[test]
    fn slot_f1_requires_exact_spans() {
        let want = vec![vec![s(0, 2, "x"), s(3, 4, "y")]];
        let got = vec![vec![s(0, 2, "x"), s(3, 5, "y")]]; // second is off by one token
        // precision 1/2, recall 1/2
        assert!((slot_f1(&got, &want) - 0.5).abs() < 1e-12);
        assert_eq!(slot_f1(&[vec![]], &[vec![]]), 0.0);
    }

    #[test]
    fn frame_needs_intent_and_all_slots() {
        let want_i = vec!["a".to_owned(), "b".to_owned()];
        let got_i = vec!["a".to_owned(), "x".to_owned()];
        let want = vec![vec![s(0, 1, "p"), s(2, 3, "q")], vec![]];
        let got = vec![vec![s(2, 3, "q"), s(0, 1, "p")], vec![]];
        assert!((frame_accuracy(&got_i, &want_i, &got, &want) - 0.5).abs() < 1e-12);
    }
}
