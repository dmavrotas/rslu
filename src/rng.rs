//! A tiny deterministic PRNG, so training is reproducible without a dependency.

/// `SplitMix64`: fast, small, statistically fine for shuffling and noise.
pub(crate) struct SplitMix64(u64);

impl SplitMix64 {
    pub(crate) fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub(crate) fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)`.
    pub(crate) fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform in `0..n`; `n` must be non-zero.
    pub(crate) fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }

    /// Fisher–Yates shuffle.
    pub(crate) fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            items.swap(i, self.below(i + 1));
        }
    }

    /// A shuffled `0..n`.
    pub(crate) fn permutation(&mut self, n: usize) -> Vec<usize> {
        let mut order: Vec<usize> = (0..n).collect();
        self.shuffle(&mut order);
        order
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_same_stream() {
        let (mut a, mut b) = (SplitMix64::new(3), SplitMix64::new(3));
        assert!((0..10).all(|_| a.next_u64() == b.next_u64()));
    }

    #[test]
    fn floats_are_in_unit_interval_and_permutations_are_complete() {
        let mut r = SplitMix64::new(1);
        assert!(
            (0..1000)
                .map(|_| r.next_f64())
                .all(|f| (0.0..1.0).contains(&f))
        );
        let mut p = r.permutation(50);
        p.sort_unstable();
        assert_eq!(p, (0..50).collect::<Vec<_>>());
    }
}
