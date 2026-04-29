//! Recall decoder: emergent R2 spike sets back to text candidates.
//!
//! R1 receives fixed text-SDR addresses, but the assemblies STDP grows
//! inside R2 are *emergent* — they live at neuron indices the encoder
//! never directly addressed. There is no inverse hash. Instead, the
//! decoder keeps a learnt mapping from concept words to the R2 indices
//! that fire in response to that concept. Pattern-completion candidates
//! are then ranked by how many of a stored engram's bits the live
//! recall set contains.
//!
//! The ratio is asymmetric on purpose:
//!     score = |recall ∩ stored| / |stored|
//! That asks "how much of the stored engram is *contained* in the
//! recalled set", which is exactly what associative recall after
//! pattern completion should produce.

use std::collections::HashMap;

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct EngramDictionary {
    entries: HashMap<String, Vec<u32>>,
}

impl EngramDictionary {
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn contains(&self, word: &str) -> bool {
        self.entries.contains_key(word)
    }

    pub fn get(&self, word: &str) -> Option<&[u32]> {
        self.entries.get(word).map(|v| v.as_slice())
    }

    pub fn words(&self) -> impl Iterator<Item = &str> {
        self.entries.keys().map(|s| s.as_str())
    }

    /// Store the R2 spike pattern observed for `word`. The indices are
    /// sorted and deduplicated so later overlap queries can run in
    /// linear-merge time.
    pub fn learn_concept(&mut self, word: &str, active_r2_indices: &[u32]) {
        let mut indices = active_r2_indices.to_vec();
        indices.sort_unstable();
        indices.dedup();
        self.entries.insert(word.to_string(), indices);
    }

    /// Return the `k` engrams with the highest containment score
    /// against `active_r2_indices`, sorted descending. Useful when the
    /// right cut-off varies per query — top-k removes the threshold
    /// guesswork. Empty engrams are skipped. Ties are broken
    /// alphabetically so the result is deterministic.
    pub fn decode_top(&self, active_r2_indices: &[u32], k: usize) -> Vec<(String, f32)> {
        let mut active = active_r2_indices.to_vec();
        active.sort_unstable();
        active.dedup();

        let mut scored: Vec<(String, f32)> = self
            .entries
            .iter()
            .filter(|(_, stored)| !stored.is_empty())
            .map(|(word, stored)| {
                let overlap = sorted_overlap(stored, &active);
                let ratio = overlap as f32 / stored.len() as f32;
                (word.clone(), ratio)
            })
            .collect();
        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        scored.truncate(k);
        scored
    }

    /// Score every stored engram against `active_r2_indices` and return
    /// the words whose containment ratio (|active ∩ stored| / |stored|)
    /// meets `min_overlap_ratio`, sorted from highest score to lowest.
    /// Empty engrams are skipped.
    pub fn decode(&self, active_r2_indices: &[u32], min_overlap_ratio: f32) -> Vec<(String, f32)> {
        let mut active = active_r2_indices.to_vec();
        active.sort_unstable();
        active.dedup();

        let mut results: Vec<(String, f32)> = Vec::new();
        for (word, stored) in &self.entries {
            if stored.is_empty() {
                continue;
            }
            let overlap = sorted_overlap(stored, &active);
            let ratio = overlap as f32 / stored.len() as f32;
            if ratio >= min_overlap_ratio {
                results.push((word.clone(), ratio));
            }
        }
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }
}

/// Linear-merge intersection size for two sorted-unique slices.
fn sorted_overlap(a: &[u32], b: &[u32]) -> usize {
    let (mut i, mut j, mut count) = (0usize, 0usize, 0usize);
    while i < a.len() && j < b.len() {
        if a[i] == b[j] {
            count += 1;
            i += 1;
            j += 1;
        } else if a[i] < b[j] {
            i += 1;
        } else {
            j += 1;
        }
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_dictionary_decodes_to_empty() {
        let d = EngramDictionary::new();
        assert!(d.decode(&[1, 2, 3], 0.5).is_empty());
    }

    #[test]
    fn perfect_match_scores_one() {
        let mut d = EngramDictionary::new();
        d.learn_concept("rust", &[3, 1, 2]);
        let out = d.decode(&[1, 2, 3], 0.5);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].0, "rust");
        assert!((out[0].1 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn half_overlap_scores_half() {
        let mut d = EngramDictionary::new();
        d.learn_concept("hello", &[1, 2, 3, 4]);
        let out = d.decode(&[1, 2, 99, 100], 0.4);
        assert_eq!(out.len(), 1);
        assert!((out[0].1 - 0.5).abs() < 1e-6);
    }

    #[test]
    fn ratio_filter_drops_weak_matches() {
        let mut d = EngramDictionary::new();
        d.learn_concept("strong", &[1, 2, 3, 4, 5]);
        d.learn_concept("weak", &[10, 11, 12, 13, 14, 15, 16, 17, 18, 19]);

        let active = vec![1, 2, 3, 4, 10];
        let out = d.decode(&active, 0.5);
        // strong: 4/5 = 0.8 → in. weak: 1/10 = 0.1 → out.
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].0, "strong");
    }

    #[test]
    fn results_are_sorted_descending() {
        let mut d = EngramDictionary::new();
        d.learn_concept("a", &[1, 2, 3, 4]); // 4 bits
        d.learn_concept("b", &[1, 2]); // 2 bits
        d.learn_concept("c", &[1, 2, 3]); // 3 bits

        let active = vec![1, 2, 3, 4];
        let out = d.decode(&active, 0.0);
        assert_eq!(out.len(), 3);
        // a: 4/4=1.0   c: 3/3=1.0   b: 2/2=1.0  — all perfect; only need
        // a stable order. Build something with distinguishable scores:
        let mut d = EngramDictionary::new();
        d.learn_concept("a", &[1, 2, 3, 4]); // 4/4 = 1.0
        d.learn_concept("b", &[1, 2, 3, 99]); // 3/4 = 0.75
        d.learn_concept("c", &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]); // 4/10 = 0.4
        let out = d.decode(&active, 0.0);
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].0, "a");
        assert_eq!(out[1].0, "b");
        assert_eq!(out[2].0, "c");
        assert!(out[0].1 > out[1].1 && out[1].1 > out[2].1);
    }

    #[test]
    fn duplicate_indices_in_input_are_handled() {
        let mut d = EngramDictionary::new();
        d.learn_concept("rust", &[1, 2, 3]);
        let out = d.decode(&[1, 1, 1, 2, 2, 3, 3, 3], 0.5);
        assert_eq!(out.len(), 1);
        assert!((out[0].1 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn decode_top_returns_k_best_results() {
        let mut d = EngramDictionary::new();
        d.learn_concept("strong", &[1, 2, 3, 4]); // 4/4 = 1.00
        d.learn_concept("medium", &[1, 2, 3, 99]); // 3/4 = 0.75
        d.learn_concept("weak", &[1, 2, 3, 4, 5, 6, 7, 8]); // 4/8 = 0.50

        let active = vec![1, 2, 3, 4];
        let top1 = d.decode_top(&active, 1);
        assert_eq!(top1.len(), 1);
        assert_eq!(top1[0].0, "strong");

        let top2 = d.decode_top(&active, 2);
        assert_eq!(top2.len(), 2);
        assert_eq!(top2[0].0, "strong");
        assert_eq!(top2[1].0, "medium");

        let top10 = d.decode_top(&active, 10);
        assert_eq!(top10.len(), 3); // capped by dictionary size
        assert!(top10[0].1 >= top10[1].1 && top10[1].1 >= top10[2].1);
    }

    #[test]
    fn decode_top_breaks_ties_alphabetically() {
        let mut d = EngramDictionary::new();
        d.learn_concept("zebra", &[1, 2, 3]);
        d.learn_concept("alpha", &[1, 2, 3]);

        let top = d.decode_top(&[1, 2, 3], 2);
        assert_eq!(top[0].0, "alpha");
        assert_eq!(top[1].0, "zebra");
    }

    #[test]
    fn relearning_overwrites() {
        let mut d = EngramDictionary::new();
        d.learn_concept("rust", &[1, 2, 3]);
        d.learn_concept("rust", &[10, 20, 30]);
        assert_eq!(d.get("rust"), Some(&[10, 20, 30][..]));
    }
}
