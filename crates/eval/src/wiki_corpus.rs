//! A small "Wikipedia-shaped" corpus.
//!
//! Five short paragraphs distilled from real Wikipedia articles
//! (CC BY-SA 4.0 — content remains under that license). Each topic is
//! intentionally distant from the others — geology, transport,
//! beverages, biology, architecture — so the brain can be measured on
//! whether it keeps engrams separable when training across domains.
//!
//! Snippets are kept short (~25–35 content words after stop-word
//! filtering) so the default Brain (R1=1000 / R2=2000) has enough
//! sparsity left for clean engrams.

/// Returns `&'static [&'static str]` so callers can pass it straight
/// into `run_javis_pipeline_on`.
pub fn wiki_corpus() -> &'static [&'static str] {
    &[
        // Volcano
        "A volcano is a rupture in the crust of a planet that allows \
         hot lava and gases to escape from a magma chamber below the \
         surface. Most volcanoes form where tectonic plates diverge or \
         converge. Eruptions can be explosive or effusive depending on \
         the composition of the magma.",
        // Bicycle
        "A bicycle is a human powered vehicle with two wheels attached \
         to a frame. The rider pushes pedals connected by a chain that \
         drives the rear wheel. Modern bicycles use lightweight \
         aluminium frames and pneumatic rubber tires. Cycling provides \
         efficient transport and aerobic exercise.",
        // Coffee
        "Coffee is a brewed beverage prepared from roasted coffee beans, \
         the seeds of berries from the Coffea plant. Caffeine in coffee \
         acts as a stimulant on the central nervous system. Brazil and \
         Vietnam produce most of the world's coffee. Espresso uses high \
         pressure to extract a concentrated shot.",
        // Photosynthesis
        "Photosynthesis is a biological process used by plants and algae \
         to convert light energy into chemical energy stored in glucose. \
         Chlorophyll inside chloroplasts absorbs sunlight to split water \
         and release oxygen. Carbon dioxide from the air combines with \
         hydrogen to form sugars.",
        // Eiffel Tower
        "The Eiffel Tower is a wrought iron lattice tower in Paris \
         France. It was built by Gustave Eiffel for the 1889 World Fair. \
         At 330 meters tall it was the tallest structure for forty \
         years. Today it is the most visited paid monument in the world.",
    ]
}

/// Topics paired with the keyword most likely to recall the
/// corresponding paragraph. Used by the benchmark to drive a query per
/// topic.
pub fn wiki_queries() -> &'static [(&'static str, &'static str)] {
    &[
        ("volcano", "volcano"),
        ("bicycle", "bicycle"),
        ("coffee", "coffee"),
        ("photosynthesis", "photosynthesis"),
        ("eiffel", "eiffel"),
    ]
}
