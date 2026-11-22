//! Benchmark harness for rslu: dataset parsers, metrics, evaluation profiles,
//! the corpus registry, a downloader and the CI regression gate. The
//! `rslu-bench` binary is a thin command line over these modules.

pub mod check;
pub mod dataset;
pub mod eval;
pub mod fetch;
pub mod metrics;
pub mod sets;
