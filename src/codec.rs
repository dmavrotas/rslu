//! The binary model format: a small, versioned, little-endian encoding.
//!
//! Every model file starts with the magic bytes `RSLU`, a `u32` format version
//! and the model kind (for example `"logreg"`). Numbers are little-endian,
//! floats are stored as full `f64` bit patterns (so a loaded model predicts
//! exactly what the saved one did), strings and lists are prefixed with a
//! `u32` length. Decoding never panics and never allocates more than the input
//! could describe, so a corrupt or hostile file is an error, not a crash.

use std::error::Error as StdError;
use std::fmt;
use std::io;

const MAGIC: &[u8; 4] = b"RSLU";
/// The format version this build writes and reads.
pub const VERSION: u32 = 1;

/// Why a model could not be loaded.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum DecodeError {
    /// Reading the input failed.
    Io(io::ErrorKind),
    /// The input does not start with the rslu magic bytes.
    NotAModel,
    /// The input was written by a newer, incompatible format version.
    UnsupportedVersion(u32),
    /// The input holds a different kind of model than requested.
    WrongKind(String),
    /// The input ended early.
    Truncated,
    /// The input is malformed; the message says how.
    Corrupt(&'static str),
}

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(kind) => write!(f, "rslu: reading the model failed: {kind}"),
            Self::NotAModel => f.write_str("rslu: not an rslu model"),
            Self::UnsupportedVersion(v) => write!(
                f,
                "rslu: model format version {v} is not supported (this build reads {VERSION})"
            ),
            Self::WrongKind(kind) => write!(f, "rslu: the input holds a {kind:?} model"),
            Self::Truncated => f.write_str("rslu: the model data ends early"),
            Self::Corrupt(why) => write!(f, "rslu: corrupt model: {why}"),
        }
    }
}

impl StdError for DecodeError {}

impl From<io::Error> for DecodeError {
    fn from(e: io::Error) -> Self {
        Self::Io(e.kind())
    }
}

/// Appends values to a byte buffer.
pub(crate) struct Writer(Vec<u8>);

impl Writer {
    /// Starts a model of `kind` with the header.
    pub(crate) fn new(kind: &str) -> Self {
        let mut w = Self(MAGIC.to_vec());
        w.u32(VERSION);
        w.str(kind);
        w
    }

    pub(crate) fn finish(self) -> Vec<u8> {
        self.0
    }

    pub(crate) fn u8(&mut self, v: u8) {
        self.0.push(v);
    }

    pub(crate) fn u32(&mut self, v: u32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }

    pub(crate) fn u64(&mut self, v: u64) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }

    pub(crate) fn usize(&mut self, v: usize) {
        self.u64(v as u64);
    }

    pub(crate) fn f64(&mut self, v: f64) {
        self.u64(v.to_bits());
    }

    /// A `u32` length prefix. Models never come near 4 billion entries.
    pub(crate) fn len(&mut self, n: usize) {
        self.u32(u32::try_from(n).expect("rslu: model too large to encode"));
    }

    pub(crate) fn str(&mut self, s: &str) {
        self.len(s.len());
        self.0.extend_from_slice(s.as_bytes());
    }

    pub(crate) fn f64s(&mut self, vs: &[f64]) {
        for &v in vs {
            self.f64(v);
        }
    }
}

/// Reads values from a byte slice, checking every bound.
pub(crate) struct Reader<'a>(&'a [u8]);

impl<'a> Reader<'a> {
    /// Checks the header and that the input holds a model of `kind`.
    pub(crate) fn new(bytes: &'a [u8], kind: &str) -> Result<Self, DecodeError> {
        let Some(rest) = bytes.strip_prefix(MAGIC) else {
            return Err(DecodeError::NotAModel);
        };
        let mut r = Self(rest);
        let version = r.u32()?;
        if version != VERSION {
            return Err(DecodeError::UnsupportedVersion(version));
        }
        let found = r.str()?;
        if found != kind {
            return Err(DecodeError::WrongKind(found));
        }
        Ok(r)
    }

    /// Fails unless every byte was consumed.
    pub(crate) fn finish(self) -> Result<(), DecodeError> {
        if self.0.is_empty() {
            Ok(())
        } else {
            Err(DecodeError::Corrupt("trailing bytes"))
        }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], DecodeError> {
        if n > self.0.len() {
            return Err(DecodeError::Truncated);
        }
        let (head, rest) = self.0.split_at(n);
        self.0 = rest;
        Ok(head)
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N], DecodeError> {
        let mut out = [0; N];
        out.copy_from_slice(self.take(N)?);
        Ok(out)
    }

    pub(crate) fn u8(&mut self) -> Result<u8, DecodeError> {
        Ok(self.take(1)?[0])
    }

    pub(crate) fn u32(&mut self) -> Result<u32, DecodeError> {
        Ok(u32::from_le_bytes(self.array()?))
    }

    pub(crate) fn u64(&mut self) -> Result<u64, DecodeError> {
        Ok(u64::from_le_bytes(self.array()?))
    }

    pub(crate) fn usize(&mut self) -> Result<usize, DecodeError> {
        usize::try_from(self.u64()?).map_err(|_| DecodeError::Corrupt("count too large"))
    }

    pub(crate) fn f64(&mut self) -> Result<f64, DecodeError> {
        Ok(f64::from_bits(self.u64()?))
    }

    pub(crate) fn bool(&mut self) -> Result<bool, DecodeError> {
        match self.u8()? {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(DecodeError::Corrupt("invalid flag")),
        }
    }

    /// A length prefix for `n` items of at least `min_item_bytes` each,
    /// rejected up front when the remaining input cannot hold them (so a
    /// corrupt length never triggers a huge allocation).
    pub(crate) fn len(&mut self, min_item_bytes: usize) -> Result<usize, DecodeError> {
        let n = self.u32()? as usize;
        if n.saturating_mul(min_item_bytes) > self.0.len() {
            return Err(DecodeError::Truncated);
        }
        Ok(n)
    }

    pub(crate) fn str(&mut self) -> Result<String, DecodeError> {
        let n = self.len(1)?;
        String::from_utf8(self.take(n)?.to_vec())
            .map_err(|_| DecodeError::Corrupt("string is not UTF-8"))
    }

    /// Exactly `n` floats, with no length prefix (the caller knows `n`).
    pub(crate) fn f64s(&mut self, n: usize) -> Result<Vec<f64>, DecodeError> {
        let bytes = self.take(n.checked_mul(8).ok_or(DecodeError::Truncated)?)?;
        Ok(bytes
            .as_chunks::<8>()
            .0
            .iter()
            .map(|c| f64::from_le_bytes(*c))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn values_round_trip() {
        let mut w = Writer::new("test");
        w.u8(7);
        w.u32(70_000);
        w.usize(1 << 40);
        w.f64(-0.1);
        w.str("café");
        w.f64s(&[1.5, f64::MIN_POSITIVE]);
        let bytes = w.finish();
        let mut r = Reader::new(&bytes, "test").unwrap();
        assert_eq!(r.u8().unwrap(), 7);
        assert_eq!(r.u32().unwrap(), 70_000);
        assert_eq!(r.usize().unwrap(), 1 << 40);
        assert_eq!(r.f64().unwrap().to_bits(), (-0.1f64).to_bits());
        assert_eq!(r.str().unwrap(), "café");
        assert_eq!(r.f64s(2).unwrap(), [1.5, f64::MIN_POSITIVE]);
        r.finish().unwrap();
    }

    #[test]
    fn header_is_checked() {
        assert_eq!(
            Reader::new(b"nope", "x").err(),
            Some(DecodeError::NotAModel)
        );
        let mut future = MAGIC.to_vec();
        future.extend_from_slice(&99u32.to_le_bytes());
        assert_eq!(
            Reader::new(&future, "x").err(),
            Some(DecodeError::UnsupportedVersion(99))
        );
        let other = Writer::new("tagger").finish();
        assert_eq!(
            Reader::new(&other, "logreg").err(),
            Some(DecodeError::WrongKind("tagger".into()))
        );
    }

    #[test]
    fn a_huge_length_is_rejected_before_allocating() {
        let mut w = Writer::new("x");
        w.u32(u32::MAX);
        let bytes = w.finish();
        let mut r = Reader::new(&bytes, "x").unwrap();
        assert_eq!(r.len(8).err(), Some(DecodeError::Truncated));
    }
}
