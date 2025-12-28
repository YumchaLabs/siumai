//! Protocol standards owned by this crate.
#![deny(unsafe_code)]

#[cfg(any(
    feature = "openai-standard",
    feature = "openai",
    feature = "groq",
    feature = "xai",
    feature = "minimaxi"
))]
pub mod openai;
