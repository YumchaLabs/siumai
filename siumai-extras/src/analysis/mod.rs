//! Analysis Tools (extras crate)
//!
//! This module hosts various analysis tools for AI model responses,
//! including thinking content analysis.

mod thinking;

pub use thinking::{PatternType, ReasoningPattern, ThinkingAnalysis, analyze_thinking_content};
