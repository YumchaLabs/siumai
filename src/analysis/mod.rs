//! Analysis Tools
//!
//! This module provides various analysis tools for AI model responses,
//! including thinking content analysis, performance metrics, and more.

pub mod thinking;

pub use thinking::{PatternType, ReasoningPattern, ThinkingAnalysis, analyze_thinking_content};
