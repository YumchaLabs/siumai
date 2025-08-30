//! Thinking Content Analysis
//!
//! This module provides tools for analyzing thinking content from AI models,
//! inspired by Cherry Studio's thinking analysis capabilities.

use serde::{Deserialize, Serialize};

/// Analysis results for thinking content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThinkingAnalysis {
    /// Total length of thinking content in characters
    pub length: usize,

    /// Number of words in thinking content
    pub word_count: usize,

    /// Number of reasoning steps identified
    pub reasoning_steps: usize,

    /// Number of questions asked during thinking
    pub questions_count: usize,

    /// Complexity score (0.0 to 1.0)
    pub complexity_score: f64,

    /// Reasoning patterns found
    pub patterns: Vec<ReasoningPattern>,

    /// Time spent thinking (if available)
    pub thinking_time_ms: Option<u64>,

    /// Key insights extracted
    pub insights: Vec<String>,
}

/// Types of reasoning patterns that can be identified
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningPattern {
    /// Type of pattern
    pub pattern_type: PatternType,

    /// Number of occurrences
    pub count: usize,

    /// Confidence score (0.0 to 1.0)
    pub confidence: f64,

    /// Example text where pattern was found
    pub example: Option<String>,
}

/// Different types of reasoning patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PatternType {
    /// Sequential reasoning (step 1, step 2, etc.)
    Sequential,

    /// Causal reasoning (because, therefore, so)
    Causal,

    /// Comparative reasoning (better than, worse than, similar to)
    Comparative,

    /// Hypothetical reasoning (if, suppose, assume)
    Hypothetical,

    /// Analytical reasoning (analyze, examine, consider)
    Analytical,

    /// Problem-solving (solution, approach, method)
    ProblemSolving,

    /// Self-correction (actually, wait, let me reconsider)
    SelfCorrection,

    /// Uncertainty (maybe, perhaps, might be)
    Uncertainty,
}

/// Analyze thinking content and extract insights
pub fn analyze_thinking_content(content: &str) -> ThinkingAnalysis {
    let length = content.len();
    let word_count = content.split_whitespace().count();

    // Identify reasoning patterns
    let patterns = identify_reasoning_patterns(content);

    // Count reasoning steps
    let reasoning_steps = count_reasoning_steps(content);

    // Count questions
    let questions_count = content.matches('?').count();

    // Calculate complexity score
    let complexity_score = calculate_complexity_score(content, &patterns);

    // Extract key insights
    let insights = extract_insights(content);

    ThinkingAnalysis {
        length,
        word_count,
        reasoning_steps,
        questions_count,
        complexity_score,
        patterns,
        thinking_time_ms: None, // Would need to be provided externally
        insights,
    }
}

/// Identify different types of reasoning patterns in the text
fn identify_reasoning_patterns(content: &str) -> Vec<ReasoningPattern> {
    let mut patterns = Vec::new();
    let content_lower = content.to_lowercase();

    // Sequential reasoning patterns
    let sequential_indicators = [
        "step 1",
        "step 2",
        "step 3",
        "first",
        "second",
        "third",
        "then",
        "next",
        "finally",
        "initially",
        "subsequently",
        "lastly",
    ];
    let sequential_count = count_pattern_occurrences(&content_lower, &sequential_indicators);
    if sequential_count > 0 {
        patterns.push(ReasoningPattern {
            pattern_type: PatternType::Sequential,
            count: sequential_count,
            confidence: calculate_pattern_confidence(sequential_count, content.len()),
            example: find_pattern_example(content, &sequential_indicators),
        });
    }

    // Causal reasoning patterns
    let causal_indicators = [
        "because",
        "therefore",
        "thus",
        "hence",
        "so",
        "as a result",
        "consequently",
        "due to",
        "since",
        "given that",
    ];
    let causal_count = count_pattern_occurrences(&content_lower, &causal_indicators);
    if causal_count > 0 {
        patterns.push(ReasoningPattern {
            pattern_type: PatternType::Causal,
            count: causal_count,
            confidence: calculate_pattern_confidence(causal_count, content.len()),
            example: find_pattern_example(content, &causal_indicators),
        });
    }

    // Hypothetical reasoning patterns
    let hypothetical_indicators = [
        "if",
        "suppose",
        "assume",
        "what if",
        "imagine",
        "consider",
        "let's say",
        "hypothetically",
        "in case",
    ];
    let hypothetical_count = count_pattern_occurrences(&content_lower, &hypothetical_indicators);
    if hypothetical_count > 0 {
        patterns.push(ReasoningPattern {
            pattern_type: PatternType::Hypothetical,
            count: hypothetical_count,
            confidence: calculate_pattern_confidence(hypothetical_count, content.len()),
            example: find_pattern_example(content, &hypothetical_indicators),
        });
    }

    // Self-correction patterns
    let correction_indicators = [
        "actually",
        "wait",
        "let me reconsider",
        "on second thought",
        "correction",
        "i mean",
        "rather",
        "instead",
        "my mistake",
    ];
    let correction_count = count_pattern_occurrences(&content_lower, &correction_indicators);
    if correction_count > 0 {
        patterns.push(ReasoningPattern {
            pattern_type: PatternType::SelfCorrection,
            count: correction_count,
            confidence: calculate_pattern_confidence(correction_count, content.len()),
            example: find_pattern_example(content, &correction_indicators),
        });
    }

    // Problem-solving patterns
    let problem_solving_indicators = [
        "solution", "approach", "method", "strategy", "way to", "how to", "solve", "resolve",
        "address", "tackle",
    ];
    let problem_solving_count =
        count_pattern_occurrences(&content_lower, &problem_solving_indicators);
    if problem_solving_count > 0 {
        patterns.push(ReasoningPattern {
            pattern_type: PatternType::ProblemSolving,
            count: problem_solving_count,
            confidence: calculate_pattern_confidence(problem_solving_count, content.len()),
            example: find_pattern_example(content, &problem_solving_indicators),
        });
    }

    patterns
}

/// Count occurrences of pattern indicators
fn count_pattern_occurrences(content: &str, indicators: &[&str]) -> usize {
    indicators
        .iter()
        .map(|&indicator| content.matches(indicator).count())
        .sum()
}

/// Calculate confidence score for a pattern based on frequency and content length
fn calculate_pattern_confidence(count: usize, content_length: usize) -> f64 {
    let frequency = count as f64 / (content_length as f64 / 100.0); // per 100 characters
    (frequency * 10.0).min(1.0) // Cap at 1.0
}

/// Find an example of where a pattern occurs in the text
fn find_pattern_example(content: &str, indicators: &[&str]) -> Option<String> {
    for &indicator in indicators {
        if let Some(pos) = content.to_lowercase().find(indicator) {
            let start = pos.saturating_sub(20);
            let end = (pos + indicator.len() + 30).min(content.len());
            return Some(content[start..end].to_string());
        }
    }
    None
}

/// Count reasoning steps in the content
fn count_reasoning_steps(content: &str) -> usize {
    let step_patterns = [
        r"step \d+",
        r"^\d+\.",
        r"first",
        r"second",
        r"third",
        r"then",
        r"next",
        r"finally",
    ];

    // Simple counting - in a real implementation, you might use regex
    let content_lower = content.to_lowercase();
    step_patterns
        .iter()
        .map(|_pattern| {
            // Simplified counting for now
            content_lower.matches("step").count()
                + content_lower.matches("first").count()
                + content_lower.matches("then").count()
                + content_lower.matches("next").count()
        })
        .sum::<usize>()
        .min(20) // Cap at reasonable number
}

/// Calculate overall complexity score
fn calculate_complexity_score(content: &str, patterns: &[ReasoningPattern]) -> f64 {
    let length_factor = (content.len() as f64 / 1000.0).min(1.0); // Longer = more complex
    let pattern_factor = (patterns.len() as f64 / 5.0).min(1.0); // More patterns = more complex
    let question_factor = (content.matches('?').count() as f64 / 10.0).min(1.0); // More questions = more complex

    (length_factor + pattern_factor + question_factor) / 3.0
}

/// Extract key insights from thinking content
fn extract_insights(content: &str) -> Vec<String> {
    let mut insights = Vec::new();

    // Look for conclusion indicators
    let conclusion_patterns = [
        "in conclusion",
        "therefore",
        "the answer is",
        "the solution is",
        "this means",
        "we can conclude",
        "the result is",
    ];

    for pattern in conclusion_patterns.iter() {
        if let Some(pos) = content.to_lowercase().find(pattern) {
            let start = pos;
            let end = content[pos..]
                .find('.')
                .map(|i| pos + i + 1)
                .unwrap_or(content.len());
            if end > start {
                insights.push(content[start..end].trim().to_string());
            }
        }
    }

    // Limit to most relevant insights
    insights.truncate(3);
    insights
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_analysis() {
        let content = "First, I need to understand the problem. Then, I'll consider different approaches. Finally, I'll choose the best solution.";
        let analysis = analyze_thinking_content(content);

        assert!(analysis.reasoning_steps > 0);
        assert!(analysis.complexity_score > 0.0);
        assert!(!analysis.patterns.is_empty());
    }

    #[test]
    fn test_pattern_identification() {
        let content = "Because of this, therefore we can conclude that the answer is correct.";
        let analysis = analyze_thinking_content(content);

        let causal_patterns: Vec<_> = analysis
            .patterns
            .iter()
            .filter(|p| matches!(p.pattern_type, PatternType::Causal))
            .collect();

        assert!(!causal_patterns.is_empty());
    }

    #[test]
    fn test_complexity_scoring() {
        let simple_content = "The answer is 42.";
        let complex_content = "First, let me analyze this step by step. If we consider the various factors, then we need to evaluate each option. Actually, wait - let me reconsider this approach. What if we try a different method?";

        let simple_analysis = analyze_thinking_content(simple_content);
        let complex_analysis = analyze_thinking_content(complex_content);

        assert!(complex_analysis.complexity_score > simple_analysis.complexity_score);
    }
}
