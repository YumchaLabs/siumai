//! AI SDK-style nullish/optional value helpers.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// Value that can be provided as no value, one value, or many values.
///
/// This is the Rust data carrier for AI SDK `Arrayable<T>`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(untagged)]
pub enum Arrayable<T> {
    /// Multiple values.
    Array(Vec<T>),
    /// A single value.
    Single(T),
    /// No value, equivalent to JavaScript `undefined` in `asArray`.
    None,
}

impl<T> Default for Arrayable<T> {
    fn default() -> Self {
        Self::None
    }
}

impl<T> Arrayable<T> {
    /// Create an empty `Arrayable`.
    pub const fn none() -> Self {
        Self::None
    }

    /// Create a single-value `Arrayable`.
    pub fn single(value: T) -> Self {
        Self::Single(value)
    }

    /// Create a multi-value `Arrayable`.
    pub fn array(values: impl Into<Vec<T>>) -> Self {
        Self::Array(values.into())
    }

    /// Normalize into a vector.
    pub fn into_vec(self) -> Vec<T> {
        match self {
            Self::None => Vec::new(),
            Self::Single(value) => vec![value],
            Self::Array(values) => values,
        }
    }
}

impl<T> From<Vec<T>> for Arrayable<T> {
    fn from(value: Vec<T>) -> Self {
        Self::Array(value)
    }
}

impl<T: Clone> From<&[T]> for Arrayable<T> {
    fn from(value: &[T]) -> Self {
        Self::Array(value.to_vec())
    }
}

impl<T> From<Option<T>> for Arrayable<T> {
    fn from(value: Option<T>) -> Self {
        match value {
            Some(value) => Self::Single(value),
            None => Self::None,
        }
    }
}

impl<T> From<Option<Vec<T>>> for Arrayable<T> {
    fn from(value: Option<Vec<T>>) -> Self {
        match value {
            Some(value) => Self::Array(value),
            None => Self::None,
        }
    }
}

/// Normalize a missing, single, or multi-value input into a vector.
///
/// This mirrors AI SDK `asArray`, with [`Arrayable::None`] representing
/// JavaScript `undefined`.
pub fn as_array<T>(value: Arrayable<T>) -> Vec<T> {
    value.into_vec()
}

/// Return whether an optional value is present.
///
/// This is the Rust `Option` equivalent of AI SDK `isNonNullable`.
pub fn is_non_nullable<T>(value: &Option<T>) -> bool {
    value.is_some()
}

/// Remove missing optional values from an iterator.
///
/// This is the Rust `Option` equivalent of AI SDK `filterNullable`.
pub fn filter_nullable<T>(values: impl IntoIterator<Item = Option<T>>) -> Vec<T> {
    values.into_iter().flatten().collect()
}

/// Remove entries whose values are `None`.
///
/// This mirrors AI SDK `removeUndefinedEntries`, with Rust `None` representing
/// both JavaScript `undefined` and `null` in typed option maps.
pub fn remove_undefined_entries<K, V>(
    record: impl IntoIterator<Item = (K, Option<V>)>,
) -> BTreeMap<K, V>
where
    K: Ord,
{
    record
        .into_iter()
        .filter_map(|(key, value)| value.map(|value| (key, value)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_arrayable_values() {
        assert!(as_array::<&str>(Arrayable::none()).is_empty());
        assert_eq!(as_array(Arrayable::single("a")), vec!["a"]);
        assert_eq!(as_array(Arrayable::array(["a", "b"])), vec!["a", "b"]);
        assert_eq!(as_array(Arrayable::from(Some("a"))), vec!["a"]);
        assert_eq!(as_array(Arrayable::from(vec!["a", "b"])), vec!["a", "b"]);
    }

    #[test]
    fn arrayable_deserializes_arrays_before_single_json_values() {
        let parsed: Arrayable<serde_json::Value> =
            serde_json::from_value(serde_json::json!(["a", "b"])).expect("arrayable array");

        assert!(matches!(parsed, Arrayable::Array(values) if values.len() == 2));
    }

    #[test]
    fn detects_present_option_values() {
        assert!(is_non_nullable(&Some("value")));
        assert!(!is_non_nullable::<String>(&None));
    }

    #[test]
    fn filters_missing_values() {
        assert_eq!(
            filter_nullable([Some("a"), None, Some("b")]),
            vec!["a", "b"]
        );
    }

    #[test]
    fn removes_undefined_entries() {
        let filtered = remove_undefined_entries([
            ("keep", Some("yes")),
            ("drop", None),
            ("also_keep", Some("ok")),
        ]);

        assert_eq!(filtered.get("keep"), Some(&"yes"));
        assert_eq!(filtered.get("also_keep"), Some(&"ok"));
        assert!(!filtered.contains_key("drop"));
    }
}
