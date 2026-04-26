//! AI SDK-style nullish/optional value helpers.

use std::collections::BTreeMap;

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
