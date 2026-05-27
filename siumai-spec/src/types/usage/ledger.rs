use super::Usage;

/// Tracks cumulative usage snapshots for one provider/model call.
///
/// Streaming providers can report usage more than once for the same call. Those values are
/// cumulative snapshots, so the latest snapshot replaces the previous one. Explicit multi-call or
/// multi-step aggregation remains owned by `Usage::merge()`.
#[derive(Debug, Clone, Default)]
pub struct UsageSnapshotLedger {
    latest: Option<Usage>,
}

impl UsageSnapshotLedger {
    /// Create an empty snapshot ledger.
    pub const fn new() -> Self {
        Self { latest: None }
    }

    /// Create a ledger seeded with an initial snapshot.
    pub fn with_snapshot(snapshot: Usage) -> Self {
        Self {
            latest: Some(snapshot),
        }
    }

    /// Returns true when no provider-call snapshot has been recorded.
    pub const fn is_empty(&self) -> bool {
        self.latest.is_none()
    }

    /// Record the latest cumulative snapshot, replacing any earlier snapshot for the same call.
    pub fn record_snapshot(&mut self, snapshot: Usage) {
        self.latest = Some(snapshot);
    }

    /// Borrow the latest cumulative snapshot.
    pub fn latest(&self) -> Option<&Usage> {
        self.latest.as_ref()
    }

    /// Clone the latest cumulative snapshot.
    pub fn latest_cloned(&self) -> Option<Usage> {
        self.latest.clone()
    }

    /// Return the latest snapshot, or a fallback value when the ledger is empty.
    pub fn latest_or_else(&self, fallback: Option<&Usage>) -> Option<Usage> {
        self.latest_cloned().or_else(|| fallback.cloned())
    }

    /// Consume the ledger and return the latest cumulative snapshot.
    pub fn into_latest(self) -> Option<Usage> {
        self.latest
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ledger_replaces_repeated_snapshots_without_aggregating() {
        let mut ledger = UsageSnapshotLedger::new();
        ledger.record_snapshot(
            Usage::builder()
                .prompt_tokens(10)
                .completion_tokens(4)
                .total_tokens(14)
                .with_raw_usage_value(serde_json::json!({
                    "snapshot": "first",
                    "total_tokens": 14
                }))
                .build(),
        );
        ledger.record_snapshot(
            Usage::builder()
                .prompt_tokens(12)
                .completion_tokens(5)
                .total_tokens(17)
                .with_raw_usage_value(serde_json::json!({
                    "snapshot": "second",
                    "total_tokens": 17
                }))
                .build(),
        );

        let usage = ledger.latest().expect("latest usage snapshot");
        assert_eq!(usage.prompt_tokens(), Some(12));
        assert_eq!(usage.completion_tokens(), Some(5));
        assert_eq!(usage.total_tokens(), Some(17));
        assert_eq!(
            usage.raw_usage_value(),
            Some(serde_json::json!({
                "snapshot": "second",
                "total_tokens": 17
            }))
        );
    }

    #[test]
    fn ledger_uses_fallback_only_when_no_snapshot_was_recorded() {
        let fallback = Usage::new(1, 2);
        let mut ledger = UsageSnapshotLedger::new();
        assert_eq!(
            ledger
                .latest_or_else(Some(&fallback))
                .and_then(|usage| usage.total_tokens()),
            Some(3)
        );

        ledger.record_snapshot(Usage::new(5, 8));

        assert_eq!(
            ledger
                .latest_or_else(Some(&fallback))
                .and_then(|usage| usage.total_tokens()),
            Some(13)
        );
    }
}
