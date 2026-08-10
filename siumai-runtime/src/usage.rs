use siumai_core::{Usage, UsageUpdate, UsageUpdateKind, UsageValue};

/// Reconciles every usage observation produced by one provider call.
///
/// Snapshots are cumulative observations, so repeated or regressing values do
/// not add to the call total. Explicit deltas advance the current observation
/// exactly once. Terminal usage is treated as one final cumulative snapshot.
#[derive(Default)]
pub(crate) struct CallUsageReconciler {
    current: Option<Usage>,
}

impl CallUsageReconciler {
    pub(crate) fn observe(&mut self, update: &UsageUpdate) {
        match update.kind() {
            UsageUpdateKind::Snapshot => self.observe_snapshot(update.usage()),
            UsageUpdateKind::Delta => self.observe_delta(update.usage()),
            _ => {}
        }
    }

    pub(crate) fn settle(mut self, terminal: Option<&Usage>) -> Usage {
        if let Some(terminal) = terminal {
            self.observe_snapshot(terminal);
        }
        self.current.unwrap_or_default()
    }

    fn observe_snapshot(&mut self, snapshot: &Usage) {
        match &mut self.current {
            Some(current) => reconcile_snapshot(current, snapshot),
            None => self.current = Some(snapshot.clone()),
        }
    }

    fn observe_delta(&mut self, delta: &Usage) {
        let current = self.current.get_or_insert_with(Usage::default);
        merge_delta(current, delta);
    }
}

fn merge_delta(current: &mut Usage, delta: &Usage) {
    current.input_tokens = add_delta(current.input_tokens, delta.input_tokens);
    current.output_tokens = add_delta(current.output_tokens, delta.output_tokens);
    current.total_tokens = add_delta(current.total_tokens, delta.total_tokens);
    current.reasoning_tokens = add_delta(current.reasoning_tokens, delta.reasoning_tokens);
    current.cache_read_tokens = add_delta(current.cache_read_tokens, delta.cache_read_tokens);
    current.cache_write_tokens = add_delta(current.cache_write_tokens, delta.cache_write_tokens);
    current.audio_input_tokens = add_delta(current.audio_input_tokens, delta.audio_input_tokens);
    current.audio_output_tokens = add_delta(current.audio_output_tokens, delta.audio_output_tokens);
    current.orchestration_tokens =
        add_delta(current.orchestration_tokens, delta.orchestration_tokens);
    current.provider.extend(delta.provider.clone());
}

fn add_delta(current: UsageValue, delta: UsageValue) -> UsageValue {
    match delta {
        UsageValue::Unknown => current,
        UsageValue::Known(delta) => match current {
            UsageValue::Unknown => UsageValue::Known(delta),
            UsageValue::Known(current) => current
                .checked_add(delta)
                .map_or(UsageValue::Unknown, UsageValue::Known),
        },
    }
}

fn reconcile_snapshot(current: &mut Usage, snapshot: &Usage) {
    current.input_tokens = reconcile_value(current.input_tokens, snapshot.input_tokens);
    current.output_tokens = reconcile_value(current.output_tokens, snapshot.output_tokens);
    current.total_tokens = reconcile_value(current.total_tokens, snapshot.total_tokens);
    current.reasoning_tokens = reconcile_value(current.reasoning_tokens, snapshot.reasoning_tokens);
    current.cache_read_tokens =
        reconcile_value(current.cache_read_tokens, snapshot.cache_read_tokens);
    current.cache_write_tokens =
        reconcile_value(current.cache_write_tokens, snapshot.cache_write_tokens);
    current.audio_input_tokens =
        reconcile_value(current.audio_input_tokens, snapshot.audio_input_tokens);
    current.audio_output_tokens =
        reconcile_value(current.audio_output_tokens, snapshot.audio_output_tokens);
    current.orchestration_tokens =
        reconcile_value(current.orchestration_tokens, snapshot.orchestration_tokens);
    if !snapshot.provider.is_empty() {
        current.provider.clone_from(&snapshot.provider);
    }
}

fn reconcile_value(current: UsageValue, snapshot: UsageValue) -> UsageValue {
    match (current, snapshot) {
        (UsageValue::Known(current), UsageValue::Known(snapshot)) => {
            UsageValue::Known(current.max(snapshot))
        }
        (UsageValue::Unknown, UsageValue::Known(snapshot)) => UsageValue::Known(snapshot),
        (UsageValue::Known(current), UsageValue::Unknown) => UsageValue::Known(current),
        (UsageValue::Unknown, UsageValue::Unknown) => UsageValue::Unknown,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Case {
        name: &'static str,
        updates: Vec<UsageUpdate>,
        terminal: Option<Usage>,
        expected_total: UsageValue,
        expected_input: UsageValue,
    }

    #[test]
    fn reconciles_one_provider_call_without_double_counting() {
        let cases = [
            Case {
                name: "repeated cumulative snapshots",
                updates: vec![
                    UsageUpdate::snapshot(usage(Some(2), Some(4))),
                    UsageUpdate::snapshot(usage(Some(2), Some(4))),
                    UsageUpdate::snapshot(usage(Some(3), Some(7))),
                ],
                terminal: Some(usage(Some(3), Some(7))),
                expected_total: UsageValue::Known(7),
                expected_input: UsageValue::Known(3),
            },
            Case {
                name: "explicit delta applied once",
                updates: vec![
                    UsageUpdate::snapshot(usage(Some(2), Some(4))),
                    UsageUpdate::delta(usage(Some(1), Some(2))),
                ],
                terminal: Some(usage(Some(3), Some(6))),
                expected_total: UsageValue::Known(6),
                expected_input: UsageValue::Known(3),
            },
            Case {
                name: "delta omission preserves the known dimension",
                updates: vec![
                    UsageUpdate::snapshot(usage(Some(2), Some(4))),
                    UsageUpdate::delta(usage(None, Some(2))),
                ],
                terminal: None,
                expected_total: UsageValue::Known(6),
                expected_input: UsageValue::Known(2),
            },
            Case {
                name: "terminal snapshot advances observation",
                updates: vec![UsageUpdate::snapshot(usage(Some(2), Some(4)))],
                terminal: Some(usage(Some(3), Some(6))),
                expected_total: UsageValue::Known(6),
                expected_input: UsageValue::Known(3),
            },
            Case {
                name: "terminal snapshot does not regress observation",
                updates: vec![UsageUpdate::snapshot(usage(Some(3), Some(6)))],
                terminal: Some(usage(None, Some(4))),
                expected_total: UsageValue::Known(6),
                expected_input: UsageValue::Known(3),
            },
            Case {
                name: "delta-only terminal snapshot is not duplicated",
                updates: vec![UsageUpdate::delta(usage(Some(1), Some(2)))],
                terminal: Some(usage(Some(1), Some(2))),
                expected_total: UsageValue::Known(2),
                expected_input: UsageValue::Known(1),
            },
            Case {
                name: "missing dimensions remain unknown",
                updates: vec![UsageUpdate::snapshot(usage(None, Some(4)))],
                terminal: None,
                expected_total: UsageValue::Known(4),
                expected_input: UsageValue::Unknown,
            },
        ];

        for case in cases {
            let mut reconciler = CallUsageReconciler::default();
            for update in &case.updates {
                reconciler.observe(update);
            }
            let settled = reconciler.settle(case.terminal.as_ref());

            assert_eq!(settled.total_tokens, case.expected_total, "{}", case.name);
            assert_eq!(settled.input_tokens, case.expected_input, "{}", case.name);
        }
    }

    fn usage(input: Option<u64>, total: Option<u64>) -> Usage {
        Usage::default()
            .with_input_tokens(input)
            .with_total_tokens(total)
    }
}
