use eventsource_stream::Event;
use siumai_core::execution::streaming::ChatStreamEventCore;
use siumai_std_openai::openai::responses::OpenAiResponsesStandard;

#[test]
fn openai_responses_stream_converter_handles_text_and_completed() {
    let standard = OpenAiResponsesStandard::new();
    let converter = standard.create_stream_converter("openai_responses");

    let events = vec![
        Event {
            event: "response.output_text.delta".to_string(),
            data: r#"{"type":"response.output_text.delta","delta":"Hello ","output_index":0}"#
                .to_string(),
            id: "1".to_string(),
            retry: None,
        },
        Event {
            event: "response.output_text.delta".to_string(),
            data: r#"{"type":"response.output_text.delta","delta":"world","output_index":0}"#
                .to_string(),
            id: "2".to_string(),
            retry: None,
        },
        Event {
            event: "response.completed".to_string(),
            data: r#"{"type":"response.completed"}"#.to_string(),
            id: "3".to_string(),
            retry: None,
        },
    ];

    let mut core_events = Vec::new();
    for ev in events {
        let converted = converter.convert_event(ev);
        for item in converted {
            core_events.push(item.expect("event should be ok"));
        }
    }

    // Expect StreamStart + two content deltas + StreamEnd
    assert!(
        core_events.len() >= 4,
        "expected at least 4 events, got {}",
        core_events.len()
    );

    // First event: StreamStart
    matches!(core_events[0], ChatStreamEventCore::StreamStart {});

    // Collect concatenated content
    let mut text = String::new();
    for ev in &core_events {
        if let ChatStreamEventCore::ContentDelta { delta, .. } = ev {
            text.push_str(delta);
        }
    }
    assert_eq!(text, "Hello world");

    // Last event should be StreamEnd
    assert!(matches!(
        core_events.last().unwrap(),
        ChatStreamEventCore::StreamEnd { .. }
    ));
}
