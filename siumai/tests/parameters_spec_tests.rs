//! Aggregator for spec-aligned parameter mapping tests under tests/parameters/.
//! This ensures subdirectory tests are picked up by cargo as a single test target.

#[path = "parameters/responses_message_mapping_test.rs"]
mod responses_message_mapping_test;

#[path = "parameters/rerank_openai_compatible_transform_test.rs"]
mod rerank_openai_compatible_transform_test;

#[path = "parameters/moderation_array_input_shape_test.rs"]
mod moderation_array_input_shape_test;
