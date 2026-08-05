//! Thin server projections over Siumai's provider-neutral runtime.
//!
//! This crate does not own a tool loop. Plain routes perform one model call;
//! local execution exists only after the host installs an explicit trusted
//! tool route backed by [`siumai_runtime::ToolLoop`].

#![deny(unsafe_code)]

#[cfg(feature = "axum")]
pub mod axum;

mod event;
mod gateway;

pub use event::GatewayEvent;
pub use gateway::{ServerGateway, ServerGatewayError};
