//! Test support helpers (crate-internal).
//!
//! This module centralizes process-global test concerns such as environment
//! variable mutation guards. Rust tests run concurrently by default, so any test
//! that mutates env vars should take the shared lock.

#![allow(unsafe_code)]

use std::sync::Mutex;

pub(crate) static ENV_LOCK: Mutex<()> = Mutex::new(());

pub(crate) struct EnvGuard {
    key: &'static str,
    previous: Option<String>,
}

impl EnvGuard {
    pub(crate) fn set(key: &'static str, value: &str) -> Self {
        let previous = std::env::var(key).ok();
        unsafe {
            std::env::set_var(key, value);
        }
        Self { key, previous }
    }

    #[allow(dead_code)]
    pub(crate) fn remove(key: &'static str) -> Self {
        let previous = std::env::var(key).ok();
        unsafe {
            std::env::remove_var(key);
        }
        Self { key, previous }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        match &self.previous {
            Some(v) => unsafe {
                std::env::set_var(self.key, v);
            },
            None => unsafe {
                std::env::remove_var(self.key);
            },
        }
    }
}
