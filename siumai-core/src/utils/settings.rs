//! AI SDK-style environment setting loaders.

use crate::types::{LoadAPIKeyError, LoadSettingError};

/// Options for [`load_api_key`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadApiKeyOptions {
    /// Explicit API key value.
    pub api_key: Option<String>,
    /// Environment variable to read when `api_key` is missing.
    pub environment_variable_name: String,
    /// Parameter name shown in missing-key errors.
    pub api_key_parameter_name: String,
    /// Human-readable provider/setting description.
    pub description: String,
}

impl LoadApiKeyOptions {
    /// Create API-key loading options.
    pub fn new(
        environment_variable_name: impl Into<String>,
        description: impl Into<String>,
    ) -> Self {
        Self {
            api_key: None,
            environment_variable_name: environment_variable_name.into(),
            api_key_parameter_name: "apiKey".to_string(),
            description: description.into(),
        }
    }

    /// Set an explicit API key.
    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Set the parameter name used in missing-key errors.
    pub fn with_api_key_parameter_name(
        mut self,
        api_key_parameter_name: impl Into<String>,
    ) -> Self {
        self.api_key_parameter_name = api_key_parameter_name.into();
        self
    }
}

/// Load an API key from an explicit option or environment variable.
///
/// Rust values are already typed as strings, so the JavaScript-only "must be a
/// string" branches are not modeled. Empty strings are preserved to match the
/// AI SDK helper's string semantics.
pub fn load_api_key(options: LoadApiKeyOptions) -> Result<String, LoadAPIKeyError> {
    if let Some(api_key) = options.api_key {
        return Ok(api_key);
    }

    std::env::var(&options.environment_variable_name).map_err(|_| {
        LoadAPIKeyError::new(format!(
            "{} API key is missing. Pass it using the '{}' parameter or the {} environment variable.",
            options.description, options.api_key_parameter_name, options.environment_variable_name
        ))
    })
}

/// Options for [`load_setting`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadSettingOptions {
    /// Explicit setting value.
    pub setting_value: Option<String>,
    /// Environment variable to read when `setting_value` is missing.
    pub environment_variable_name: String,
    /// Parameter/setting name shown in missing-setting errors.
    pub setting_name: String,
    /// Human-readable setting description.
    pub description: String,
}

impl LoadSettingOptions {
    /// Create setting loading options.
    pub fn new(
        environment_variable_name: impl Into<String>,
        setting_name: impl Into<String>,
        description: impl Into<String>,
    ) -> Self {
        Self {
            setting_value: None,
            environment_variable_name: environment_variable_name.into(),
            setting_name: setting_name.into(),
            description: description.into(),
        }
    }

    /// Set an explicit setting value.
    pub fn with_setting_value(mut self, setting_value: impl Into<String>) -> Self {
        self.setting_value = Some(setting_value.into());
        self
    }
}

/// Load a required setting from an explicit option or environment variable.
pub fn load_setting(options: LoadSettingOptions) -> Result<String, LoadSettingError> {
    if let Some(setting_value) = options.setting_value {
        return Ok(setting_value);
    }

    std::env::var(&options.environment_variable_name).map_err(|_| {
        LoadSettingError::new(format!(
            "{} setting is missing. Pass it using the '{}' parameter or the {} environment variable.",
            options.description, options.setting_name, options.environment_variable_name
        ))
    })
}

/// Options for [`load_optional_setting`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadOptionalSettingOptions {
    /// Explicit setting value.
    pub setting_value: Option<String>,
    /// Environment variable to read when `setting_value` is missing.
    pub environment_variable_name: String,
}

impl LoadOptionalSettingOptions {
    /// Create optional setting loading options.
    pub fn new(environment_variable_name: impl Into<String>) -> Self {
        Self {
            setting_value: None,
            environment_variable_name: environment_variable_name.into(),
        }
    }

    /// Set an explicit setting value.
    pub fn with_setting_value(mut self, setting_value: impl Into<String>) -> Self {
        self.setting_value = Some(setting_value.into());
        self
    }
}

/// Load an optional setting from an explicit option or environment variable.
pub fn load_optional_setting(options: LoadOptionalSettingOptions) -> Option<String> {
    options
        .setting_value
        .or_else(|| std::env::var(&options.environment_variable_name).ok())
}

#[cfg(test)]
mod tests {
    #![allow(unsafe_code)]

    use super::*;
    use std::sync::{Mutex, MutexGuard};

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn lock_env() -> MutexGuard<'static, ()> {
        ENV_LOCK.lock().unwrap_or_else(|error| error.into_inner())
    }

    struct EnvGuard {
        key: &'static str,
        previous: Option<String>,
    }

    impl EnvGuard {
        fn set(key: &'static str, value: &str) -> Self {
            let previous = std::env::var(key).ok();
            unsafe {
                std::env::set_var(key, value);
            }
            Self { key, previous }
        }

        fn remove(key: &'static str) -> Self {
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
                Some(value) => unsafe {
                    std::env::set_var(self.key, value);
                },
                None => unsafe {
                    std::env::remove_var(self.key);
                },
            }
        }
    }

    #[test]
    fn load_api_key_prefers_explicit_value() {
        let _lock = lock_env();
        let _guard = EnvGuard::set("SIUMAI_TEST_LOAD_API_KEY", "from-env");

        let key = load_api_key(
            LoadApiKeyOptions::new("SIUMAI_TEST_LOAD_API_KEY", "Test").with_api_key("from-option"),
        )
        .expect("api key");

        assert_eq!(key, "from-option");
    }

    #[test]
    fn load_api_key_reads_environment_or_errors() {
        let _lock = lock_env();
        let _guard = EnvGuard::set("SIUMAI_TEST_LOAD_API_KEY", "from-env");

        assert_eq!(
            load_api_key(LoadApiKeyOptions::new("SIUMAI_TEST_LOAD_API_KEY", "Test"))
                .expect("api key"),
            "from-env"
        );

        let _guard = EnvGuard::remove("SIUMAI_TEST_LOAD_API_KEY_MISSING");
        let error = load_api_key(LoadApiKeyOptions::new(
            "SIUMAI_TEST_LOAD_API_KEY_MISSING",
            "Test",
        ))
        .expect_err("missing api key");
        assert!(error.message.contains("SIUMAI_TEST_LOAD_API_KEY_MISSING"));
    }

    #[test]
    fn load_setting_prefers_explicit_then_environment() {
        let _lock = lock_env();
        let _guard = EnvGuard::set("SIUMAI_TEST_LOAD_SETTING", "from-env");

        assert_eq!(
            load_setting(
                LoadSettingOptions::new("SIUMAI_TEST_LOAD_SETTING", "setting", "Test")
                    .with_setting_value("from-option"),
            )
            .expect("setting"),
            "from-option"
        );
        assert_eq!(
            load_setting(LoadSettingOptions::new(
                "SIUMAI_TEST_LOAD_SETTING",
                "setting",
                "Test",
            ))
            .expect("setting"),
            "from-env"
        );
    }

    #[test]
    fn load_optional_setting_returns_none_when_missing() {
        let _lock = lock_env();
        let _guard = EnvGuard::remove("SIUMAI_TEST_LOAD_OPTIONAL_SETTING");

        assert_eq!(
            load_optional_setting(
                LoadOptionalSettingOptions::new("SIUMAI_TEST_LOAD_OPTIONAL_SETTING")
                    .with_setting_value("from-option"),
            ),
            Some("from-option".to_string())
        );
        assert_eq!(
            load_optional_setting(LoadOptionalSettingOptions::new(
                "SIUMAI_TEST_LOAD_OPTIONAL_SETTING"
            )),
            None
        );
    }
}
