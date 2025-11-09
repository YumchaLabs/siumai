//! OpenTelemetry integration for distributed tracing and metrics
//!
//! This module provides utilities for initializing OpenTelemetry with OTLP exporters,
//! collecting metrics, and creating middleware for automatic LLM request tracing.
//!
//! ## Features
//!
//! - **Distributed Tracing**: Automatic trace propagation across services
//! - **Metrics Collection**: Request latency, token usage, error rates
//! - **OTLP Export**: Send telemetry to Jaeger, Zipkin, Prometheus, etc.
//! - **Middleware Integration**: Automatic instrumentation of LLM requests
//!
//! ## Example
//!
//! ```rust,ignore
//! use siumai_extras::otel::{init_opentelemetry, OtelConfig};
//!
//! // Initialize OpenTelemetry with OTLP exporter
//! let config = OtelConfig::builder()
//!     .service_name("my-llm-service")
//!     .otlp_endpoint("http://localhost:4317")
//!     .build();
//!
//! let _guard = init_opentelemetry(config).await?;
//! ```

use crate::error::{ExtrasError, Result};
use opentelemetry::{KeyValue, global, trace::Tracer};
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk::{
    Resource,
    metrics::{PeriodicReader, SdkMeterProvider},
    trace::{RandomIdGenerator, Sampler, SdkTracerProvider},
};
use std::time::Duration;

/// OpenTelemetry configuration
#[derive(Debug, Clone)]
pub struct OtelConfig {
    /// Service name for telemetry
    pub service_name: String,
    /// OTLP endpoint (e.g., "http://localhost:4317")
    pub otlp_endpoint: Option<String>,
    /// Enable stdout exporter for debugging
    pub enable_stdout: bool,
    /// Trace sampling ratio (0.0 to 1.0)
    pub trace_sample_ratio: f64,
    /// Metrics export interval in seconds
    pub metrics_export_interval_secs: u64,
    /// Additional resource attributes
    pub resource_attributes: Vec<(String, String)>,
}

impl Default for OtelConfig {
    fn default() -> Self {
        Self {
            service_name: "siumai-service".to_string(),
            otlp_endpoint: None,
            enable_stdout: false,
            trace_sample_ratio: 1.0,
            metrics_export_interval_secs: 60,
            resource_attributes: Vec::new(),
        }
    }
}

impl OtelConfig {
    /// Create a new builder for OtelConfig
    pub fn builder() -> OtelConfigBuilder {
        OtelConfigBuilder::default()
    }
}

/// Builder for OtelConfig
#[derive(Debug, Default)]
pub struct OtelConfigBuilder {
    service_name: Option<String>,
    otlp_endpoint: Option<String>,
    enable_stdout: Option<bool>,
    trace_sample_ratio: Option<f64>,
    metrics_export_interval_secs: Option<u64>,
    resource_attributes: Vec<(String, String)>,
}

impl OtelConfigBuilder {
    /// Set the service name
    pub fn service_name(mut self, name: impl Into<String>) -> Self {
        self.service_name = Some(name.into());
        self
    }

    /// Set the OTLP endpoint
    pub fn otlp_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.otlp_endpoint = Some(endpoint.into());
        self
    }

    /// Enable stdout exporter for debugging
    pub fn enable_stdout(mut self, enable: bool) -> Self {
        self.enable_stdout = Some(enable);
        self
    }

    /// Set the trace sampling ratio (0.0 to 1.0)
    pub fn trace_sample_ratio(mut self, ratio: f64) -> Self {
        self.trace_sample_ratio = Some(ratio.clamp(0.0, 1.0));
        self
    }

    /// Set the metrics export interval in seconds
    pub fn metrics_export_interval_secs(mut self, secs: u64) -> Self {
        self.metrics_export_interval_secs = Some(secs);
        self
    }

    /// Add a resource attribute
    pub fn add_resource_attribute(
        mut self,
        key: impl Into<String>,
        value: impl Into<String>,
    ) -> Self {
        self.resource_attributes.push((key.into(), value.into()));
        self
    }

    /// Build the configuration
    pub fn build(self) -> OtelConfig {
        OtelConfig {
            service_name: self
                .service_name
                .unwrap_or_else(|| "siumai-service".to_string()),
            otlp_endpoint: self.otlp_endpoint,
            enable_stdout: self.enable_stdout.unwrap_or(false),
            trace_sample_ratio: self.trace_sample_ratio.unwrap_or(1.0),
            metrics_export_interval_secs: self.metrics_export_interval_secs.unwrap_or(60),
            resource_attributes: self.resource_attributes,
        }
    }
}

/// Guard that shuts down OpenTelemetry on drop
pub struct OtelGuard {
    _tracer_provider: Option<SdkTracerProvider>,
    _meter_provider: Option<SdkMeterProvider>,
}

impl Drop for OtelGuard {
    fn drop(&mut self) {
        // Shutdown tracer provider
        if let Some(provider) = self._tracer_provider.take()
            && let Err(e) = provider.shutdown()
        {
            eprintln!("Error shutting down tracer provider: {:?}", e);
        }

        // Shutdown meter provider
        if let Some(provider) = self._meter_provider.take()
            && let Err(e) = provider.shutdown()
        {
            eprintln!("Error shutting down meter provider: {:?}", e);
        }

        // global shutdown_tracer_provider was removed in otel 0.31; explicit shutdowns above suffice
    }
}

/// Initialize OpenTelemetry with the given configuration
///
/// This function sets up:
/// - Trace provider with OTLP and/or stdout exporter
/// - Meter provider for metrics collection
/// - Global tracer and meter providers
///
/// ## Arguments
///
/// - `config`: OpenTelemetry configuration
///
/// ## Returns
///
/// - `Ok(OtelGuard)`: Guard that must be kept alive for the duration of the program
/// - `Err(ExtrasError)`: If initialization fails
///
/// ## Example
///
/// ```rust,ignore
/// use siumai_extras::otel::{init_opentelemetry, OtelConfig};
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let config = OtelConfig::builder()
///         .service_name("my-service")
///         .otlp_endpoint("http://localhost:4317")
///         .build();
///
///     let _guard = init_opentelemetry(config).await?;
///
///     // Your application code here
///
///     Ok(())
/// }
/// ```
pub async fn init_opentelemetry(config: OtelConfig) -> Result<OtelGuard> {
    // Build resource
    let mut resource_kvs = vec![
        KeyValue::new(
            opentelemetry_semantic_conventions::resource::SERVICE_NAME,
            config.service_name.clone(),
        ),
        KeyValue::new(
            opentelemetry_semantic_conventions::resource::SERVICE_VERSION,
            env!("CARGO_PKG_VERSION"),
        ),
    ];

    for (key, value) in &config.resource_attributes {
        resource_kvs.push(KeyValue::new(key.clone(), value.clone()));
    }

    // Build Resource via public builder API (Resource::new is private in 0.31)
    let resource = Resource::builder_empty()
        .with_attributes(resource_kvs)
        .build();

    // Initialize tracer provider
    let tracer_provider = init_tracer_provider(&config, resource.clone()).await?;
    global::set_tracer_provider(tracer_provider.clone());

    // Initialize meter provider
    let meter_provider = init_meter_provider(&config, resource).await?;
    global::set_meter_provider(meter_provider.clone());

    Ok(OtelGuard {
        _tracer_provider: Some(tracer_provider),
        _meter_provider: Some(meter_provider),
    })
}

/// Initialize tracer provider
async fn init_tracer_provider(
    config: &OtelConfig,
    resource: Resource,
) -> Result<SdkTracerProvider> {
    let mut builder = SdkTracerProvider::builder()
        .with_resource(resource)
        .with_id_generator(RandomIdGenerator::default())
        .with_sampler(Sampler::TraceIdRatioBased(config.trace_sample_ratio));

    // Add OTLP exporter if endpoint is configured
    if let Some(endpoint) = &config.otlp_endpoint {
        let exporter = opentelemetry_otlp::SpanExporter::builder()
            .with_tonic()
            .with_endpoint(endpoint)
            .build()
            .map_err(|e| {
                ExtrasError::TelemetryInit(format!("Failed to create OTLP span exporter: {}", e))
            })?;

        builder = builder.with_batch_exporter(exporter);
    }

    // Add stdout exporter if enabled
    if config.enable_stdout {
        let exporter = opentelemetry_stdout::SpanExporter::default();
        builder = builder.with_simple_exporter(exporter);
    }

    Ok(builder.build())
}

/// Initialize meter provider
async fn init_meter_provider(config: &OtelConfig, resource: Resource) -> Result<SdkMeterProvider> {
    let mut builder = SdkMeterProvider::builder().with_resource(resource);

    // Add OTLP exporter if endpoint is configured
    if let Some(endpoint) = &config.otlp_endpoint {
        let exporter = opentelemetry_otlp::MetricExporter::builder()
            .with_tonic()
            .with_endpoint(endpoint)
            .build()
            .map_err(|e| {
                ExtrasError::TelemetryInit(format!("Failed to create OTLP metric exporter: {}", e))
            })?;

        let reader = PeriodicReader::builder(exporter)
            .with_interval(Duration::from_secs(config.metrics_export_interval_secs))
            .build();

        builder = builder.with_reader(reader);
    }

    // Add stdout exporter if enabled
    if config.enable_stdout {
        let exporter = opentelemetry_stdout::MetricExporter::default();
        let reader = PeriodicReader::builder(exporter)
            .with_interval(Duration::from_secs(config.metrics_export_interval_secs))
            .build();

        builder = builder.with_reader(reader);
    }

    Ok(builder.build())
}

/// Get the global tracer for siumai
pub fn tracer() -> impl Tracer {
    global::tracer("siumai")
}
