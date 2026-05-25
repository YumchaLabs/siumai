use crate::provider_options::GatewayOptions;

pub trait GatewayChatRequestExt {
    fn with_gateway_options(self, options: GatewayOptions) -> Self;
}

impl GatewayChatRequestExt for crate::types::ChatRequest {
    fn with_gateway_options(mut self, options: GatewayOptions) -> Self {
        self.provider_options_map.insert(
            "gateway",
            serde_json::to_value(options).expect("serialize GatewayOptions"),
        );
        self
    }
}

pub trait GatewayEmbeddingRequestExt {
    fn with_gateway_options(self, options: GatewayOptions) -> Self;
}

impl GatewayEmbeddingRequestExt for crate::types::EmbeddingRequest {
    fn with_gateway_options(mut self, options: GatewayOptions) -> Self {
        self.provider_options_map.insert(
            "gateway",
            serde_json::to_value(options).expect("serialize GatewayOptions"),
        );
        self
    }
}
