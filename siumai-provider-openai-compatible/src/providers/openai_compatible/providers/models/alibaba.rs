//! Alibaba Cloud / Qwen model constants aligned with the AI SDK Alibaba package.
pub const QWEN3_MAX: &str = "qwen3-max";
pub const QWEN3_MAX_PREVIEW: &str = "qwen3-max-preview";
pub const QWEN_PLUS: &str = "qwen-plus";
pub const QWEN_PLUS_LATEST: &str = "qwen-plus-latest";
pub const QWEN_FLASH: &str = "qwen-flash";
pub const QWEN_TURBO: &str = "qwen-turbo";
pub const QWEN_TURBO_LATEST: &str = "qwen-turbo-latest";
pub const QWEN3_235B_A22B: &str = "qwen3-235b-a22b";
pub const QWEN3_32B: &str = "qwen3-32b";
pub const QWEN3_30B_A3B: &str = "qwen3-30b-a3b";
pub const QWEN3_14B: &str = "qwen3-14b";
pub const QWEN3_NEXT_80B_A3B_THINKING: &str = "qwen3-next-80b-a3b-thinking";
pub const QWEN3_235B_A22B_THINKING_2507: &str = "qwen3-235b-a22b-thinking-2507";
pub const QWEN3_30B_A3B_THINKING_2507: &str = "qwen3-30b-a3b-thinking-2507";
pub const QWQ_PLUS: &str = "qwq-plus";
pub const QWQ_PLUS_LATEST: &str = "qwq-plus-latest";
pub const QWQ_32B: &str = "qwq-32b";
pub const QWEN_CODER: &str = "qwen-coder";
pub const QWEN3_CODER_PLUS: &str = "qwen3-coder-plus";
pub const QWEN3_CODER_FLASH: &str = "qwen3-coder-flash";

pub const CHAT: &str = QWEN_PLUS;

pub const WAN2_6_T2V: &str = "wan2.6-t2v";
pub const WAN2_5_T2V_PREVIEW: &str = "wan2.5-t2v-preview";
pub const WAN2_6_I2V: &str = "wan2.6-i2v";
pub const WAN2_6_I2V_FLASH: &str = "wan2.6-i2v-flash";
pub const WAN2_6_R2V: &str = "wan2.6-r2v";
pub const WAN2_6_R2V_FLASH: &str = "wan2.6-r2v-flash";

pub const VIDEO: &str = WAN2_6_T2V;

pub const ALL: &[&str] = &[
    QWEN3_MAX,
    QWEN3_MAX_PREVIEW,
    QWEN_PLUS,
    QWEN_PLUS_LATEST,
    QWEN_FLASH,
    QWEN_TURBO,
    QWEN_TURBO_LATEST,
    QWEN3_235B_A22B,
    QWEN3_32B,
    QWEN3_30B_A3B,
    QWEN3_14B,
    QWEN3_NEXT_80B_A3B_THINKING,
    QWEN3_235B_A22B_THINKING_2507,
    QWEN3_30B_A3B_THINKING_2507,
    QWQ_PLUS,
    QWQ_PLUS_LATEST,
    QWQ_32B,
    QWEN_CODER,
    QWEN3_CODER_PLUS,
    QWEN3_CODER_FLASH,
];

pub const ALL_VIDEO: &[&str] = &[
    WAN2_6_T2V,
    WAN2_5_T2V_PREVIEW,
    WAN2_6_I2V,
    WAN2_6_I2V_FLASH,
    WAN2_6_R2V,
    WAN2_6_R2V_FLASH,
];

pub fn all_models() -> Vec<String> {
    ALL.iter().map(|&model| model.to_string()).collect()
}

pub fn all_video_models() -> Vec<String> {
    ALL_VIDEO.iter().map(|&model| model.to_string()).collect()
}
