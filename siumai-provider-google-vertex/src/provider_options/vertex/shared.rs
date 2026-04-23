use serde::{Deserialize, Serialize};

/// Shared Vertex person-generation policy used by image and video model options.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VertexPersonGeneration {
    #[serde(rename = "dont_allow")]
    DontAllow,
    #[serde(rename = "allow_adult")]
    AllowAdult,
    #[serde(rename = "allow_all")]
    AllowAll,
}

impl VertexPersonGeneration {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::DontAllow => "dont_allow",
            Self::AllowAdult => "allow_adult",
            Self::AllowAll => "allow_all",
        }
    }
}

impl From<VertexPersonGeneration> for String {
    fn from(value: VertexPersonGeneration) -> Self {
        value.as_str().to_string()
    }
}
