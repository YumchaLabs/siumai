//! Bridge module for Gemini standard (external-only re-export)

pub mod gemini {
    pub mod chat {
        pub use siumai_std_gemini::gemini::chat::*;
    }
    pub mod embedding {
        pub use siumai_std_gemini::gemini::embedding::*;
    }
    pub mod image {
        pub use siumai_std_gemini::gemini::image::*;
    }
}
