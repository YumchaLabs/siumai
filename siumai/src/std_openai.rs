// Bridge module for OpenAI standard (external-only re-export)

pub mod openai {
    pub mod image {
        pub use siumai_std_openai::openai::image::*;
    }
    pub mod chat {
        pub use siumai_std_openai::openai::chat::*;
    }
    pub mod embedding {
        pub use siumai_std_openai::openai::embedding::*;
    }
    pub mod rerank {
        pub use siumai_std_openai::openai::rerank::*;
    }
    pub mod responses {
        pub use siumai_std_openai::openai::responses::*;
    }
}
