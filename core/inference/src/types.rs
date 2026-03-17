//! OpenAI-compatible API types for yNet inference.

use serde::{Deserialize, Serialize};

// ---- Request types ----

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    #[serde(default)]
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub stream: Option<bool>,
    /// yNet extension: pay with this wallet address
    #[serde(default)]
    pub ynet_address: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

// ---- Response types ----

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    #[serde(default)]
    pub system_fingerprint: Option<String>,
    pub choices: Vec<ChatChoice>,
    pub usage: Option<Usage>,
    /// yNet extension: cost in nYNET
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ynet_cost: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatChoice {
    pub index: u32,
    pub message: Option<ChatMessage>,
    pub delta: Option<ChatMessage>,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

// ---- Streaming chunk ----

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
}

// ---- Model listing ----

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelList {
    pub object: String,
    pub data: Vec<ModelInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    pub object: String,
    pub owned_by: String,
    /// yNet extension: which nodes have this model loaded
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ynet_nodes: Option<u32>,
}

// ---- P2P inference protocol ----

/// Messages for inference coordination over P2P.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InferenceMessage {
    /// Node announces its inference capabilities.
    NodeCapability(NodeCapability),
    /// Forward an inference request to a remote node.
    InferenceRequest {
        request_id: String,
        from_peer: String,
        request: ChatCompletionRequest,
    },
    /// Stream a chunk of the response back.
    InferenceChunk {
        request_id: String,
        to_peer: String,
        chunk: String,
        done: bool,
    },
    /// Report inference error.
    InferenceError {
        request_id: String,
        to_peer: String,
        error: String,
    },
}

/// A node's inference capabilities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeCapability {
    /// Node's wallet address.
    pub address: String,
    /// P2P peer ID.
    pub peer_id: String,
    /// GPU info.
    pub gpus: Vec<GpuInfo>,
    /// Total VRAM in MB.
    pub vram_total_mb: u64,
    /// Currently loaded models (ready to serve).
    pub loaded_models: Vec<LoadedModel>,
    /// Max concurrent requests this node can handle.
    pub max_concurrent: u32,
    /// Current queue depth.
    pub queue_depth: u32,
    /// Timestamp of this announcement.
    pub timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuInfo {
    pub name: String,
    pub vram_mb: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadedModel {
    /// Model identifier (e.g., "kimi-k2.5-q4", "qwen2-72b")
    pub model_id: String,
    /// Backend serving this model
    pub backend: InferenceBackend,
    /// Local port the backend listens on
    pub port: u16,
    /// Max context length
    pub max_context: u32,
    /// Tokens per second throughput
    pub tokens_per_sec: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum InferenceBackend {
    /// llama.cpp server (llama-server / llama-cli --server)
    LlamaCpp,
    /// vLLM (OpenAI-compatible server)
    Vllm,
    /// Ollama
    Ollama,
    /// Any OpenAI-compatible server at a custom URL
    Custom,
}
