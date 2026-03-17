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
    // === Phase 1: Basic inference ===
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

    // === Phase 2: Distributed inference ===
    /// Node announces itself with pricing (Phase 2.1)
    NodeAnnouncement(NodeAnnouncement),
    /// Periodic heartbeat to show node is alive
    NodeHeartbeat {
        peer_id: String,
        timestamp: u64,
        queue_depth: u32,
        load: f32,
    },
    /// Request list of available nodes
    NodeListRequest {
        model_filter: Option<String>,
        max_price: Option<u64>,
    },
    /// Response with available nodes
    NodeListResponse {
        nodes: Vec<NodeInfo>,
    },
    /// Route inference to specific node (from router)
    RoutedInference {
        request_id: String,
        target_peer: String,
        from_peer: String,
        request: ChatCompletionRequest,
        max_price: Option<u64>,
    },
    /// Billing record after successful inference
    InferenceBilling {
        request_id: String,
        node_address: String,
        user_address: String,
        input_tokens: u32,
        output_tokens: u32,
        price_per_1k: u64,
        total_cost_nynet: u64,
    },
}

/// Full node announcement with pricing (Phase 2.1)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeAnnouncement {
    /// Node's wallet address for receiving payments
    pub address: String,
    /// P2P peer ID
    pub peer_id: String,
    /// Network addresses (multiaddr format)
    pub listen_addrs: Vec<String>,
    /// GPU information
    pub gpus: Vec<GpuInfo>,
    /// Total VRAM in MB
    pub vram_total_mb: u64,
    /// Available models with pricing
    pub models: Vec<ModelPricing>,
    /// Max concurrent requests
    pub max_concurrent: u32,
    /// Current queue depth
    pub queue_depth: u32,
    /// Current load (0.0 - 1.0)
    pub load: f32,
    /// Timestamp of this announcement
    pub timestamp: u64,
    /// Node's reputation score (0-100)
    pub reputation: u32,
}

/// Model with pricing information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPricing {
    /// Model identifier
    pub model_id: String,
    /// Backend type
    pub backend: InferenceBackend,
    /// Port for this model
    pub port: u16,
    /// Price per 1000 input tokens (nYNET)
    pub price_input_per_1k: u64,
    /// Price per 1000 output tokens (nYNET)
    pub price_output_per_1k: u64,
    /// Max context length
    pub max_context: u32,
    /// Average latency in ms
    pub avg_latency_ms: u32,
}

/// Simplified node info for routing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeInfo {
    pub peer_id: String,
    pub address: String,
    pub model_id: String,
    pub price_input_per_1k: u64,
    pub price_output_per_1k: u64,
    pub avg_latency_ms: u32,
    pub load: f32,
    pub reputation: u32,
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
