//! yNet inference layer — transforms the P2P network into a decentralized AI inference platform.
//!
//! Architecture:
//! - **Service**: manages local model backends (vLLM, llama.cpp, Ollama)
//! - **Registry**: tracks which nodes have which models (network-wide view)
//! - **Gateway**: OpenAI-compatible HTTP API, routes to best available node
//!
//! Models load once and stay resident in GPU memory. Requests route to nodes
//! that have the model loaded, with smart load balancing.

pub mod gateway;
pub mod registry;
pub mod service;
pub mod types;

pub use gateway::{GatewayState, LocalBackendInfo, StreamEvent};
pub use registry::NodeRegistry;
pub use service::InferenceService;
pub use types::{
    ChatCompletionRequest, ChatCompletionResponse, InferenceBackend, InferenceMessage,
    NodeCapability,
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ChatMessage, GpuInfo, LoadedModel};

    #[test]
    fn test_capability_round_trip() {
        let cap = NodeCapability {
            address: "ynet1abc".to_string(),
            peer_id: "peer1".to_string(),
            gpus: vec![GpuInfo {
                name: "RTX 4090".to_string(),
                vram_mb: 24576,
            }],
            vram_total_mb: 24576,
            loaded_models: vec![LoadedModel {
                model_id: "kimi-k2.5-q4".to_string(),
                backend: InferenceBackend::LlamaCpp,
                port: 8080,
                max_context: 4096,
                tokens_per_sec: 30.0,
            }],
            max_concurrent: 4,
            queue_depth: 1,
            timestamp: 1234567890,
        };

        let json = serde_json::to_string(&cap).unwrap();
        let decoded: NodeCapability = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.peer_id, "peer1");
        assert_eq!(decoded.loaded_models[0].model_id, "kimi-k2.5-q4");
    }

    #[test]
    fn test_inference_message_serialize() {
        let msg = InferenceMessage::InferenceRequest {
            request_id: "req-1".to_string(),
            from_peer: "peer-a".to_string(),
            request: ChatCompletionRequest {
                model: "kimi-k2.5".to_string(),
                messages: vec![ChatMessage {
                    role: "user".to_string(),
                    content: "hello".to_string(),
                }],
                temperature: Some(0.7),
                max_tokens: Some(100),
                stream: Some(true),
                ynet_address: None,
            },
        };

        let json = serde_json::to_string(&msg).unwrap();
        let decoded: InferenceMessage = serde_json::from_str(&json).unwrap();
        match decoded {
            InferenceMessage::InferenceRequest { request_id, .. } => {
                assert_eq!(request_id, "req-1");
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn test_registry_routing() {
        let mut reg = NodeRegistry::new();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        // Node with low queue
        reg.update(NodeCapability {
            address: "addr1".to_string(),
            peer_id: "fast-node".to_string(),
            gpus: vec![],
            vram_total_mb: 24576,
            loaded_models: vec![LoadedModel {
                model_id: "kimi-k2.5".to_string(),
                backend: InferenceBackend::Vllm,
                port: 8000,
                max_context: 4096,
                tokens_per_sec: 50.0,
            }],
            max_concurrent: 8,
            queue_depth: 0,
            timestamp: now,
        });

        // Node with high queue
        reg.update(NodeCapability {
            address: "addr2".to_string(),
            peer_id: "busy-node".to_string(),
            gpus: vec![],
            vram_total_mb: 24576,
            loaded_models: vec![LoadedModel {
                model_id: "kimi-k2.5".to_string(),
                backend: InferenceBackend::Vllm,
                port: 8000,
                max_context: 4096,
                tokens_per_sec: 50.0,
            }],
            max_concurrent: 8,
            queue_depth: 5,
            timestamp: now,
        });

        let nodes = reg.find_nodes_for_model("kimi-k2.5");
        assert_eq!(nodes.len(), 2);
        assert_eq!(nodes[0].peer_id, "fast-node"); // lower queue wins
    }
}
