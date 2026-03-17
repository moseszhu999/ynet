//! yNet inference layer — transforms the P2P network into a decentralized AI inference platform.
//!
//! Architecture:
//! - **Service**: manages local model backends (vLLM, llama.cpp, Ollama)
//! - **Registry**: tracks which nodes have which models (network-wide view)
//! - **Gateway**: OpenAI-compatible HTTP API, routes to best available node
//!
//! Models load once and stay resident in GPU memory. Requests route to nodes
//! that have the model loaded, with smart load balancing.

pub mod failover;
pub mod gateway;
pub mod registry;
pub mod reputation;
pub mod router;
pub mod service;
pub mod sharding;
pub mod types;
pub mod utils;

pub use gateway::{GatewayState, LocalBackendInfo, StreamEvent};
pub use registry::NodeRegistry;
pub use reputation::{ReputationEvent, ReputationEventType, ReputationTracker, SharedReputationTracker};
pub use router::{InferenceRouter, RoutingDecision, RoutingPreferences, RoutingReason};
pub use service::InferenceService;
pub use sharding::{ResultAggregator, ShardConfig, ShardInput, ShardOutput, ShardResult, ShardStrategy, ShardType, ShardingManager};
pub use types::{
    ChatCompletionRequest, ChatCompletionResponse, InferenceBackend, InferenceMessage,
    NodeCapability, NodeAnnouncement, NodeInfo, ModelPricing, GpuInfo, LoadedModel,
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

    // === Phase 2 tests ===

    #[test]
    fn test_node_announcement_round_trip() {
        let ann = NodeAnnouncement {
            address: "ynet1abc".to_string(),
            peer_id: "peer1".to_string(),
            listen_addrs: vec!["/ip4/127.0.0.1/tcp/4001".to_string()],
            gpus: vec![GpuInfo {
                name: "RTX 4090".to_string(),
                vram_mb: 24576,
            }],
            vram_total_mb: 24576,
            models: vec![ModelPricing {
                model_id: "kimi-k2.5".to_string(),
                backend: InferenceBackend::LlamaCpp,
                port: 8080,
                price_input_per_1k: 100,
                price_output_per_1k: 50,
                max_context: 4096,
                avg_latency_ms: 100,
            }],
            max_concurrent: 4,
            queue_depth: 1,
            load: 0.25,
            timestamp: 1234567890,
            reputation: 85,
        };

        let json = serde_json::to_string(&ann).unwrap();
        let decoded: NodeAnnouncement = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.peer_id, "peer1");
        assert_eq!(decoded.models.len(), 1);
        assert_eq!(decoded.models[0].price_input_per_1k, 100);
    }

    #[test]
    fn test_phase2_message_variants() {
        // NodeHeartbeat
        let heartbeat = InferenceMessage::NodeHeartbeat {
            peer_id: "peer1".to_string(),
            timestamp: 1234567890,
            queue_depth: 2,
            load: 0.5,
        };
        let json = serde_json::to_string(&heartbeat).unwrap();
        let decoded: InferenceMessage = serde_json::from_str(&json).unwrap();
        match decoded {
            InferenceMessage::NodeHeartbeat { peer_id, queue_depth, load, .. } => {
                assert_eq!(peer_id, "peer1");
                assert_eq!(queue_depth, 2);
                assert!((load - 0.5).abs() < 0.001);
            }
            _ => panic!("wrong variant"),
        }

        // NodeListRequest
        let list_req = InferenceMessage::NodeListRequest {
            model_filter: Some("kimi-k2.5".to_string()),
            max_price: Some(1000),
        };
        let json = serde_json::to_string(&list_req).unwrap();
        let decoded: InferenceMessage = serde_json::from_str(&json).unwrap();
        match decoded {
            InferenceMessage::NodeListRequest { model_filter, max_price } => {
                assert_eq!(model_filter, Some("kimi-k2.5".to_string()));
                assert_eq!(max_price, Some(1000));
            }
            _ => panic!("wrong variant"),
        }

        // RoutedInference
        let routed = InferenceMessage::RoutedInference {
            request_id: "req-1".to_string(),
            target_peer: "target-peer".to_string(),
            from_peer: "from-peer".to_string(),
            request: ChatCompletionRequest {
                model: "kimi-k2.5".to_string(),
                messages: vec![],
                temperature: None,
                max_tokens: None,
                stream: None,
                ynet_address: None,
            },
            max_price: Some(500),
        };
        let json = serde_json::to_string(&routed).unwrap();
        let decoded: InferenceMessage = serde_json::from_str(&json).unwrap();
        match decoded {
            InferenceMessage::RoutedInference { request_id, target_peer, max_price, .. } => {
                assert_eq!(request_id, "req-1");
                assert_eq!(target_peer, "target-peer");
                assert_eq!(max_price, Some(500));
            }
            _ => panic!("wrong variant"),
        }

        // InferenceBilling
        let billing = InferenceMessage::InferenceBilling {
            request_id: "req-1".to_string(),
            node_address: "ynet1node".to_string(),
            user_address: "ynet1user".to_string(),
            input_tokens: 100,
            output_tokens: 50,
            price_per_1k: 100,
            total_cost_nynet: 15,
        };
        let json = serde_json::to_string(&billing).unwrap();
        let decoded: InferenceMessage = serde_json::from_str(&json).unwrap();
        match decoded {
            InferenceMessage::InferenceBilling {
                request_id,
                input_tokens,
                output_tokens,
                total_cost_nynet,
                ..
            } => {
                assert_eq!(request_id, "req-1");
                assert_eq!(input_tokens, 100);
                assert_eq!(output_tokens, 50);
                assert_eq!(total_cost_nynet, 15);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn test_registry_find_nodes_with_pricing() {
        let mut reg = NodeRegistry::new();
        let now = crate::utils::current_timestamp();

        // Add announcement with pricing
        reg.update_announcement(NodeAnnouncement {
            address: "ynet1node1".to_string(),
            peer_id: "node1".to_string(),
            listen_addrs: vec![],
            gpus: vec![],
            vram_total_mb: 24576,
            models: vec![ModelPricing {
                model_id: "kimi-k2.5".to_string(),
                backend: InferenceBackend::Vllm,
                port: 8000,
                price_input_per_1k: 100,
                price_output_per_1k: 50,
                max_context: 4096,
                avg_latency_ms: 100,
            }],
            max_concurrent: 4,
            queue_depth: 1,
            load: 0.25,
            timestamp: now,
            reputation: 90,
        });

        reg.update_announcement(NodeAnnouncement {
            address: "ynet1node2".to_string(),
            peer_id: "node2".to_string(),
            listen_addrs: vec![],
            gpus: vec![],
            vram_total_mb: 24576,
            models: vec![ModelPricing {
                model_id: "kimi-k2.5".to_string(),
                backend: InferenceBackend::LlamaCpp,
                port: 8080,
                price_input_per_1k: 200, // More expensive
                price_output_per_1k: 100,
                max_context: 4096,
                avg_latency_ms: 50, // Faster
            }],
            max_concurrent: 4,
            queue_depth: 0,
            load: 0.1,
            timestamp: now,
            reputation: 80,
        });

        // Find nodes with max price filter
        let nodes = reg.find_nodes_with_pricing("kimi-k2.5", Some(150));
        assert_eq!(nodes.len(), 1); // Only node1 matches price filter
        assert_eq!(nodes[0].peer_id, "node1");

        // Find all nodes
        let all_nodes = reg.find_nodes_with_pricing("kimi-k2.5", None);
        assert_eq!(all_nodes.len(), 2);
    }

    #[test]
    fn test_registry_get_all_nodes_info() {
        let mut reg = NodeRegistry::new();
        let now = crate::utils::current_timestamp();

        reg.update_announcement(NodeAnnouncement {
            address: "ynet1node1".to_string(),
            peer_id: "node1".to_string(),
            listen_addrs: vec![],
            gpus: vec![],
            vram_total_mb: 24576,
            models: vec![
                ModelPricing {
                    model_id: "model-a".to_string(),
                    backend: InferenceBackend::Vllm,
                    port: 8000,
                    price_input_per_1k: 100,
                    price_output_per_1k: 50,
                    max_context: 4096,
                    avg_latency_ms: 100,
                },
                ModelPricing {
                    model_id: "model-b".to_string(),
                    backend: InferenceBackend::Vllm,
                    port: 8001,
                    price_input_per_1k: 150,
                    price_output_per_1k: 75,
                    max_context: 8192,
                    avg_latency_ms: 150,
                },
            ],
            max_concurrent: 4,
            queue_depth: 1,
            load: 0.25,
            timestamp: now,
            reputation: 90,
        });

        let all_info = reg.get_all_nodes_info();
        assert_eq!(all_info.len(), 2); // Two models = two NodeInfo entries
    }
}
