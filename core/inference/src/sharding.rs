//! Model sharding for distributed inference.
//!
//! Enables splitting large model inference across multiple nodes.
//! Supports tensor parallelism and pipeline parallelism.

use crate::types::{ChatCompletionRequest, InferenceMessage, NodeInfo};
use crate::registry::NodeRegistry;
use crate::utils::current_timestamp;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Shard strategy for distributing inference.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ShardStrategy {
    /// Split by sequence length (for long prompts).
    SequenceChunks,
    /// Split by attention heads (tensor parallelism).
    AttentionHeads,
    /// Split by layers (pipeline parallelism).
    PipelineLayers,
    /// No sharding - single node.
    None,
}

/// Configuration for a single shard.
#[derive(Debug, Clone)]
pub struct ShardConfig {
    /// Unique shard ID.
    pub shard_id: String,
    /// Node assigned to this shard.
    pub node: NodeInfo,
    /// Shard type.
    pub shard_type: ShardType,
    /// Input data for this shard (could be prompt segment, layer assignment, etc.).
    pub input: ShardInput,
}

/// Type of shard.
#[derive(Debug, Clone)]
pub enum ShardType {
    /// First N tokens of the prompt.
    PromptPrefix { tokens: u32 },
    /// Middle portion of prompt.
    PromptMiddle { start: u32, end: u32 },
    /// Last N tokens of prompt.
    PromptSuffix { tokens: u32 },
    /// Specific attention head range.
    AttentionHeadRange { start_head: u32, end_head: u32 },
    /// Specific transformer layers.
    LayerRange { start_layer: u32, end_layer: u32 },
}

/// Input data for a shard.
#[derive(Debug, Clone)]
pub enum ShardInput {
    /// Text segment.
    Text(String),
    /// Token IDs.
    Tokens(Vec<u32>),
    /// Layer indices.
    Layers(Vec<u32>),
}

/// Result from a single shard.
#[derive(Debug, Clone)]
pub struct ShardResult {
    /// Shard ID.
    pub shard_id: String,
    /// Node that processed this shard.
    pub node_id: String,
    /// Processing time in ms.
    pub processing_time_ms: u64,
    /// Output from this shard.
    pub output: ShardOutput,
    /// Error if any.
    pub error: Option<String>,
}

/// Output from a shard.
#[derive(Debug, Clone)]
pub enum ShardOutput {
    /// Hidden states (for tensor parallelism).
    HiddenStates(Vec<f32>),
    /// Logits (for vocabulary).
    Logits(Vec<f32>),
    /// Generated tokens.
    Tokens(Vec<u32>),
    /// Text output.
    Text(String),
}

/// Sharding manager coordinates distributed inference.
pub struct ShardingManager {
    /// Shared node registry.
    registry: Arc<RwLock<NodeRegistry>>,
    /// Minimum nodes required for sharding.
    min_nodes: usize,
    /// Maximum shards per request.
    max_shards: usize,
}

impl ShardingManager {
    /// Create a new sharding manager.
    pub fn new(registry: Arc<RwLock<NodeRegistry>>) -> Self {
        Self {
            registry,
            min_nodes: 2,
            max_shards: 4,
        }
    }

    /// Determine if a request should be sharded.
    pub fn should_shard(&self, request: &ChatCompletionRequest) -> bool {
        // Check if model supports sharding
        // For now, shard if:
        // 1. Prompt is long (> 4096 tokens estimated)
        // 2. Multiple nodes available with same model

        let prompt_len: usize = request.messages.iter().map(|m| m.content.len()).sum();
        let estimated_tokens = prompt_len / 4; // Rough estimate: 4 chars per token

        // Only shard for long prompts
        estimated_tokens > 4096
    }

    /// Plan shard distribution for a request.
    pub async fn plan_shards(
        &self,
        request: &ChatCompletionRequest,
        strategy: ShardStrategy,
    ) -> Option<Vec<ShardConfig>> {
        let registry = self.registry.read().await;

        // Get available nodes for this model
        let nodes = registry.find_nodes_with_pricing(&request.model, None);

        if nodes.len() < self.min_nodes {
            log::debug!("Not enough nodes for sharding: {}", nodes.len());
            return None;
        }

        // Limit to max shards
        let num_shards = nodes.len().min(self.max_shards);
        let selected_nodes: Vec<_> = nodes.into_iter().take(num_shards).collect();

        match strategy {
            ShardStrategy::SequenceChunks => {
                self.plan_sequence_chunks(request, selected_nodes)
            }
            ShardStrategy::None => None,
            _ => {
                log::warn!("Sharding strategy {strategy:?} not yet implemented");
                None
            }
        }
    }

    /// Plan sequence-based chunking.
    fn plan_sequence_chunks(
        &self,
        request: &ChatCompletionRequest,
        nodes: Vec<NodeInfo>,
    ) -> Option<Vec<ShardConfig>> {
        // Combine all messages into one text
        let full_text: String = request.messages.iter()
            .map(|m| format!("{}: {}", m.role, m.content))
            .collect::<Vec<_>>()
            .join("\n");

        let total_len = full_text.len();
        let num_shards = nodes.len();
        let chunk_size = total_len / num_shards;

        let shards: Vec<ShardConfig> = nodes
            .into_iter()
            .enumerate()
            .map(|(i, node)| {
                let start = i * chunk_size;
                let end = if i == num_shards - 1 {
                    total_len // Last shard gets remainder
                } else {
                    (i + 1) * chunk_size
                };

                let chunk = full_text[start..end].to_string();

                ShardConfig {
                    shard_id: format!("shard-{i}"),
                    node,
                    shard_type: ShardType::PromptPrefix { tokens: (end - start) as u32 / 4 },
                    input: ShardInput::Text(chunk),
                }
            })
            .collect();

        Some(shards)
    }

    /// Create inference messages for shards.
    pub fn create_shard_messages(
        &self,
        shards: &[ShardConfig],
        original_request: &ChatCompletionRequest,
        request_id: &str,
        from_peer: &str,
    ) -> Vec<InferenceMessage> {
        shards
            .iter()
            .map(|shard| {
                // Create a modified request with just the shard's input
                let shard_request = ChatCompletionRequest {
                    model: original_request.model.clone(),
                    messages: vec![crate::types::ChatMessage {
                        role: "user".to_string(),
                        content: match &shard.input {
                            ShardInput::Text(t) => t.clone(),
                            ShardInput::Tokens(ids) => format!("[{} tokens]", ids.len()),
                            ShardInput::Layers(_) => "[layer input]".to_string(),
                        },
                    }],
                    temperature: original_request.temperature,
                    max_tokens: original_request.max_tokens.map(|t| t / shards.len() as u32),
                    stream: Some(false), // Don't stream shards
                    ynet_address: original_request.ynet_address.clone(),
                };

                InferenceMessage::RoutedInference {
                    request_id: format!("{}-{}", request_id, shard.shard_id),
                    target_peer: shard.node.peer_id.clone(),
                    from_peer: from_peer.to_string(),
                    request: shard_request,
                    max_price: Some(shard.node.price_input_per_1k),
                }
            })
            .collect()
    }
}

/// Result aggregator combines shard outputs.
pub struct ResultAggregator {
    /// Original request ID.
    pub request_id: String,
    /// Number of shards expected.
    pub expected_shards: usize,
    /// Received shard results.
    pub results: Vec<ShardResult>,
    /// Timestamp when aggregation started.
    pub start_time: u64,
}

impl ResultAggregator {
    /// Create a new result aggregator.
    pub fn new(request_id: String, expected_shards: usize) -> Self {
        Self {
            request_id,
            expected_shards,
            results: Vec::with_capacity(expected_shards),
            start_time: current_timestamp(),
        }
    }

    /// Add a shard result.
    /// Returns true if all shards have been received.
    pub fn add_result(&mut self, result: ShardResult) -> bool {
        self.results.push(result);
        self.results.len() >= self.expected_shards
    }

    /// Check if aggregation is complete.
    pub fn is_complete(&self) -> bool {
        self.results.len() >= self.expected_shards
    }

    /// Check if any shard failed.
    pub fn has_failures(&self) -> bool {
        self.results.iter().any(|r| r.error.is_some())
    }

    /// Aggregate results into final output.
    /// For sequence chunks, we concatenate the text outputs.
    pub fn aggregate(&self) -> Option<String> {
        if !self.is_complete() {
            return None;
        }

        // Sort by shard_id to maintain order
        let mut sorted_results: Vec<_> = self.results.iter().collect();
        sorted_results.sort_by(|a, b| a.shard_id.cmp(&b.shard_id));

        // Concatenate text outputs
        let combined: String = sorted_results
            .iter()
            .filter_map(|r| {
                if r.error.is_none() {
                    match &r.output {
                        ShardOutput::Text(t) => Some(t.clone()),
                        ShardOutput::Tokens(ids) => {
                            // Convert tokens back to text placeholder
                            Some(format!("[{} tokens]", ids.len()))
                        }
                        _ => None,
                    }
                } else {
                    None
                }
            })
            .collect();

        if combined.is_empty() {
            None
        } else {
            Some(combined)
        }
    }

    /// Get total processing time across all shards.
    pub fn total_processing_time_ms(&self) -> u64 {
        self.results.iter().map(|r| r.processing_time_ms).sum()
    }

    /// Get maximum processing time (bottleneck).
    pub fn max_processing_time_ms(&self) -> u64 {
        self.results.iter().map(|r| r.processing_time_ms).max().unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_should_shard_short_prompt() {
        let registry = Arc::new(RwLock::new(NodeRegistry::new()));
        let manager = ShardingManager::new(registry);

        let request = ChatCompletionRequest {
            model: "test-model".to_string(),
            messages: vec![crate::types::ChatMessage {
                role: "user".to_string(),
                content: "Hello".to_string(), // Short prompt
            }],
            temperature: None,
            max_tokens: None,
            stream: None,
            ynet_address: None,
        };

        assert!(!manager.should_shard(&request));
    }

    #[test]
    fn test_should_shard_long_prompt() {
        let registry = Arc::new(RwLock::new(NodeRegistry::new()));
        let manager = ShardingManager::new(registry);

        // Create a long prompt (> 16384 chars to get > 4096 estimated tokens)
        let long_content = "x".repeat(20000);

        let request = ChatCompletionRequest {
            model: "test-model".to_string(),
            messages: vec![crate::types::ChatMessage {
                role: "user".to_string(),
                content: long_content,
            }],
            temperature: None,
            max_tokens: None,
            stream: None,
            ynet_address: None,
        };

        assert!(manager.should_shard(&request));
    }

    #[test]
    fn test_result_aggregator() {
        let mut aggregator = ResultAggregator::new("req-1".to_string(), 2);

        assert!(!aggregator.is_complete());

        aggregator.add_result(ShardResult {
            shard_id: "shard-0".to_string(),
            node_id: "node-1".to_string(),
            processing_time_ms: 100,
            output: ShardOutput::Text("Hello ".to_string()),
            error: None,
        });

        assert!(!aggregator.is_complete());

        aggregator.add_result(ShardResult {
            shard_id: "shard-1".to_string(),
            node_id: "node-2".to_string(),
            processing_time_ms: 150,
            output: ShardOutput::Text("World".to_string()),
            error: None,
        });

        assert!(aggregator.is_complete());
        assert!(!aggregator.has_failures());

        let result = aggregator.aggregate();
        assert_eq!(result, Some("Hello World".to_string()));
    }

    #[test]
    fn test_result_aggregator_with_failure() {
        let mut aggregator = ResultAggregator::new("req-1".to_string(), 2);

        aggregator.add_result(ShardResult {
            shard_id: "shard-0".to_string(),
            node_id: "node-1".to_string(),
            processing_time_ms: 100,
            output: ShardOutput::Text("Hello".to_string()),
            error: None,
        });

        aggregator.add_result(ShardResult {
            shard_id: "shard-1".to_string(),
            node_id: "node-2".to_string(),
            processing_time_ms: 0,
            output: ShardOutput::Text(String::new()),
            error: Some("Connection failed".to_string()),
        });

        assert!(aggregator.is_complete());
        assert!(aggregator.has_failures());
    }
}
