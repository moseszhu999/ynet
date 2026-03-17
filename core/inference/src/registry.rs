//! Node capability registry — tracks which nodes can serve which models.
//! Nodes broadcast their capabilities periodically via P2P.
//! The registry maintains a network-wide view for smart routing.

use crate::types::{NodeCapability, NodeAnnouncement, NodeInfo, ModelPricing};
use crate::utils::current_timestamp;
use crate::router::InferenceRouter;
use std::collections::HashMap;

/// How long a capability announcement is valid (seconds).
const CAPABILITY_TTL_SECS: u64 = 30;

/// Registry of inference-capable nodes across the network.
pub struct NodeRegistry {
    /// peer_id → latest capability announcement (Phase 1).
    nodes: HashMap<String, NodeCapability>,
    /// peer_id → full node announcement (Phase 2).
    announcements: HashMap<String, NodeAnnouncement>,
}

impl Default for NodeRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl NodeRegistry {
    pub fn new() -> Self {
        NodeRegistry {
            nodes: HashMap::new(),
            announcements: HashMap::new(),
        }
    }

    /// Update registry with a capability announcement from a peer (Phase 1).
    pub fn update(&mut self, cap: NodeCapability) {
        self.nodes.insert(cap.peer_id.clone(), cap);
    }

    /// Update registry with a full node announcement (Phase 2).
    pub fn update_announcement(&mut self, ann: NodeAnnouncement) {
        self.announcements.insert(ann.peer_id.clone(), ann);
    }

    /// Remove stale entries (no announcement within TTL).
    pub fn evict_stale(&mut self) {
        let now = current_timestamp();

        self.nodes.retain(|_, cap| now - cap.timestamp < CAPABILITY_TTL_SECS);
        self.announcements.retain(|_, ann| now - ann.timestamp < CAPABILITY_TTL_SECS);
    }

    /// Find nodes that have a given model loaded, sorted by best candidate first.
    /// Criteria: lowest queue_depth, then highest tokens_per_sec.
    pub fn find_nodes_for_model(&self, model_id: &str) -> Vec<&NodeCapability> {
        let now = current_timestamp();

        let mut candidates: Vec<&NodeCapability> = self
            .nodes
            .values()
            .filter(|cap| {
                // Not stale
                now - cap.timestamp < CAPABILITY_TTL_SECS
                    // Has the model loaded
                    && cap.loaded_models.iter().any(|m| m.model_id == model_id)
                    // Has capacity
                    && cap.queue_depth < cap.max_concurrent
            })
            .collect();

        // Sort: lowest queue first, then highest throughput
        candidates.sort_by(|a, b| {
            a.queue_depth
                .cmp(&b.queue_depth)
                .then_with(|| {
                    let a_tps = a.loaded_models.iter()
                        .find(|m| m.model_id == model_id)
                        .map(|m| m.tokens_per_sec)
                        .unwrap_or(0.0);
                    let b_tps = b.loaded_models.iter()
                        .find(|m| m.model_id == model_id)
                        .map(|m| m.tokens_per_sec)
                        .unwrap_or(0.0);
                    b_tps.partial_cmp(&a_tps).unwrap_or(std::cmp::Ordering::Equal)
                })
        });

        candidates
    }

    // ========== Phase 2: Enhanced routing ==========

    /// Find nodes with pricing information (Phase 2).
    /// Returns NodeInfo suitable for routing decisions.
    pub fn find_nodes_with_pricing(&self, model_id: &str, max_price_per_1k: Option<u64>) -> Vec<NodeInfo> {
        let now = current_timestamp();

        let mut nodes: Vec<NodeInfo> = Vec::new();

        for ann in self.announcements.values() {
            if now - ann.timestamp >= CAPABILITY_TTL_SECS {
                continue;
            }
            if ann.load >= 0.9 {
                continue; // Skip overloaded nodes
            }

            for model in &ann.models {
                if model.model_id != model_id {
                    continue;
                }

                // Price filter
                if let Some(max_price) = max_price_per_1k {
                    if model.price_input_per_1k > max_price {
                        continue;
                    }
                }

                nodes.push(NodeInfo {
                    peer_id: ann.peer_id.clone(),
                    address: ann.address.clone(),
                    model_id: model.model_id.clone(),
                    price_input_per_1k: model.price_input_per_1k,
                    price_output_per_1k: model.price_output_per_1k,
                    avg_latency_ms: model.avg_latency_ms,
                    load: ann.load,
                    reputation: ann.reputation,
                });
            }
        }

        // Sort by composite score: price + latency + load + reputation
        nodes.sort_by(|a, b| {
            let score_a = Self::calculate_route_score(a);
            let score_b = Self::calculate_route_score(b);
            score_a.partial_cmp(&score_b).unwrap_or(std::cmp::Ordering::Equal)
        });

        nodes
    }

    /// Calculate routing score (lower is better).
    /// Delegates to the unified scoring function in InferenceRouter.
    fn calculate_route_score(node: &NodeInfo) -> f64 {
        InferenceRouter::calculate_score_default(node)
    }

    /// List all available models across the network (deduplicated).
    pub fn available_models(&self) -> Vec<(String, u32)> {
        let now = current_timestamp();

        let mut model_counts: HashMap<String, u32> = HashMap::new();
        for cap in self.nodes.values() {
            if now - cap.timestamp >= CAPABILITY_TTL_SECS {
                continue;
            }
            for model in &cap.loaded_models {
                *model_counts.entry(model.model_id.clone()).or_default() += 1;
            }
        }

        let mut models: Vec<_> = model_counts.into_iter().collect();
        models.sort_by(|a, b| b.1.cmp(&a.1));
        models
    }

    /// List available models with pricing (Phase 2).
    pub fn available_models_with_pricing(&self) -> Vec<ModelPricing> {
        let now = current_timestamp();

        let mut models: HashMap<String, ModelPricing> = HashMap::new();

        for ann in self.announcements.values() {
            if now - ann.timestamp >= CAPABILITY_TTL_SECS {
                continue;
            }
            for model in &ann.models {
                // Keep the lowest price for each model
                match models.get_mut(&model.model_id) {
                    Some(existing) => {
                        if model.price_input_per_1k < existing.price_input_per_1k {
                            *existing = model.clone();
                        }
                    }
                    None => {
                        models.insert(model.model_id.clone(), model.clone());
                    }
                }
            }
        }

        let mut result: Vec<_> = models.into_values().collect();
        result.sort_by(|a, b| a.model_id.cmp(&b.model_id));
        result
    }

    /// Total number of active inference nodes.
    pub fn active_node_count(&self) -> usize {
        let now = current_timestamp();
        self.nodes
            .values()
            .filter(|cap| now - cap.timestamp < CAPABILITY_TTL_SECS)
            .count()
    }

    /// Get a specific node's capability.
    pub fn get_node(&self, peer_id: &str) -> Option<&NodeCapability> {
        self.nodes.get(peer_id)
    }

    /// Get a specific node's announcement (Phase 2).
    pub fn get_announcement(&self, peer_id: &str) -> Option<&NodeAnnouncement> {
        self.announcements.get(peer_id)
    }

    /// Get all available nodes as NodeInfo (Phase 2).
    /// Useful for responding to NodeListRequest.
    pub fn get_all_nodes_info(&self) -> Vec<NodeInfo> {
        let now = current_timestamp();

        self.announcements
            .values()
            .filter(|ann| now - ann.timestamp < CAPABILITY_TTL_SECS)
            .flat_map(|ann| {
                ann.models.iter().map(|m| NodeInfo {
                    peer_id: ann.peer_id.clone(),
                    address: ann.address.clone(),
                    model_id: m.model_id.clone(),
                    price_input_per_1k: m.price_input_per_1k,
                    price_output_per_1k: m.price_output_per_1k,
                    avg_latency_ms: m.avg_latency_ms,
                    load: ann.load,
                    reputation: ann.reputation,
                })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{GpuInfo, InferenceBackend, LoadedModel};

    fn make_cap(peer_id: &str, models: Vec<&str>, queue: u32, tps: f32) -> NodeCapability {
        let now = current_timestamp();

        NodeCapability {
            address: format!("ynet1{peer_id}"),
            peer_id: peer_id.to_string(),
            gpus: vec![GpuInfo {
                name: "RTX 4090".to_string(),
                vram_mb: 24576,
            }],
            vram_total_mb: 24576,
            loaded_models: models
                .into_iter()
                .map(|m| LoadedModel {
                    model_id: m.to_string(),
                    backend: InferenceBackend::LlamaCpp,
                    port: 8080,
                    max_context: 4096,
                    tokens_per_sec: tps,
                })
                .collect(),
            max_concurrent: 4,
            queue_depth: queue,
            timestamp: now,
        }
    }

    #[test]
    fn test_find_nodes_for_model() {
        let mut reg = NodeRegistry::new();

        reg.update(make_cap("peer1", vec!["kimi-k2.5"], 2, 30.0));
        reg.update(make_cap("peer2", vec!["kimi-k2.5", "qwen2-72b"], 0, 25.0));
        reg.update(make_cap("peer3", vec!["qwen2-72b"], 1, 40.0));

        let nodes = reg.find_nodes_for_model("kimi-k2.5");
        assert_eq!(nodes.len(), 2);
        // peer2 should be first (queue=0 vs queue=2)
        assert_eq!(nodes[0].peer_id, "peer2");
        assert_eq!(nodes[1].peer_id, "peer1");

        let nodes = reg.find_nodes_for_model("qwen2-72b");
        assert_eq!(nodes.len(), 2);
        // peer2 first (queue=0), then peer3 (queue=1)
        assert_eq!(nodes[0].peer_id, "peer2");
    }

    #[test]
    fn test_available_models() {
        let mut reg = NodeRegistry::new();
        reg.update(make_cap("p1", vec!["kimi-k2.5"], 0, 30.0));
        reg.update(make_cap("p2", vec!["kimi-k2.5", "qwen2-72b"], 0, 25.0));
        reg.update(make_cap("p3", vec!["llama3-70b"], 0, 20.0));

        let models = reg.available_models();
        assert_eq!(models.len(), 3);
        // kimi-k2.5 has 2 nodes, should be first
        assert_eq!(models[0].0, "kimi-k2.5");
        assert_eq!(models[0].1, 2);
    }

    #[test]
    fn test_full_queue_excluded() {
        let mut reg = NodeRegistry::new();
        // Node with max_concurrent=4 but queue_depth=4 → full
        let mut cap = make_cap("peer1", vec!["kimi-k2.5"], 4, 30.0);
        cap.max_concurrent = 4;
        reg.update(cap);

        let nodes = reg.find_nodes_for_model("kimi-k2.5");
        assert_eq!(nodes.len(), 0); // excluded because full
    }
}
