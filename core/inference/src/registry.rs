//! Node capability registry — tracks which nodes can serve which models.
//! Nodes broadcast their capabilities periodically via P2P.
//! The registry maintains a network-wide view for smart routing.

use crate::types::NodeCapability;
use std::collections::HashMap;

/// How long a capability announcement is valid (seconds).
const CAPABILITY_TTL_SECS: u64 = 30;

/// Registry of inference-capable nodes across the network.
pub struct NodeRegistry {
    /// peer_id → latest capability announcement.
    nodes: HashMap<String, NodeCapability>,
}

impl NodeRegistry {
    pub fn new() -> Self {
        NodeRegistry {
            nodes: HashMap::new(),
        }
    }

    /// Update registry with a capability announcement from a peer.
    pub fn update(&mut self, cap: NodeCapability) {
        self.nodes.insert(cap.peer_id.clone(), cap);
    }

    /// Remove stale entries (no announcement within TTL).
    pub fn evict_stale(&mut self) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        self.nodes.retain(|_, cap| now - cap.timestamp < CAPABILITY_TTL_SECS);
    }

    /// Find nodes that have a given model loaded, sorted by best candidate first.
    /// Criteria: lowest queue_depth, then highest tokens_per_sec.
    pub fn find_nodes_for_model(&self, model_id: &str) -> Vec<&NodeCapability> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

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

    /// List all available models across the network (deduplicated).
    pub fn available_models(&self) -> Vec<(String, u32)> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

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

    /// Total number of active inference nodes.
    pub fn active_node_count(&self) -> usize {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        self.nodes
            .values()
            .filter(|cap| now - cap.timestamp < CAPABILITY_TTL_SECS)
            .count()
    }

    /// Get a specific node's capability.
    pub fn get_node(&self, peer_id: &str) -> Option<&NodeCapability> {
        self.nodes.get(peer_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{GpuInfo, InferenceBackend, LoadedModel};

    fn make_cap(peer_id: &str, models: Vec<&str>, queue: u32, tps: f32) -> NodeCapability {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        NodeCapability {
            address: format!("ynet1{}", peer_id),
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
