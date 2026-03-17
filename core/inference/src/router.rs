//! Smart inference router — selects optimal nodes for distributed inference.
//!
//! The router evaluates available nodes based on multiple factors:
//! - Price: Cost per 1K tokens
//! - Latency: Average response time
//! - Load: Current queue depth / max capacity
//! - Reputation: Historical reliability score

use crate::types::{ChatCompletionRequest, NodeInfo};
use crate::registry::NodeRegistry;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Routing preferences for inference requests.
#[derive(Debug, Clone, Default)]
pub struct RoutingPreferences {
    /// Maximum price per 1K input tokens (nYNET). None = no limit.
    pub max_price_per_1k: Option<u64>,
    /// Price vs latency trade-off (0.0 = cheapest, 1.0 = fastest).
    pub latency_priority: f32,
    /// Minimum reputation score required (0-100). None = no minimum.
    pub min_reputation: Option<u32>,
    /// Maximum acceptable load (0.0-1.0). None = no limit.
    pub max_load: Option<f32>,
}

/// Result of a routing decision.
#[derive(Debug, Clone)]
pub struct RoutingDecision {
    /// Selected node information.
    pub node: NodeInfo,
    /// Calculated score (lower is better).
    pub score: f64,
    /// Number of candidates considered.
    pub candidates_count: usize,
    /// Reason for selection.
    pub reason: RoutingReason,
}

/// Why a particular node was selected.
#[derive(Debug, Clone, PartialEq)]
pub enum RoutingReason {
    /// Lowest price among candidates.
    LowestPrice,
    /// Lowest latency among candidates.
    LowestLatency,
    /// Best overall score.
    BestOverallScore,
    /// Only available candidate.
    OnlyCandidate,
    /// Highest reputation.
    HighestReputation,
}

/// Smart router for distributed inference.
pub struct InferenceRouter {
    /// Shared node registry.
    registry: Arc<RwLock<NodeRegistry>>,
    /// Default routing preferences.
    default_preferences: RoutingPreferences,
}

impl InferenceRouter {
    /// Create a new router with the given registry.
    pub fn new(registry: Arc<RwLock<NodeRegistry>>) -> Self {
        Self {
            registry,
            default_preferences: RoutingPreferences::default(),
        }
    }

    /// Set default routing preferences.
    pub fn with_default_preferences(mut self, prefs: RoutingPreferences) -> Self {
        self.default_preferences = prefs;
        self
    }

    /// Find the best node for a given inference request.
    ///
    /// Returns `None` if no suitable nodes are available.
    pub async fn find_best_node(
        &self,
        request: &ChatCompletionRequest,
        preferences: Option<&RoutingPreferences>,
    ) -> Option<RoutingDecision> {
        let prefs = preferences.unwrap_or(&self.default_preferences);
        let registry = self.registry.read().await;

        // Get candidates from registry
        let candidates = registry.find_nodes_with_pricing(
            &request.model,
            prefs.max_price_per_1k,
        );

        if candidates.is_empty() {
            log::warn!("No nodes available for model {}", request.model);
            return None;
        }

        // Filter by additional criteria
        let filtered: Vec<&NodeInfo> = candidates
            .iter()
            .filter(|node| {
                // Reputation filter
                if let Some(min_rep) = prefs.min_reputation {
                    if node.reputation < min_rep {
                        return false;
                    }
                }
                // Load filter
                if let Some(max_load) = prefs.max_load {
                    if node.load > max_load {
                        return false;
                    }
                }
                true
            })
            .collect();

        if filtered.is_empty() {
            log::warn!(
                "All {} candidates filtered out for model {}",
                candidates.len(),
                request.model
            );
            return None;
        }

        // Score and rank candidates
        let scored: Vec<(&NodeInfo, f64, RoutingReason)> = filtered
            .iter()
            .map(|node| {
                let score = Self::calculate_score(node, prefs);
                let reason = Self::determine_reason(node, &filtered);
                (*node, score, reason)
            })
            .collect();

        // Select best (lowest score)
        let (best_node, best_score, reason) = scored
            .into_iter()
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))?;

        Some(RoutingDecision {
            node: best_node.clone(),
            score: best_score,
            candidates_count: filtered.len(),
            reason,
        })
    }

    /// Calculate routing score for a node (lower is better).
    pub fn calculate_score(node: &NodeInfo, prefs: &RoutingPreferences) -> f64 {
        // Normalize factors to 0-1 range
        let price_score = (node.price_input_per_1k as f64) / 10000.0; // assuming max ~10000 nYNET
        let latency_score = (node.avg_latency_ms as f64) / 2000.0;     // assuming max ~2000ms
        let load_score = node.load as f64;
        let reputation_score = 1.0 - (node.reputation as f64 / 100.0); // inverse

        // Weighted combination based on latency_priority
        let price_weight = (1.0 - prefs.latency_priority) as f64;
        let latency_weight = prefs.latency_priority as f64;

        // Fixed weights for load and reputation
        let load_weight = 0.2_f64;
        let reputation_weight = 0.1_f64;

        // Normalize weights
        let total_weight = price_weight + latency_weight + load_weight + reputation_weight;

        (price_weight * price_score
            + latency_weight * latency_score
            + load_weight * load_score
            + reputation_weight * reputation_score)
            / total_weight
    }

    /// Calculate routing score with default preferences.
    /// This is a convenience function for simple use cases.
    pub fn calculate_score_default(node: &NodeInfo) -> f64 {
        Self::calculate_score(node, &RoutingPreferences::default())
    }

    /// Determine the primary reason for selecting a node.
    fn determine_reason(selected: &NodeInfo, all_candidates: &[&NodeInfo]) -> RoutingReason {
        if all_candidates.len() == 1 {
            return RoutingReason::OnlyCandidate;
        }

        // Check if selected is strictly the best in each category
        let is_strictly_lowest_price = all_candidates
            .iter()
            .all(|n| n.price_input_per_1k >= selected.price_input_per_1k)
            && all_candidates
                .iter()
                .any(|n| n.price_input_per_1k > selected.price_input_per_1k);

        let is_strictly_lowest_latency = all_candidates
            .iter()
            .all(|n| n.avg_latency_ms >= selected.avg_latency_ms)
            && all_candidates
                .iter()
                .any(|n| n.avg_latency_ms > selected.avg_latency_ms);

        let is_strictly_highest_reputation = all_candidates
            .iter()
            .all(|n| n.reputation <= selected.reputation)
            && all_candidates
                .iter()
                .any(|n| n.reputation < selected.reputation);

        // Priority: price > latency > reputation > overall
        if is_strictly_lowest_price && is_strictly_lowest_latency {
            RoutingReason::BestOverallScore
        } else if is_strictly_lowest_price {
            RoutingReason::LowestPrice
        } else if is_strictly_lowest_latency {
            RoutingReason::LowestLatency
        } else if is_strictly_highest_reputation {
            RoutingReason::HighestReputation
        } else {
            RoutingReason::BestOverallScore
        }
    }

    /// Get all available nodes for a model.
    pub async fn get_available_nodes(
        &self,
        model_id: &str,
        max_price: Option<u64>,
    ) -> Vec<NodeInfo> {
        let registry = self.registry.read().await;
        registry.find_nodes_with_pricing(model_id, max_price)
    }

    /// Estimate cost for a request.
    pub fn estimate_cost(
        node: &NodeInfo,
        input_tokens: u32,
        output_tokens: u32,
    ) -> u64 {
        let input_cost = (input_tokens as u64 * node.price_input_per_1k) / 1000;
        let output_cost = (output_tokens as u64 * node.price_output_per_1k) / 1000;
        input_cost + output_cost
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_node(
        peer_id: &str,
        price: u64,
        latency_ms: u32,
        load: f32,
        reputation: u32,
    ) -> NodeInfo {
        NodeInfo {
            peer_id: peer_id.to_string(),
            address: format!("ynet1{}", peer_id),
            model_id: "test-model".to_string(),
            price_input_per_1k: price,
            price_output_per_1k: price / 2,
            avg_latency_ms: latency_ms,
            load,
            reputation,
        }
    }

    #[test]
    fn test_calculate_score() {
        let prefs = RoutingPreferences {
            latency_priority: 0.5,
            ..Default::default()
        };

        // Low price, low latency → low score (good)
        let good_node = make_node("good", 100, 50, 0.1, 90);
        let good_score = InferenceRouter::calculate_score(&good_node, &prefs);

        // High price, high latency → high score (bad)
        let bad_node = make_node("bad", 1000, 500, 0.9, 10);
        let bad_score = InferenceRouter::calculate_score(&bad_node, &prefs);

        assert!(good_score < bad_score, "Good node should have lower score");
    }

    #[test]
    fn test_determine_reason() {
        let cheap = make_node("cheap", 100, 200, 0.3, 50);
        let fast = make_node("fast", 200, 50, 0.3, 50);
        let trusted = make_node("trusted", 200, 200, 0.3, 99);

        // Only candidate
        assert_eq!(
            InferenceRouter::determine_reason(&cheap, &[&cheap]),
            RoutingReason::OnlyCandidate
        );

        // Lowest price
        let candidates: Vec<&NodeInfo> = vec![&cheap, &fast];
        assert_eq!(
            InferenceRouter::determine_reason(&cheap, &candidates),
            RoutingReason::LowestPrice
        );

        // Highest reputation
        let candidates: Vec<&NodeInfo> = vec![&cheap, &trusted];
        assert_eq!(
            InferenceRouter::determine_reason(&trusted, &candidates),
            RoutingReason::HighestReputation
        );
    }

    #[test]
    fn test_estimate_cost() {
        let node = make_node("test", 100, 50, 0.1, 50);

        // 1000 input + 500 output tokens
        let cost = InferenceRouter::estimate_cost(&node, 1000, 500);

        // Input: 1000 * 100 / 1000 = 100
        // Output: 500 * 50 / 1000 = 25
        assert_eq!(cost, 125);
    }
}
