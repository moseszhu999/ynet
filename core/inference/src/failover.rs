//! Failover handling for distributed inference.
//!
//! Provides automatic failover when nodes become unavailable or unresponsive.

use crate::registry::NodeRegistry;
use crate::router::{InferenceRouter, RoutingPreferences};
use crate::types::{ChatCompletionRequest, NodeInfo};
use crate::utils::current_timestamp;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;

/// Maximum number of retry attempts before giving up.
const MAX_RETRIES: usize = 3;

/// Request timeout in seconds.
const REQUEST_TIMEOUT_SECS: u64 = 30;

/// How long to consider a node "failed" after an error.
const NODE_FAILURE_COOLDOWN_SECS: u64 = 60;

/// Health status of a node.
#[derive(Debug, Clone)]
pub struct NodeHealth {
    /// Number of consecutive failures.
    pub consecutive_failures: u32,
    /// Last failure timestamp.
    pub last_failure: Option<u64>,
    /// Whether the node is currently marked as failed.
    pub is_failed: bool,
    /// Last successful response timestamp.
    pub last_success: Option<u64>,
}

impl Default for NodeHealth {
    fn default() -> Self {
        Self {
            consecutive_failures: 0,
            last_failure: None,
            is_failed: false,
            last_success: None,
        }
    }
}

/// Tracks node health for failover decisions.
pub struct FailoverManager {
    /// Health status per peer_id.
    health: HashMap<String, NodeHealth>,
    /// Router for selecting backup nodes.
    router: InferenceRouter,
}

impl FailoverManager {
    /// Create a new failover manager.
    pub fn new(registry: Arc<RwLock<NodeRegistry>>) -> Self {
        let router = InferenceRouter::new(registry);
        Self {
            health: HashMap::new(),
            router,
        }
    }

    /// Get a healthy node for the request, with fallback options.
    /// Returns (primary_node, backup_nodes).
    pub async fn select_node_with_fallback(
        &mut self,
        request: &ChatCompletionRequest,
        preferences: Option<&RoutingPreferences>,
    ) -> Option<(NodeInfo, Vec<NodeInfo>)> {
        let now = crate::utils::current_timestamp();

        // Clear expired failure states
        self.clear_expired_failures(now);

        // Get all candidates from router
        let all_candidates = self.router.get_available_nodes(&request.model, None).await;

        if all_candidates.is_empty() {
            log::warn!("No nodes available for model {}", request.model);
            return None;
        }

        // Filter out failed nodes
        let (healthy, failed): (Vec<_>, Vec<_>) = all_candidates
            .into_iter()
            .partition(|node| self.is_node_healthy(&node.peer_id, now));

        if healthy.is_empty() {
            log::warn!(
                "All {} nodes for model {} are marked as failed",
                failed.len(),
                request.model
            );
            // If all nodes are failed, try the least-recently-failed one
            return self.select_least_failed(&failed, now);
        }

        // Sort healthy nodes by score
        let prefs = preferences.cloned().unwrap_or_default();
        let mut sorted_healthy: Vec<_> = healthy
            .into_iter()
            .map(|node| {
                let score = InferenceRouter::calculate_score(&node, &prefs);
                (node, score)
            })
            .collect();
        sorted_healthy.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let primary = sorted_healthy.first()?.0.clone();
        let backups: Vec<NodeInfo> = sorted_healthy.iter().skip(1).map(|(n, _)| n.clone()).collect();

        Some((primary, backups))
    }

    /// Mark a node as failed.
    pub fn mark_node_failed(&mut self, peer_id: &str) {
        let now = crate::utils::current_timestamp();
        let health = self.health.entry(peer_id.to_string()).or_default();
        health.consecutive_failures += 1;
        health.last_failure = Some(now);
        health.is_failed = true;

        log::warn!(
            "Node {} marked as failed (consecutive failures: {})",
            peer_id,
            health.consecutive_failures
        );
    }

    /// Mark a node as successful.
    pub fn mark_node_success(&mut self, peer_id: &str) {
        let now = crate::utils::current_timestamp();
        let health = self.health.entry(peer_id.to_string()).or_default();
        health.consecutive_failures = 0;
        health.last_success = Some(now);
        health.is_failed = false;

        log::debug!("Node {} marked as healthy", peer_id);
    }

    /// Check if a node is currently healthy.
    fn is_node_healthy(&self, peer_id: &str, now: u64) -> bool {
        match self.health.get(peer_id) {
            Some(health) => {
                if health.is_failed {
                    // Check if cooldown has expired
                    if let Some(last_failure) = health.last_failure {
                        now - last_failure >= NODE_FAILURE_COOLDOWN_SECS
                    } else {
                        true
                    }
                } else {
                    true
                }
            }
            None => true,
        }
    }

    /// Select the least recently failed node when all are marked as failed.
    #[allow(unused_variables)]
    fn select_least_failed(&self, failed: &[NodeInfo], _now: u64) -> Option<(NodeInfo, Vec<NodeInfo>)> {
        let sorted: Vec<_> = failed
            .iter()
            .map(|node| {
                let last_failure = self.health
                    .get(&node.peer_id)
                    .and_then(|h| h.last_failure)
                    .unwrap_or(0);
                (node.clone(), last_failure)
            })
            .collect();

        // Sort by oldest failure (smallest timestamp = longest ago)
        let mut sorted: Vec<_> = sorted.into_iter().collect();
        sorted.sort_by(|a, b| a.1.cmp(&b.1));

        let primary = sorted.first()?.0.clone();
        let backups: Vec<NodeInfo> = sorted.iter().skip(1).map(|(n, _)| n.clone()).collect();

        Some((primary, backups))
    }

    /// Clear expired failure states.
    fn clear_expired_failures(&mut self, now: u64) {
        for health in self.health.values_mut() {
            if health.is_failed {
                if let Some(last_failure) = health.last_failure {
                    if now - last_failure >= NODE_FAILURE_COOLDOWN_SECS {
                        health.is_failed = false;
                        log::debug!("Node failure state cleared after cooldown");
                    }
                }
            }
        }
    }

    /// Get health statistics.
    pub fn health_stats(&self) -> FailoverStats {
        let total = self.health.len();
        let failed = self.health.values().filter(|h| h.is_failed).count();
        let healthy = total - failed;

        FailoverStats {
            total_nodes: total,
            healthy_nodes: healthy,
            failed_nodes: failed,
        }
    }
}

/// Statistics about node health.
#[derive(Debug, Clone)]
pub struct FailoverStats {
    pub total_nodes: usize,
    pub healthy_nodes: usize,
    pub failed_nodes: usize,
}

/// Result of a failover attempt.
#[derive(Debug, Clone)]
pub enum FailoverResult {
    /// Request succeeded on this node.
    Success { peer_id: String },
    /// Request failed, should retry with next node.
    Failed { peer_id: String, error: String },
    /// All nodes exhausted, no more retries.
    Exhausted { last_error: String },
}

/// Executes a request with automatic failover.
pub struct FailoverExecutor {
    manager: Arc<RwLock<FailoverManager>>,
    max_retries: usize,
    timeout: Duration,
}

impl FailoverExecutor {
    /// Create a new failover executor.
    pub fn new(manager: Arc<RwLock<FailoverManager>>) -> Self {
        Self {
            manager,
            max_retries: MAX_RETRIES,
            timeout: Duration::from_secs(REQUEST_TIMEOUT_SECS),
        }
    }

    /// Set custom max retries.
    pub fn with_max_retries(mut self, max_retries: usize) -> Self {
        self.max_retries = max_retries;
        self
    }

    /// Set custom timeout.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Get the number of allowed retries.
    pub fn max_retries(&self) -> usize {
        self.max_retries
    }

    /// Get the request timeout.
    pub fn timeout(&self) -> Duration {
        self.timeout
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_health_default() {
        let health = NodeHealth::default();
        assert_eq!(health.consecutive_failures, 0);
        assert!(!health.is_failed);
    }

    #[test]
    fn test_mark_node_failed() {
        let registry = Arc::new(RwLock::new(NodeRegistry::new()));
        let mut manager = FailoverManager::new(registry);

        manager.mark_node_failed("peer1");

        let stats = manager.health_stats();
        assert_eq!(stats.failed_nodes, 1);
        assert_eq!(stats.healthy_nodes, 0);
    }

    #[test]
    fn test_mark_node_success() {
        let registry = Arc::new(RwLock::new(NodeRegistry::new()));
        let mut manager = FailoverManager::new(registry);

        manager.mark_node_failed("peer1");
        manager.mark_node_success("peer1");

        let stats = manager.health_stats();
        assert_eq!(stats.failed_nodes, 0);
        assert_eq!(stats.healthy_nodes, 1);
    }

    #[test]
    fn test_failover_stats() {
        let registry = Arc::new(RwLock::new(NodeRegistry::new()));
        let mut manager = FailoverManager::new(registry);

        manager.mark_node_failed("peer1");
        manager.mark_node_failed("peer2");
        manager.mark_node_success("peer3");

        let stats = manager.health_stats();
        assert_eq!(stats.total_nodes, 3);
        assert_eq!(stats.failed_nodes, 2);
        assert_eq!(stats.healthy_nodes, 1);
    }
}
