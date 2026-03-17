//! Reputation tracking system for distributed inference.
//!
//! Tracks node performance and adjusts reputation scores based on behavior.

use crate::utils::current_timestamp;
use std::collections::VecDeque;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Maximum number of events to keep per node.
const MAX_EVENTS_PER_NODE: usize = 100;

/// Reputation points gained for successful request.
const REPUTATION_GAIN_SUCCESS: i32 = 1;

/// Reputation points lost for failed request.
const REPUTATION_LOSS_FAILURE: i32 = -5;

/// Reputation points lost for timeout.
const REPUTATION_LOSS_TIMEOUT: i32 = -3;

/// Reputation points lost for malicious behavior.
const REPUTATION_LOSS_MALICIOUS: i32 = -50;

/// How often to decay old reputation events (in seconds).
const DECAY_INTERVAL_SECS: u64 = 3600; // 1 hour

/// Decay factor for old reputation (multiplied by this).
const DECAY_FACTOR: f64 = 0.9;

/// Minimum reputation score.
const MIN_REPUTATION: i32 = 0;

/// Maximum reputation score.
const MAX_REPUTATION: i32 = 100;

/// Event record for reputation calculation.
#[derive(Debug, Clone)]
pub struct ReputationEvent {
    /// Timestamp of the event.
    pub timestamp: u64,
    /// Points change (positive or negative).
    pub delta: i32,
    /// Type of event.
    pub event_type: ReputationEventType,
}

/// Types of reputation events.
#[derive(Debug, Clone, Copy)]
pub enum ReputationEventType {
    /// Request completed successfully.
    Success,
    /// Request failed with error.
    Failure,
    /// Request timed out.
    Timeout,
    /// Malicious behavior detected.
    Malicious,
    /// Manual adjustment.
    Manual,
}

/// Reputation record for a single node.
#[derive(Debug, Clone)]
pub struct NodeReputation {
    /// Current reputation score (0-100).
    pub score: i32,
    /// Recent events (up to MAX_EVENTS_PER_NODE).
    pub events: VecDeque<ReputationEvent>,
    /// Last time reputation was recalculated.
    pub last_recalc: u64,
}

impl Default for NodeReputation {
    fn default() -> Self {
        Self {
            score: 50, // Start at neutral reputation
            events: VecDeque::with_capacity(MAX_EVENTS_PER_NODE),
            last_recalc: current_timestamp(),
        }
    }
}

/// Global reputation tracker for all nodes.
pub struct ReputationTracker {
    /// Reputation records per peer_id.
    records: HashMap<String, NodeReputation>,
}

impl Default for ReputationTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl ReputationTracker {
    /// Create a new reputation tracker.
    pub fn new() -> Self {
        Self {
            records: HashMap::new(),
        }
    }

    /// Record a successful request.
    pub fn record_success(&mut self, peer_id: &str) {
        self.add_event(peer_id, ReputationEvent {
            timestamp: current_timestamp(),
            delta: REPUTATION_GAIN_SUCCESS,
            event_type: ReputationEventType::Success,
        });
    }

    /// Record a failed request.
    pub fn record_failure(&mut self, peer_id: &str) {
        self.add_event(peer_id, ReputationEvent {
            timestamp: current_timestamp(),
            delta: REPUTATION_LOSS_FAILURE,
            event_type: ReputationEventType::Failure,
        });
    }

    /// Record a timed out request.
    pub fn record_timeout(&mut self, peer_id: &str) {
        self.add_event(peer_id, ReputationEvent {
            timestamp: current_timestamp(),
            delta: REPUTATION_LOSS_TIMEOUT,
            event_type: ReputationEventType::Timeout,
        });
    }

    /// Record malicious behavior.
    pub fn record_malicious(&mut self, peer_id: &str) {
        self.add_event(peer_id, ReputationEvent {
            timestamp: current_timestamp(),
            delta: REPUTATION_LOSS_MALICIOUS,
            event_type: ReputationEventType::Malicious,
        });
    }

    /// Manually adjust reputation.
    pub fn adjust_reputation(&mut self, peer_id: &str, delta: i32) {
        self.add_event(peer_id, ReputationEvent {
            timestamp: current_timestamp(),
            delta,
            event_type: ReputationEventType::Manual,
        });
    }

    /// Get the current reputation score for a node.
    pub fn get_reputation(&self, peer_id: &str) -> i32 {
        self.records.get(peer_id).map(|r| r.score).unwrap_or(50)
    }

    /// Get all nodes sorted by reputation (highest first).
    pub fn get_nodes_by_reputation(&self) -> Vec<(String, i32)> {
        let mut nodes: Vec<_> = self
            .records
            .iter()
            .map(|(k, v)| (k.clone(), v.score))
            .collect();
        nodes.sort_by(|a, b| b.1.cmp(&a.1));
        nodes
    }

    /// Add a reputation event.
    fn add_event(&mut self, peer_id: &str, event: ReputationEvent) {
        let record = self.records.entry(peer_id.to_string()).or_default();

        // Add event
        record.events.push_back(event.clone());

        // Trim old events if needed
        if record.events.len() > MAX_EVENTS_PER_NODE {
            record.events.pop_front();
        }

        // Update score (clamped)
        record.score = (record.score + event.delta).clamp(MIN_REPUTATION, MAX_REPUTATION);
    }

    /// Apply time-based decay to all nodes.
    pub fn apply_decay(&mut self) {
        let now = current_timestamp();

        for record in self.records.values_mut() {
            // Check if decay interval has passed
            if now - record.last_recalc < DECAY_INTERVAL_SECS {
                continue;
            }

            // Apply decay to old events
            let mut total_delta = 0;
            for event in record.events.iter_mut() {
                let age_hours = (now - event.timestamp) / 3600;
                if age_hours > 0 {
                    // Apply decay factor for each hour of age
                    let decay_multiplier = DECAY_FACTOR.powf(age_hours as f64);
                    event.delta = (event.delta as f64 * decay_multiplier) as i32;
                    total_delta += event.delta;
                }
            }

            // Recalculate score from decayed events
            record.score = (50 + total_delta).clamp(MIN_REPUTATION, MAX_REPUTATION);
            record.last_recalc = now;
        }
    }

    /// Clear all reputation data (for testing).
    #[cfg(test)]
    pub fn clear(&mut self) {
        self.records.clear();
    }
}

/// Shared reputation tracker.
pub type SharedReputationTracker = Arc<RwLock<ReputationTracker>>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reputation_success() {
        let mut tracker = ReputationTracker::new();

        tracker.record_success("node1");
        assert_eq!(tracker.get_reputation("node1"), 51);

        tracker.record_success("node1");
        assert_eq!(tracker.get_reputation("node1"), 52);
    }

    #[test]
    fn test_reputation_failure() {
        let mut tracker = ReputationTracker::new();

        tracker.record_failure("node1");
        assert_eq!(tracker.get_reputation("node1"), 45);

        tracker.record_timeout("node1");
        assert_eq!(tracker.get_reputation("node1"), 42);
    }

    #[test]
    fn test_reputation_malicious() {
        let mut tracker = ReputationTracker::new();

        tracker.record_malicious("node1");
        assert_eq!(tracker.get_reputation("node1"), 0); // Clamped to 0
    }

    #[test]
    fn test_reputation_clamp() {
        let mut tracker = ReputationTracker::new();

        // Try to go below 0
        for _ in 0..20 {
            tracker.record_failure("node1");
        }
        assert_eq!(tracker.get_reputation("node1"), 0); // Clamped to 0

        // Try to go above 100
        for _ in 0..100 {
            tracker.record_success("node1");
        }
        assert_eq!(tracker.get_reputation("node1"), 100); // Clamped to 100
    }

    #[test]
    fn test_get_nodes_by_reputation() {
        let mut tracker = ReputationTracker::new();

        tracker.record_success("node1"); // 51
        tracker.record_success("node1"); // 52
        tracker.record_success("node2"); // 51
        tracker.record_failure("node3"); // 45

        let sorted = tracker.get_nodes_by_reputation();
        assert_eq!(sorted.len(), 3);
        assert_eq!(sorted[0].0, "node1"); // Highest (52)
        assert_eq!(sorted[1].0, "node2"); // Second (51)
        assert_eq!(sorted[2].0, "node3"); // Lowest (45)
    }
}
