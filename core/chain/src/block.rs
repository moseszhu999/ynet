//! Block structure — deterministic random producer election, ~3s blocks.
//! Each time slot, score = hash(prev_hash + address + slot). Lowest score wins.

use crate::economics::TARGET_BLOCK_TIME_SECS;
use crate::transaction::Transaction;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Block {
    pub index: u64,
    pub timestamp: u64,
    pub slot: u64,
    pub prev_hash: String,
    pub hash: String,
    pub producer: String,
    pub election_score: String,
    pub transactions: Vec<Transaction>,
}

impl Block {
    /// Create the genesis block.
    pub fn genesis() -> Self {
        let mut block = Block {
            index: 0,
            timestamp: 0,
            slot: 0,
            prev_hash: "0".repeat(64),
            hash: String::new(),
            producer: "genesis".to_string(),
            election_score: "0".repeat(64),
            transactions: vec![],
        };
        block.hash = block.calculate_hash();
        block
    }

    /// Create and seal a new block.
    pub fn new(index: u64, prev_hash: &str, producer: &str, transactions: Vec<Transaction>) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let slot = timestamp / TARGET_BLOCK_TIME_SECS;
        let election_score = Self::compute_election_score(prev_hash, producer, slot);

        let mut block = Block {
            index,
            timestamp,
            slot,
            prev_hash: prev_hash.to_string(),
            hash: String::new(),
            producer: producer.to_string(),
            election_score,
            transactions,
        };
        block.hash = block.calculate_hash();
        block
    }

    pub fn calculate_hash(&self) -> String {
        let tx_data: String = self
            .transactions
            .iter()
            .map(|tx| tx.hash.as_str())
            .collect::<Vec<_>>()
            .join(",");

        let data = format!(
            "{}{}{}{}{}{}",
            self.index, self.timestamp, self.prev_hash, tx_data, self.producer, self.slot
        );
        let mut hasher = Sha256::new();
        hasher.update(data.as_bytes());
        hex::encode(hasher.finalize())
    }

    /// Verify the block's hash and election score are correct.
    pub fn is_valid(&self) -> bool {
        let valid_hash = self.calculate_hash() == self.hash;
        // Genesis block has special election score
        if self.index == 0 {
            return valid_hash;
        }
        let valid_score = Self::compute_election_score(&self.prev_hash, &self.producer, self.slot)
            == self.election_score;
        valid_hash && valid_score
    }

    /// Compute election score: hash(prev_hash + address + slot).
    /// Lower score = higher priority to produce.
    /// Deterministic — every node computes the same result.
    pub fn compute_election_score(prev_hash: &str, address: &str, slot: u64) -> String {
        let data = format!("{prev_hash}{address}{slot}");
        let mut hasher = Sha256::new();
        hasher.update(data.as_bytes());
        hex::encode(hasher.finalize())
    }

    /// Get the current time slot number.
    pub fn current_slot() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
            / TARGET_BLOCK_TIME_SECS
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_genesis_block() {
        let genesis = Block::genesis();
        assert_eq!(genesis.index, 0);
        assert!(!genesis.hash.is_empty());
        assert!(genesis.is_valid());
    }

    #[test]
    fn test_new_block() {
        let block = Block::new(1, &"0".repeat(64), "producer1", vec![]);
        assert!(!block.hash.is_empty());
        assert!(block.is_valid());
        assert!(!block.election_score.is_empty());
    }

    #[test]
    fn test_block_validity() {
        let mut block = Block::new(1, &"0".repeat(64), "producer1", vec![]);
        assert!(block.is_valid());
        block.hash = "bad_hash".to_string();
        assert!(!block.is_valid());
    }

    #[test]
    fn test_election_score_deterministic() {
        let prev = "a".repeat(64);
        let s1 = Block::compute_election_score(&prev, "nodeA", 100);
        let s2 = Block::compute_election_score(&prev, "nodeA", 100);
        assert_eq!(s1, s2); // same inputs → same output

        let s3 = Block::compute_election_score(&prev, "nodeB", 100);
        assert_ne!(s1, s3); // different address → different score
    }

    #[test]
    fn test_election_fairness() {
        // Over many slots, different nodes should "win" (have lowest score)
        let prev = "0".repeat(64);
        let nodes = ["nodeA", "nodeB", "nodeC", "nodeD"];
        let mut wins = [0u32; 4];

        for slot in 0..1000 {
            let scores: Vec<String> = nodes
                .iter()
                .map(|n| Block::compute_election_score(&prev, n, slot))
                .collect();
            let min_idx = scores
                .iter()
                .enumerate()
                .min_by_key(|(_, s)| s.as_str())
                .unwrap()
                .0;
            wins[min_idx] += 1;
        }

        // Each node should win at least 100 times out of 1000 (expect ~250)
        for (i, &w) in wins.iter().enumerate() {
            assert!(
                w > 100,
                "node {} only won {} times out of 1000, expected ~250",
                nodes[i],
                w
            );
        }
    }
}
