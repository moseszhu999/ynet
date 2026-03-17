//! Blockchain state — chain, mempool, balances.
//! Deterministic random block producer election.
//! score = hash(prev_hash + address + slot), lowest wins.

use crate::block::Block;
use crate::economics::BLOCK_BOOKKEEPING_FEE;
use crate::transaction::{Transaction, TxType};
use crate::{ChainStatus, WalletInfo};
use log::info;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Blockchain {
    chain: Vec<Block>,
    pending_transactions: Vec<Transaction>,
    balances: HashMap<String, u64>,
}

impl Blockchain {
    pub fn new() -> Self {
        let genesis = Block::genesis();
        Blockchain {
            chain: vec![genesis],
            pending_transactions: Vec::new(),
            balances: HashMap::new(),
        }
    }

    pub fn height(&self) -> u64 {
        self.chain.len() as u64 - 1
    }

    pub fn latest_block(&self) -> &Block {
        self.chain.last().unwrap()
    }

    pub fn get_balance(&self, address: &str) -> u64 {
        *self.balances.get(address).unwrap_or(&0)
    }

    pub fn get_chain(&self) -> &[Block] {
        &self.chain
    }

    pub fn get_pending_transactions(&self) -> &[Transaction] {
        &self.pending_transactions
    }

    pub fn status(&self) -> ChainStatus {
        ChainStatus {
            height: self.height(),
            latest_hash: self.latest_block().hash.clone(),
            pending_tx_count: self.pending_transactions.len(),
        }
    }

    pub fn wallet_info(&self, address: &str) -> WalletInfo {
        WalletInfo {
            address: address.to_string(),
            balance: self.get_balance(address),
        }
    }

    /// Compute this node's election score for the current slot.
    pub fn my_election_score(&self, my_address: &str) -> String {
        let slot = Block::current_slot();
        Block::compute_election_score(&self.latest_block().hash, my_address, slot)
    }

    /// Add a transaction to the mempool after validation.
    pub fn add_transaction(&mut self, tx: Transaction) -> Result<(), String> {
        if tx.tx_type == TxType::BlockBookkeeping {
            return Err("bookkeeping transactions are only added during block production".to_string());
        }

        // Check balance for transfers and API payments
        if tx.tx_type == TxType::Transfer || tx.tx_type == TxType::ApiPayment {
            let balance = self.get_balance(&tx.from);
            if balance < tx.amount {
                return Err(format!(
                    "insufficient balance: have {}, need {}",
                    balance, tx.amount
                ));
            }
        }

        // Duplicate detection
        if self.pending_transactions.iter().any(|t| t.hash == tx.hash) {
            return Err("duplicate transaction".to_string());
        }

        info!("Transaction added to mempool: {} -> {} ({} YNET)", tx.from, tx.to, tx.amount);
        self.pending_transactions.push(tx);
        Ok(())
    }

    /// Produce a new block with pending transactions.
    /// Called when this node believes it's the elected producer for the current slot.
    pub fn produce_block(&mut self, producer_address: &str) -> Block {
        let bookkeeping = Transaction::bookkeeping(producer_address, BLOCK_BOOKKEEPING_FEE);

        let mut txs = vec![bookkeeping];
        txs.append(&mut std::mem::take(&mut self.pending_transactions));

        let prev_hash = &self.latest_block().hash;
        let index = self.chain.len() as u64;

        let block = Block::new(index, prev_hash, producer_address, txs);

        info!(
            "Block #{} produced by {} (slot={}, score={}..., hash={}...)",
            block.index, producer_address, block.slot,
            &block.election_score[..8], &block.hash[..8]
        );

        self.apply_block_transactions(&block);
        self.chain.push(block.clone());

        block
    }

    /// Accept a block received from the network.
    /// If a block for the same height already exists with a higher election score,
    /// replace it (fork resolution: lowest score wins).
    pub fn accept_block(&mut self, block: Block) -> Result<(), String> {
        if !block.is_valid() {
            return Err("block hash or election score is invalid".to_string());
        }

        let latest = self.latest_block();

        // Same height — fork resolution: keep lowest election score
        if block.index == latest.index && block.index > 0 {
            if block.election_score < latest.election_score {
                info!(
                    "Fork resolution: replacing block #{} (score {}... < {}...)",
                    block.index,
                    &block.election_score[..8],
                    &latest.election_score[..8]
                );
                let old_block = self.chain.pop().unwrap();
                self.rollback_block_transactions(&old_block);
                for tx in &old_block.transactions {
                    if tx.tx_type != TxType::BlockBookkeeping {
                        self.pending_transactions.push(tx.clone());
                    }
                }
                let block_tx_hashes: Vec<String> =
                    block.transactions.iter().map(|t| t.hash.clone()).collect();
                self.pending_transactions
                    .retain(|t| !block_tx_hashes.contains(&t.hash));
                self.apply_block_transactions(&block);
                self.chain.push(block);
                return Ok(());
            } else {
                return Err("already have a block with equal or lower score at this height".to_string());
            }
        }

        // Normal case: next block
        if block.prev_hash != latest.hash {
            return Err("block prev_hash does not match".to_string());
        }
        if block.index != latest.index + 1 {
            return Err(format!(
                "block index mismatch: expected {}, got {}",
                latest.index + 1,
                block.index
            ));
        }

        let block_tx_hashes: Vec<String> = block.transactions.iter().map(|t| t.hash.clone()).collect();
        self.pending_transactions
            .retain(|t| !block_tx_hashes.contains(&t.hash));

        self.apply_block_transactions(&block);
        self.chain.push(block);
        Ok(())
    }

    /// Replace chain if a longer valid chain is received.
    pub fn replace_chain(&mut self, new_chain: Vec<Block>) -> Result<(), String> {
        if new_chain.len() <= self.chain.len() {
            return Err("received chain is not longer".to_string());
        }
        if !Self::is_valid_chain(&new_chain) {
            return Err("received chain is invalid".to_string());
        }

        self.balances.clear();
        for block in &new_chain {
            self.apply_block_transactions(block);
        }
        self.chain = new_chain;
        self.pending_transactions.clear();
        info!("Chain replaced, new height: {}", self.height());
        Ok(())
    }

    fn apply_block_transactions(&mut self, block: &Block) {
        for tx in &block.transactions {
            if tx.from != "network" && tx.amount > 0 {
                let balance = self.balances.entry(tx.from.clone()).or_insert(0);
                *balance = balance.saturating_sub(tx.amount);
            }
            if tx.amount > 0 {
                let balance = self.balances.entry(tx.to.clone()).or_insert(0);
                *balance = balance.saturating_add(tx.amount);
            }
        }
    }

    fn rollback_block_transactions(&mut self, block: &Block) {
        for tx in block.transactions.iter().rev() {
            if tx.amount > 0 {
                let balance = self.balances.entry(tx.to.clone()).or_insert(0);
                *balance = balance.saturating_sub(tx.amount);
            }
            if tx.from != "network" && tx.amount > 0 {
                let balance = self.balances.entry(tx.from.clone()).or_insert(0);
                *balance = balance.saturating_add(tx.amount);
            }
        }
    }

    fn is_valid_chain(chain: &[Block]) -> bool {
        if chain.is_empty() {
            return false;
        }
        // Validate genesis
        if !chain[0].is_valid() {
            return false;
        }
        for i in 1..chain.len() {
            let prev = &chain[i - 1];
            let curr = &chain[i];
            if curr.prev_hash != prev.hash {
                return false;
            }
            if !curr.is_valid() {
                return false;
            }
        }
        true
    }

    /// Get recent transactions (last N from the chain).
    pub fn recent_transactions(&self, limit: usize) -> Vec<&Transaction> {
        self.chain
            .iter()
            .rev()
            .flat_map(|b| b.transactions.iter())
            .take(limit)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wallet::Wallet;

    #[test]
    fn test_new_blockchain() {
        let bc = Blockchain::new();
        assert_eq!(bc.height(), 0);
        assert_eq!(bc.get_balance("anyone"), 0);
    }

    #[test]
    fn test_produce_block_bookkeeping_fee() {
        let mut bc = Blockchain::new();
        bc.produce_block("producer1");
        assert_eq!(bc.get_balance("producer1"), BLOCK_BOOKKEEPING_FEE);
        assert_eq!(bc.height(), 1);
    }

    #[test]
    fn test_block_has_election_score() {
        let mut bc = Blockchain::new();
        let block = bc.produce_block("producer1");
        assert!(!block.election_score.is_empty());
        assert_eq!(block.election_score.len(), 64);
        assert!(block.is_valid());
    }

    #[test]
    fn test_transfer() {
        let mut bc = Blockchain::new();
        let sender = Wallet::generate();
        let sender_addr = sender.address();

        for _ in 0..30 {
            bc.produce_block(&sender_addr);
        }
        let balance = bc.get_balance(&sender_addr);
        assert_eq!(balance, 30 * BLOCK_BOOKKEEPING_FEE);

        let tx = Transaction::new(
            &sender_addr,
            "recipient",
            20,
            TxType::Transfer,
            sender.signing_key(),
        );
        bc.add_transaction(tx).unwrap();
        bc.produce_block("other_producer");

        assert_eq!(bc.get_balance(&sender_addr), 30 * BLOCK_BOOKKEEPING_FEE - 20);
        assert_eq!(bc.get_balance("recipient"), 20);
    }

    #[test]
    fn test_insufficient_balance() {
        let mut bc = Blockchain::new();
        let sender = Wallet::generate();
        let tx = Transaction::new(
            &sender.address(),
            "recipient",
            100,
            TxType::Transfer,
            sender.signing_key(),
        );
        let result = bc.add_transaction(tx);
        assert!(result.is_err());
    }

    #[test]
    fn test_chain_validation() {
        let mut bc = Blockchain::new();
        bc.produce_block("p1");
        bc.produce_block("p2");
        bc.produce_block("p3");
        assert_eq!(bc.height(), 3);
        assert!(Blockchain::is_valid_chain(bc.get_chain()));
    }

    #[test]
    fn test_accept_block() {
        let mut bc1 = Blockchain::new();
        let mut bc2 = Blockchain::new();

        let block = bc1.produce_block("producer1");
        bc2.accept_block(block).unwrap();
        assert_eq!(bc2.height(), 1);
        assert_eq!(bc2.get_balance("producer1"), BLOCK_BOOKKEEPING_FEE);
    }

    #[test]
    fn test_fork_resolution_lower_score_wins() {
        let mut bc = Blockchain::new();

        let block_a = bc.produce_block("producer_a");
        let score_a = block_a.election_score.clone();

        let old = bc.chain.pop().unwrap();
        bc.rollback_block_transactions(&old);

        let block_b = bc.produce_block("producer_b");
        let score_b = block_b.election_score.clone();

        if score_a < score_b {
            let result = bc.accept_block(block_a);
            assert!(result.is_ok());
        } else {
            let result = bc.accept_block(block_a);
            assert!(result.is_err());
        }
    }

    #[test]
    fn test_election_score_verified() {
        let mut bc = Blockchain::new();
        let mut block = bc.produce_block("producer1");

        bc.chain.pop();
        bc.rollback_block_transactions(&block);

        block.election_score = "f".repeat(64);
        let result = bc.accept_block(block);
        assert!(result.is_err());
    }

    #[test]
    fn test_duplicate_transaction_rejected() {
        let mut bc = Blockchain::new();
        let sender = Wallet::generate();
        let sender_addr = sender.address();

        for _ in 0..10 {
            bc.produce_block(&sender_addr);
        }

        let tx = Transaction::new(
            &sender_addr, "recipient", 1, TxType::Transfer, sender.signing_key(),
        );
        bc.add_transaction(tx.clone()).unwrap();
        let result = bc.add_transaction(tx);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("duplicate"));
    }
}
