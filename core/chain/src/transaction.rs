//! Transaction types and signing.

use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TxType {
    Transfer,
    ComputeProof,     // Proof of Computing — real inference work
    NodeRegister,
    NodeExit,
    BlockBookkeeping, // Tiny fee for block producer (record-keeping)
    ApiPayment,       // User pays for API inference
    FeeBurn,          // Burned portion of API fees
    StakingReward,    // Distributed from staking pool
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transaction {
    pub hash: String,
    pub from: String,
    pub to: String,
    pub amount: u64,
    pub tx_type: TxType,
    pub timestamp: u64,
    pub nonce: u64,
    pub signature: String,
}

impl Transaction {
    pub fn new(from: &str, to: &str, amount: u64, tx_type: TxType, signing_key: &SigningKey) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let nonce = rand::random::<u64>();

        let mut tx = Transaction {
            hash: String::new(),
            from: from.to_string(),
            to: to.to_string(),
            amount,
            tx_type,
            timestamp,
            nonce,
            signature: String::new(),
        };

        tx.hash = tx.calculate_hash();
        let sig = signing_key.sign(tx.hash.as_bytes());
        tx.signature = hex::encode(sig.to_bytes());

        tx
    }

    /// Create a block bookkeeping transaction — no signature needed.
    /// Tiny fee paid to the block producer for record-keeping.
    pub fn bookkeeping(to: &str, amount: u64) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let nonce = rand::random::<u64>();

        let mut tx = Transaction {
            hash: String::new(),
            from: "network".to_string(),
            to: to.to_string(),
            amount,
            tx_type: TxType::BlockBookkeeping,
            timestamp,
            nonce,
            signature: String::new(),
        };
        tx.hash = tx.calculate_hash();
        tx
    }

    pub fn calculate_hash(&self) -> String {
        let data = format!(
            "{}{}{}{}{}:{}",
            self.from, self.to, self.amount, self.timestamp, self.nonce, self.tx_type_str()
        );
        let mut hasher = Sha256::new();
        hasher.update(data.as_bytes());
        hex::encode(hasher.finalize())
    }

    fn tx_type_str(&self) -> &str {
        match self.tx_type {
            TxType::Transfer => "transfer",
            TxType::ComputeProof => "compute_proof",
            TxType::NodeRegister => "node_register",
            TxType::NodeExit => "node_exit",
            TxType::BlockBookkeeping => "block_bookkeeping",
            TxType::ApiPayment => "api_payment",
            TxType::FeeBurn => "fee_burn",
            TxType::StakingReward => "staking_reward",
        }
    }

    /// Verify the transaction signature.
    pub fn verify(&self, public_key: &VerifyingKey) -> bool {
        if self.tx_type == TxType::BlockBookkeeping {
            return true; // bookkeeping tx has no signature
        }
        let sig_bytes = match hex::decode(&self.signature) {
            Ok(b) => b,
            Err(_) => return false,
        };
        let sig = match Signature::from_slice(&sig_bytes) {
            Ok(s) => s,
            Err(_) => return false,
        };
        public_key.verify(self.hash.as_bytes(), &sig).is_ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wallet::Wallet;

    #[test]
    fn test_create_and_verify_tx() {
        let wallet = Wallet::generate();
        let tx = Transaction::new(
            &wallet.address(),
            "recipient_addr",
            100,
            TxType::Transfer,
            wallet.signing_key(),
        );
        assert!(!tx.hash.is_empty());
        assert!(!tx.signature.is_empty());
        assert!(tx.verify(&wallet.verifying_key()));
    }

    #[test]
    fn test_bookkeeping_tx() {
        let tx = Transaction::bookkeeping("producer_addr", 1);
        assert_eq!(tx.from, "network");
        assert_eq!(tx.amount, 1);
        assert_eq!(tx.tx_type, TxType::BlockBookkeeping);
    }
}
