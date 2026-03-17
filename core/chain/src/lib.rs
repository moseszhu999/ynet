//! yNet blockchain engine
//! Blocks, transactions, Proof of Computing, wallet with Ed25519 keys.
//! No PoW — blocks are produced every ~3s by randomly selected nodes.

mod block;
mod blockchain;
pub mod economics;
mod transaction;
pub mod wallet;

pub use block::Block;
pub use blockchain::Blockchain;
pub use economics::{
    EconomicsSummary, EconomicsState, FeeBreakdown, FeeCalculator,
    ModelPricing, NodeReputation, PaymentCurrency, UsageRecord,
    BLOCK_BOOKKEEPING_FEE, TARGET_BLOCK_TIME_SECS,
};
pub use transaction::{Transaction, TxType};
pub use wallet::Wallet;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChainStatus {
    pub height: u64,
    pub latest_hash: String,
    pub pending_tx_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalletInfo {
    pub address: String,
    pub balance: u64,
}
