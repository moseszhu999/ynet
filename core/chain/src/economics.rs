//! yNet economic model — Proof of Computing (PoC)
//!
//! Tokenomics:
//!   - Total supply: 1 billion YNET (1_000_000_000)
//!   - Smallest unit: 1 nYNET (nano-YNET), 1 YNET = 1_000_000_000 nYNET
//!   - All on-chain amounts are in nYNET for precision
//!   - No mining rewards — all revenue comes from real API usage
//!
//! Revenue model (like OpenAI but decentralized):
//!   - Users pay per inference token (input/output)
//!   - GPU nodes earn by serving inference requests
//!   - YNET holders get discount on API pricing (like BNB on Binance)
//!   - Blocks are produced every ~3 seconds for transparent record-keeping
//!   - Block producer is randomly selected, gets tiny bookkeeping fee
//!
//! Fee distribution:
//!   - 80% → GPU node (worker)
//!   - 10% → burned (deflationary)
//!   - 5%  → staking rewards pool
//!   - 5%  → network development fund

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---- Constants ----

/// Total supply: 1 billion YNET
pub const TOTAL_SUPPLY: u64 = 1_000_000_000;

/// 1 YNET = 10^9 nYNET (nano-YNET, smallest unit)
pub const YNET_DECIMALS: u64 = 1_000_000_000;

/// Target block time: ~3 seconds (like BNB speed, but no super nodes)
pub const TARGET_BLOCK_TIME_SECS: u64 = 3;

/// Bookkeeping fee for block producer (tiny, just for record-keeping)
pub const BLOCK_BOOKKEEPING_FEE: u64 = 1; // 1 YNET per block

/// Fee distribution (basis points, total = 10000)
pub const FEE_TO_WORKER_BPS: u64 = 8000;  // 80%
pub const FEE_BURN_BPS: u64 = 1000;        // 10%
pub const FEE_STAKING_BPS: u64 = 500;      // 5%
pub const FEE_DEV_FUND_BPS: u64 = 500;     // 5%

/// YNET holder discount tiers
pub const DISCOUNT_TIERS: [(u64, u16); 4] = [
    (100,    500),   // Hold 100+ YNET  → 5% discount
    (1_000,  1000),  // Hold 1000+ YNET → 10% discount
    (10_000, 1500),  // Hold 10k+ YNET  → 15% discount
    (100_000, 2500), // Hold 100k+ YNET → 25% discount
];

// ---- API Pricing ----

/// Pricing for different model tiers (in nYNET per 1000 tokens)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPricing {
    pub model_id: String,
    pub input_price_per_1k: u64,   // nYNET per 1000 input tokens
    pub output_price_per_1k: u64,  // nYNET per 1000 output tokens
    pub tier: ModelTier,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ModelTier {
    Small,   // 7B-13B params
    Medium,  // 30B-70B params
    Large,   // 70B+ params
    Premium, // frontier models
}

impl ModelPricing {
    /// Default pricing table (similar to OpenAI scale)
    pub fn default_pricing() -> Vec<ModelPricing> {
        vec![
            ModelPricing {
                model_id: "llama-3-8b".to_string(),
                input_price_per_1k: 50_000,     // 0.00005 YNET
                output_price_per_1k: 100_000,    // 0.0001 YNET
                tier: ModelTier::Small,
            },
            ModelPricing {
                model_id: "llama-3-70b".to_string(),
                input_price_per_1k: 500_000,     // 0.0005 YNET
                output_price_per_1k: 1_000_000,  // 0.001 YNET
                tier: ModelTier::Medium,
            },
            ModelPricing {
                model_id: "llama-3-405b".to_string(),
                input_price_per_1k: 2_000_000,   // 0.002 YNET
                output_price_per_1k: 5_000_000,   // 0.005 YNET
                tier: ModelTier::Large,
            },
            ModelPricing {
                model_id: "default".to_string(),
                input_price_per_1k: 500_000,
                output_price_per_1k: 1_000_000,
                tier: ModelTier::Medium,
            },
        ]
    }

    pub fn find(model_id: &str) -> ModelPricing {
        Self::default_pricing()
            .into_iter()
            .find(|p| p.model_id == model_id)
            .unwrap_or_else(|| {
                Self::default_pricing()
                    .into_iter()
                    .find(|p| p.model_id == "default")
                    .unwrap()
            })
    }
}

// ---- Invoice / Usage ----

/// An API usage record (inference request).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageRecord {
    pub request_id: String,
    pub user_address: String,
    pub worker_address: String,
    pub model_id: String,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub total_cost_nynet: u64,
    pub discount_bps: u16,
    pub payment_currency: PaymentCurrency,
    pub timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PaymentCurrency {
    YNET,
    ETH,
    BTC,
    USDT,
    USDC,
}

// ---- Fee Calculator ----

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeeBreakdown {
    pub total_cost: u64,          // Total before discount
    pub discount_bps: u16,        // Discount in basis points
    pub discounted_cost: u64,     // After YNET holder discount
    pub to_worker: u64,           // 80%
    pub to_burn: u64,             // 10%
    pub to_staking_pool: u64,     // 5%
    pub to_dev_fund: u64,         // 5%
}

pub struct FeeCalculator;

impl FeeCalculator {
    /// Calculate the fee for an API call.
    pub fn calculate(
        model_id: &str,
        input_tokens: u64,
        output_tokens: u64,
        user_ynet_balance: u64,
        payment_currency: &PaymentCurrency,
    ) -> FeeBreakdown {
        let pricing = ModelPricing::find(model_id);

        let input_cost = (input_tokens * pricing.input_price_per_1k) / 1000;
        let output_cost = (output_tokens * pricing.output_price_per_1k) / 1000;
        let total_cost = input_cost + output_cost;

        // Calculate YNET holder discount
        let discount_bps = if *payment_currency == PaymentCurrency::YNET {
            Self::get_discount(user_ynet_balance)
        } else {
            0
        };

        let discount = (total_cost * discount_bps as u64) / 10000;
        let discounted_cost = total_cost - discount;

        // Split fees
        let to_worker = (discounted_cost * FEE_TO_WORKER_BPS) / 10000;
        let to_burn = (discounted_cost * FEE_BURN_BPS) / 10000;
        let to_staking = (discounted_cost * FEE_STAKING_BPS) / 10000;
        let to_dev = discounted_cost - to_worker - to_burn - to_staking; // remainder

        FeeBreakdown {
            total_cost,
            discount_bps,
            discounted_cost,
            to_worker,
            to_burn,
            to_staking_pool: to_staking,
            to_dev_fund: to_dev,
        }
    }

    /// Get discount based on YNET balance.
    fn get_discount(balance: u64) -> u16 {
        let mut discount = 0u16;
        for (threshold, bps) in &DISCOUNT_TIERS {
            if balance >= *threshold {
                discount = *bps;
            }
        }
        discount
    }

}

// ---- Node Reputation ----

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeReputation {
    pub address: String,
    pub tasks_completed: u64,
    pub tasks_failed: u64,
    pub total_tokens_served: u64,
    pub uptime_blocks: u64,
    pub total_earned: u64,
    pub score: u64, // 0-10000
}

impl NodeReputation {
    pub fn new(address: &str) -> Self {
        NodeReputation {
            address: address.to_string(),
            tasks_completed: 0,
            tasks_failed: 0,
            total_tokens_served: 0,
            uptime_blocks: 0,
            total_earned: 0,
            score: 5000, // start at 50%
        }
    }

    pub fn record_completion(&mut self, tokens_served: u64, earned: u64) {
        self.tasks_completed += 1;
        self.total_tokens_served += tokens_served;
        self.total_earned += earned;
        self.recalculate_score();
    }

    pub fn record_failure(&mut self) {
        self.tasks_failed += 1;
        self.recalculate_score();
    }

    pub fn record_uptime(&mut self) {
        self.uptime_blocks += 1;
    }

    fn recalculate_score(&mut self) {
        let total = self.tasks_completed + self.tasks_failed;
        if total == 0 {
            self.score = 5000;
            return;
        }
        // Success rate (0-10000)
        let success_rate = (self.tasks_completed * 10000) / total;
        // Volume bonus (up to 1000 points for 1000+ tasks)
        let volume_bonus = (self.tasks_completed.min(1000) * 1000) / 1000;
        self.score = ((success_rate * 8 + volume_bonus * 2) / 10).min(10000);
    }

    /// Higher reputation → more likely to be assigned tasks.
    pub fn assignment_weight(&self) -> u64 {
        // Score 0-10000 maps to weight 1-100
        (self.score / 100).max(1)
    }
}

// ---- Economics State ----

/// Tracks economic state: reputation, burn counter, staking/dev pool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EconomicsState {
    pub total_burned: u64,
    pub staking_pool: u64,
    pub dev_fund: u64,
    pub total_api_revenue: u64,
    pub total_tokens_served: u64,
    pub reputations: HashMap<String, NodeReputation>,
}

impl EconomicsState {
    pub fn new() -> Self {
        EconomicsState {
            total_burned: 0,
            staking_pool: 0,
            dev_fund: 0,
            total_api_revenue: 0,
            total_tokens_served: 0,
            reputations: HashMap::new(),
        }
    }

    /// Process a fee payment after API call.
    pub fn process_fee(&mut self, fee: &FeeBreakdown, worker_address: &str, tokens_served: u64) {
        self.total_burned += fee.to_burn;
        self.staking_pool += fee.to_staking_pool;
        self.dev_fund += fee.to_dev_fund;
        self.total_api_revenue += fee.discounted_cost;
        self.total_tokens_served += tokens_served;

        let rep = self.reputations
            .entry(worker_address.to_string())
            .or_insert_with(|| NodeReputation::new(worker_address));
        rep.record_completion(tokens_served, fee.to_worker);
    }

    pub fn record_failure(&mut self, worker_address: &str) {
        let rep = self.reputations
            .entry(worker_address.to_string())
            .or_insert_with(|| NodeReputation::new(worker_address));
        rep.record_failure();
    }

    pub fn get_reputation(&self, address: &str) -> Option<&NodeReputation> {
        self.reputations.get(address)
    }

    pub fn summary(&self) -> EconomicsSummary {
        EconomicsSummary {
            total_burned: self.total_burned,
            staking_pool: self.staking_pool,
            dev_fund: self.dev_fund,
            total_api_revenue: self.total_api_revenue,
            total_tokens_served: self.total_tokens_served,
            active_nodes: self.reputations.len(),
            circulating_supply: TOTAL_SUPPLY.saturating_sub(self.total_burned),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EconomicsSummary {
    pub total_burned: u64,
    pub staking_pool: u64,
    pub dev_fund: u64,
    pub total_api_revenue: u64,
    pub total_tokens_served: u64,
    pub active_nodes: usize,
    pub circulating_supply: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fee_calculation_no_discount() {
        let fee = FeeCalculator::calculate(
            "llama-3-70b",
            1000,  // 1000 input tokens
            500,   // 500 output tokens
            0,     // no YNET balance
            &PaymentCurrency::USDT,
        );
        // Input: 1000 * 500_000 / 1000 = 500_000
        // Output: 500 * 1_000_000 / 1000 = 500_000
        // Total: 1_000_000 nYNET
        assert_eq!(fee.total_cost, 1_000_000);
        assert_eq!(fee.discount_bps, 0);
        assert_eq!(fee.discounted_cost, 1_000_000);
        assert_eq!(fee.to_worker, 800_000);  // 80%
        assert_eq!(fee.to_burn, 100_000);     // 10%
    }

    #[test]
    fn test_fee_with_ynet_discount() {
        let fee = FeeCalculator::calculate(
            "llama-3-70b",
            1000,
            500,
            10_000,  // 10k YNET → 15% discount
            &PaymentCurrency::YNET,
        );
        assert_eq!(fee.total_cost, 1_000_000);
        assert_eq!(fee.discount_bps, 1500);
        assert_eq!(fee.discounted_cost, 850_000);  // 15% off
        assert_eq!(fee.to_worker, 680_000);  // 80% of 850k
    }

    #[test]
    fn test_no_discount_for_non_ynet_payment() {
        let fee = FeeCalculator::calculate(
            "llama-3-70b",
            1000,
            500,
            100_000,  // Large balance but paying in ETH
            &PaymentCurrency::ETH,
        );
        assert_eq!(fee.discount_bps, 0);
        assert_eq!(fee.discounted_cost, fee.total_cost);
    }

    #[test]
    fn test_block_bookkeeping_fee() {
        assert_eq!(BLOCK_BOOKKEEPING_FEE, 1);
        assert_eq!(TARGET_BLOCK_TIME_SECS, 3);
    }

    #[test]
    fn test_reputation() {
        let mut rep = NodeReputation::new("node1");
        assert_eq!(rep.score, 5000);

        rep.record_completion(1000, 100);
        rep.record_completion(2000, 200);
        assert!(rep.score > 5000); // should increase with completions

        rep.record_failure();
        let score_after_fail = rep.score;
        assert!(score_after_fail < 10000);
    }

    #[test]
    fn test_economics_state() {
        let mut state = EconomicsState::new();
        let fee = FeeCalculator::calculate(
            "llama-3-70b", 1000, 500, 0, &PaymentCurrency::YNET,
        );
        state.process_fee(&fee, "worker1", 1500);

        assert!(state.total_api_revenue > 0);
        assert!(state.total_burned > 0);
        assert_eq!(state.total_tokens_served, 1500);

        let rep = state.get_reputation("worker1").unwrap();
        assert_eq!(rep.tasks_completed, 1);

        let summary = state.summary();
        assert_eq!(summary.active_nodes, 1);
        assert_eq!(summary.circulating_supply, TOTAL_SUPPLY - state.total_burned);
    }

    #[test]
    fn test_discount_tiers() {
        assert_eq!(FeeCalculator::get_discount(0), 0);
        assert_eq!(FeeCalculator::get_discount(50), 0);
        assert_eq!(FeeCalculator::get_discount(100), 500);
        assert_eq!(FeeCalculator::get_discount(1_000), 1000);
        assert_eq!(FeeCalculator::get_discount(10_000), 1500);
        assert_eq!(FeeCalculator::get_discount(100_000), 2500);
        assert_eq!(FeeCalculator::get_discount(999_999), 2500);
    }
}
