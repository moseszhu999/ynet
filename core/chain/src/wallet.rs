//! Ed25519 wallet — keypair management, address derivation.

use ed25519_dalek::{SigningKey, VerifyingKey};
use rand::rngs::OsRng;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalletData {
    pub secret_key_hex: String,
}

pub struct Wallet {
    signing_key: SigningKey,
}

impl Wallet {
    /// Generate a new random wallet.
    pub fn generate() -> Self {
        let signing_key = SigningKey::generate(&mut OsRng);
        Wallet { signing_key }
    }

    /// Restore from a hex-encoded secret key.
    pub fn from_secret_hex(hex_str: &str) -> Result<Self, String> {
        let bytes = hex::decode(hex_str).map_err(|e| e.to_string())?;
        let key_bytes: [u8; 32] = bytes
            .try_into()
            .map_err(|_| "invalid key length".to_string())?;
        Ok(Wallet {
            signing_key: SigningKey::from_bytes(&key_bytes),
        })
    }

    /// Export secret key as hex for persistence.
    pub fn secret_hex(&self) -> String {
        hex::encode(self.signing_key.to_bytes())
    }

    /// Derive the wallet address from the public key (sha256 hash prefix).
    pub fn address(&self) -> String {
        let pubkey_bytes = self.signing_key.verifying_key().to_bytes();
        let mut hasher = Sha256::new();
        hasher.update(pubkey_bytes);
        let hash = hex::encode(hasher.finalize());
        format!("ynet1{}", &hash[..40])
    }

    pub fn signing_key(&self) -> &SigningKey {
        &self.signing_key
    }

    pub fn verifying_key(&self) -> VerifyingKey {
        self.signing_key.verifying_key()
    }

    /// Export serializable data for storage.
    pub fn to_data(&self) -> WalletData {
        WalletData {
            secret_key_hex: self.secret_hex(),
        }
    }

    pub fn from_data(data: &WalletData) -> Result<Self, String> {
        Self::from_secret_hex(&data.secret_key_hex)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_wallet() {
        let w = Wallet::generate();
        let addr = w.address();
        assert!(addr.starts_with("ynet1"));
        assert_eq!(addr.len(), 45); // "ynet1" + 40 hex chars
    }

    #[test]
    fn test_restore_wallet() {
        let w1 = Wallet::generate();
        let hex = w1.secret_hex();
        let w2 = Wallet::from_secret_hex(&hex).unwrap();
        assert_eq!(w1.address(), w2.address());
    }
}
