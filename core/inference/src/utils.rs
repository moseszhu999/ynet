//! Utility functions for the inference module.

/// Get the current Unix timestamp in seconds.
///
/// This is used throughout the inference module for:
/// - Node capability timestamps
/// - Node announcement timestamps
/// - Heartbeat timestamps
/// - TTL expiration checks
#[inline]
pub fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_current_timestamp() {
        let ts = current_timestamp();
        // Should be a reasonable timestamp (after 2020-01-01)
        assert!(ts > 1577836800);
        // Should be before 2100-01-01
        assert!(ts < 4102444800);
    }

    #[test]
    fn test_timestamp_monotonic() {
        let ts1 = current_timestamp();
        std::thread::sleep(std::time::Duration::from_millis(10));
        let ts2 = current_timestamp();
        // Second call should be >= first call (same second or later)
        assert!(ts2 >= ts1);
    }
}
