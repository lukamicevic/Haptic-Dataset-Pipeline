/// Operations for combining signals
#[derive(Debug, Clone)]
pub enum CombineOp {
    /// Insert signal at position with optional offset (expands signal)
    Insert { add_offset: usize },
    /// Mix signals with weighted balance
    Mix {
        mix_balance: f32,
        add_offset: usize,
        normalize: bool,
    },
    /// Replace samples at position (overwrites, same length)
    Replace { add_offset: usize },
}

/// Operations for separating signals
#[derive(Debug, Clone)]
pub enum SeparateOp {
    /// Remove signal section at position
    Remove,
    /// Unmix previously mixed signals
    Unmix { mix_balance: f32 },
}
