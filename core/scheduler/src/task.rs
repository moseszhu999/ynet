//! Task and result data structures.

use serde::{Deserialize, Serialize};

/// Specification for a compute task.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskSpec {
    /// Command to execute.
    pub command: String,
    /// Optional Docker image (if None, runs directly as subprocess).
    pub docker_image: Option<String>,
    /// Timeout in seconds.
    pub timeout_secs: u64,
    /// Requested CPU cores.
    pub cpu_cores: u32,
    /// Requested memory in MB.
    pub memory_mb: u32,
}

/// Status of a compute task.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TaskStatus {
    Pending,
    Running,
    Completed,
    Failed,
}

/// A compute task in the system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputeTask {
    pub id: String,
    pub spec: TaskSpec,
    pub submitter: String,
    pub worker: Option<String>,
    pub status: TaskStatus,
    pub result: Option<TaskResult>,
    pub created_at: u64,
    /// Price in YNET (set by submitter or auto-calculated).
    pub price: u64,
}

/// Result of a completed task.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskResult {
    pub task_id: String,
    pub worker_id: String,
    pub success: bool,
    pub exit_code: i32,
    pub stdout: String,
    pub stderr: String,
    pub duration_secs: f64,
}
