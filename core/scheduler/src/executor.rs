//! Task executor — runs tasks as subprocesses (with optional Docker isolation).
//! Enforces timeout. Only local tasks (execute_local) run without Docker.

use crate::task::{ComputeTask, TaskResult};
use log::{info, warn};
use std::time::{Duration, Instant};
use tokio::process::Command;

pub struct TaskExecutor;

impl TaskExecutor {
    pub fn new() -> Self {
        TaskExecutor
    }

    /// Execute a compute task with timeout enforcement.
    pub async fn execute(&self, task: &ComputeTask) -> TaskResult {
        let start = Instant::now();
        let timeout = Duration::from_secs(task.spec.timeout_secs);

        let result = if let Some(ref image) = task.spec.docker_image {
            self.execute_docker(task, image).await
        } else {
            self.execute_subprocess(task).await
        };

        // Enforce timeout
        let result = match tokio::time::timeout(Duration::ZERO, async { result }).await {
            Ok(r) => r,
            Err(_) => unreachable!(),
        };

        let duration = start.elapsed().as_secs_f64();

        if start.elapsed() > timeout {
            warn!("Task {} exceeded timeout ({:.0}s > {}s)", task.id, duration, task.spec.timeout_secs);
        }

        match result {
            Ok((exit_code, stdout, stderr)) => {
                info!("Task {} completed (exit={}, {:.1}s)", task.id, exit_code, duration);
                TaskResult {
                    task_id: task.id.clone(),
                    worker_id: task.worker.clone().unwrap_or_default(),
                    success: exit_code == 0,
                    exit_code,
                    stdout,
                    stderr,
                    duration_secs: duration,
                }
            }
            Err(e) => {
                warn!("Task {} failed: {}", task.id, e);
                TaskResult {
                    task_id: task.id.clone(),
                    worker_id: task.worker.clone().unwrap_or_default(),
                    success: false,
                    exit_code: -1,
                    stdout: String::new(),
                    stderr: e,
                    duration_secs: duration,
                }
            }
        }
    }

    async fn execute_subprocess(&self, task: &ComputeTask) -> Result<(i32, String, String), String> {
        let timeout = Duration::from_secs(task.spec.timeout_secs);

        let future = Command::new("sh")
            .arg("-c")
            .arg(&task.spec.command)
            .output();

        let output = tokio::time::timeout(timeout, future)
            .await
            .map_err(|_| format!("task timed out after {}s", task.spec.timeout_secs))?
            .map_err(|e| format!("failed to spawn process: {}", e))?;

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        let exit_code = output.status.code().unwrap_or(-1);

        Ok((exit_code, stdout, stderr))
    }

    async fn execute_docker(&self, task: &ComputeTask, image: &str) -> Result<(i32, String, String), String> {
        // Validate image name: only allow safe characters
        if !image.chars().all(|c| c.is_alphanumeric() || "._/-:".contains(c)) {
            return Err(format!("invalid docker image name: {}", image));
        }

        let timeout = Duration::from_secs(task.spec.timeout_secs);

        let future = Command::new("docker")
            .args([
                "run",
                "--rm",
                "--network", "none",
                "--memory", &format!("{}m", task.spec.memory_mb),
                "--cpus", &task.spec.cpu_cores.to_string(),
                image,
                "sh", "-c", &task.spec.command,
            ])
            .output();

        let output = tokio::time::timeout(timeout, future)
            .await
            .map_err(|_| format!("docker task timed out after {}s", task.spec.timeout_secs))?
            .map_err(|e| format!("failed to run docker: {}", e))?;

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        let exit_code = output.status.code().unwrap_or(-1);

        Ok((exit_code, stdout, stderr))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::task::TaskSpec;

    #[tokio::test]
    async fn test_execute_echo() {
        let executor = TaskExecutor::new();
        let task = ComputeTask {
            id: "test-1".to_string(),
            spec: TaskSpec {
                command: "echo hello world".to_string(),
                docker_image: None,
                timeout_secs: 10,
                cpu_cores: 1,
                memory_mb: 128,
            },
            submitter: "test".to_string(),
            worker: Some("worker1".to_string()),
            status: crate::task::TaskStatus::Running,
            result: None,
            created_at: 0,
            price: 0,
        };

        let result = executor.execute(&task).await;
        assert!(result.success);
        assert_eq!(result.exit_code, 0);
        assert!(result.stdout.contains("hello world"));
    }

    #[tokio::test]
    async fn test_execute_failing_command() {
        let executor = TaskExecutor::new();
        let task = ComputeTask {
            id: "test-fail".to_string(),
            spec: TaskSpec {
                command: "exit 42".to_string(),
                docker_image: None,
                timeout_secs: 10,
                cpu_cores: 1,
                memory_mb: 128,
            },
            submitter: "test".to_string(),
            worker: Some("worker1".to_string()),
            status: crate::task::TaskStatus::Running,
            result: None,
            created_at: 0,
            price: 0,
        };

        let result = executor.execute(&task).await;
        assert!(!result.success);
        assert_eq!(result.exit_code, 42);
    }

    #[tokio::test]
    async fn test_timeout_enforcement() {
        let executor = TaskExecutor::new();
        let task = ComputeTask {
            id: "test-timeout".to_string(),
            spec: TaskSpec {
                command: "sleep 30".to_string(),
                docker_image: None,
                timeout_secs: 1,
                cpu_cores: 1,
                memory_mb: 128,
            },
            submitter: "test".to_string(),
            worker: Some("worker1".to_string()),
            status: crate::task::TaskStatus::Running,
            result: None,
            created_at: 0,
            price: 0,
        };

        let result = executor.execute(&task).await;
        assert!(!result.success);
        assert!(result.stderr.contains("timed out"));
    }
}
