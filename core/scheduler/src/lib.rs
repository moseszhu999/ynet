//! yNet scheduler — task submission, distribution, execution, and result tracking.

mod executor;
mod task;

pub use executor::TaskExecutor;
pub use task::{ComputeTask, TaskResult, TaskSpec, TaskStatus};

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Messages exchanged over P2P for task coordination.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SchedulerMessage {
    /// Broadcast a new task to the network (from submitter).
    TaskSubmitted(ComputeTask),
    /// A node claims a task (from worker).
    TaskClaimed { task_id: String, worker_id: String },
    /// Task execution completed (from worker).
    TaskCompleted(TaskResult),
}

/// Scheduler status for UI display.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerStatus {
    pub is_controller: bool,
    pub pending_tasks: usize,
    pub running_tasks: usize,
    pub completed_tasks: usize,
    pub my_completed: usize,
}

/// The scheduler manages task lifecycle across the network.
pub struct Scheduler {
    /// All known tasks.
    tasks: HashMap<String, ComputeTask>,
    /// This node's wallet address.
    node_address: String,
    /// Task executor for running tasks locally.
    executor: TaskExecutor,
}

impl Scheduler {
    pub fn new(node_address: &str) -> Self {
        Scheduler {
            tasks: HashMap::new(),
            node_address: node_address.to_string(),
            executor: TaskExecutor::new(),
        }
    }

    pub fn status(&self) -> SchedulerStatus {
        let pending = self.tasks.values().filter(|t| t.status == TaskStatus::Pending).count();
        let running = self.tasks.values().filter(|t| t.status == TaskStatus::Running).count();
        let completed = self.tasks.values().filter(|t| t.status == TaskStatus::Completed).count();
        let my_completed = self.tasks.values()
            .filter(|t| t.status == TaskStatus::Completed && t.worker.as_deref() == Some(&self.node_address))
            .count();

        SchedulerStatus {
            is_controller: false, // Will be determined by consensus later
            pending_tasks: pending,
            running_tasks: running,
            completed_tasks: completed,
            my_completed,
        }
    }

    /// Submit a new task. Returns the task for P2P broadcasting.
    pub fn submit_task(&mut self, spec: TaskSpec, submitter: &str) -> ComputeTask {
        let task = ComputeTask {
            id: Uuid::new_v4().to_string(),
            spec,
            submitter: submitter.to_string(),
            worker: None,
            status: TaskStatus::Pending,
            result: None,
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            price: 0,
        };

        log::info!("Task submitted: {} ({})", task.id, task.spec.command);
        self.tasks.insert(task.id.clone(), task.clone());
        task
    }

    /// Handle a task submitted by another node via P2P.
    pub fn receive_task(&mut self, task: ComputeTask) {
        if !self.tasks.contains_key(&task.id) {
            log::info!("Received task from network: {}", task.id);
            self.tasks.insert(task.id.clone(), task);
        }
    }

    /// Claim a pending task for local execution.
    pub fn claim_task(&mut self, task_id: &str) -> Option<ComputeTask> {
        if let Some(task) = self.tasks.get_mut(task_id) {
            if task.status == TaskStatus::Pending {
                task.status = TaskStatus::Running;
                task.worker = Some(self.node_address.clone());
                log::info!("Claimed task: {task_id}");
                return Some(task.clone());
            }
        }
        None
    }

    /// Handle a task claim from another node.
    pub fn receive_claim(&mut self, task_id: &str, worker_id: &str) {
        if let Some(task) = self.tasks.get_mut(task_id) {
            if task.status == TaskStatus::Pending {
                task.status = TaskStatus::Running;
                task.worker = Some(worker_id.to_string());
                log::info!("Task {task_id} claimed by {worker_id}");
            }
        }
    }

    /// Execute a claimed task locally. Returns the result.
    pub async fn execute_task(&mut self, task_id: &str) -> Option<TaskResult> {
        let task = self.tasks.get(task_id)?.clone();
        if task.worker.as_deref() != Some(&self.node_address) {
            return None;
        }

        log::info!("Executing task: {} ({})", task.id, task.spec.command);
        let result = self.executor.execute(&task).await;

        // Update local state
        if let Some(t) = self.tasks.get_mut(task_id) {
            t.status = if result.success { TaskStatus::Completed } else { TaskStatus::Failed };
            t.result = Some(result.clone());
        }

        Some(result)
    }

    /// Handle a task result from another node.
    pub fn receive_result(&mut self, result: TaskResult) {
        if let Some(task) = self.tasks.get_mut(&result.task_id) {
            task.status = if result.success { TaskStatus::Completed } else { TaskStatus::Failed };
            task.result = Some(result);
        }
    }

    /// Get all tasks (for UI).
    pub fn list_tasks(&self) -> Vec<&ComputeTask> {
        let mut tasks: Vec<_> = self.tasks.values().collect();
        tasks.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        tasks
    }

    /// Get a specific task.
    pub fn get_task(&self, id: &str) -> Option<&ComputeTask> {
        self.tasks.get(id)
    }

    /// Find pending tasks available for claiming.
    pub fn pending_tasks(&self) -> Vec<&ComputeTask> {
        self.tasks.values()
            .filter(|t| t.status == TaskStatus::Pending)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_submit_and_claim() {
        let mut sched = Scheduler::new("worker1");
        let spec = TaskSpec {
            command: "echo hello".to_string(),
            docker_image: None,
            timeout_secs: 60,
            cpu_cores: 1,
            memory_mb: 256,
        };
        let task = sched.submit_task(spec, "submitter1");
        assert_eq!(task.status, TaskStatus::Pending);

        let claimed = sched.claim_task(&task.id).unwrap();
        assert_eq!(claimed.status, TaskStatus::Running);
        assert_eq!(claimed.worker.as_deref(), Some("worker1"));
    }

    #[test]
    fn test_status() {
        let mut sched = Scheduler::new("w1");
        let spec = TaskSpec {
            command: "ls".to_string(),
            docker_image: None,
            timeout_secs: 60,
            cpu_cores: 1,
            memory_mb: 256,
        };
        sched.submit_task(spec.clone(), "s1");
        sched.submit_task(spec, "s2");

        let status = sched.status();
        assert_eq!(status.pending_tasks, 2);
        assert_eq!(status.running_tasks, 0);
    }

    #[test]
    fn test_receive_from_network() {
        let mut sched1 = Scheduler::new("w1");
        let mut sched2 = Scheduler::new("w2");

        let spec = TaskSpec {
            command: "date".to_string(),
            docker_image: None,
            timeout_secs: 30,
            cpu_cores: 1,
            memory_mb: 128,
        };
        let task = sched1.submit_task(spec, "submitter");

        // Simulate P2P: sched2 receives the task
        sched2.receive_task(task.clone());
        assert_eq!(sched2.pending_tasks().len(), 1);

        // sched2 claims it
        let claimed = sched2.claim_task(&task.id).unwrap();
        assert_eq!(claimed.worker.as_deref(), Some("w2"));
    }
}
