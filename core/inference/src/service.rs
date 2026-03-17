//! Local inference service — manages model backends (vLLM, llama.cpp, Ollama).
//! Wraps existing inference servers as subprocesses. Models load once, stay resident.

use crate::types::{
    ChatCompletionRequest, ChatCompletionResponse, GpuInfo, InferenceBackend, LoadedModel,
    NodeCapability,
};
use log::{info, warn};
use std::collections::HashMap;
use tokio::process::{Child, Command};

/// Manages local inference backends (one per loaded model).
pub struct InferenceService {
    /// Running backend processes.
    backends: HashMap<String, BackendProcess>,
    /// This node's wallet address.
    node_address: String,
    /// This node's peer ID.
    peer_id: String,
    /// Detected GPUs.
    gpus: Vec<GpuInfo>,
    /// Current in-flight requests.
    active_requests: u32,
    /// Max concurrent requests.
    max_concurrent: u32,
}

struct BackendProcess {
    model: LoadedModel,
    process: Option<Child>,
    /// For Ollama: the actual model name to use in API requests (e.g., "qwen3:8b").
    backend_model_name: Option<String>,
}

impl InferenceService {
    pub fn new(node_address: &str, peer_id: &str) -> Self {
        InferenceService {
            backends: HashMap::new(),
            node_address: node_address.to_string(),
            peer_id: peer_id.to_string(),
            gpus: Vec::new(),
            active_requests: 0,
            max_concurrent: 4,
        }
    }

    /// Detect GPUs on this machine.
    pub async fn detect_gpus(&mut self) {
        // Try nvidia-smi
        if let Ok(output) = Command::new("nvidia-smi")
            .args(["--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
            .output()
            .await
        {
            if output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                self.gpus = stdout
                    .lines()
                    .filter_map(|line| {
                        let parts: Vec<&str> = line.split(", ").collect();
                        if parts.len() == 2 {
                            Some(GpuInfo {
                                name: parts[0].trim().to_string(),
                                vram_mb: parts[1].trim().parse().unwrap_or(0),
                            })
                        } else {
                            None
                        }
                    })
                    .collect();
                info!("Detected {} GPU(s): {:?}", self.gpus.len(), self.gpus);
                return;
            }
        }

        // Try Apple Silicon (Metal)
        #[cfg(target_os = "macos")]
        {
            if let Ok(output) = Command::new("sysctl")
                .args(["-n", "hw.memsize"])
                .output()
                .await
            {
                if output.status.success() {
                    let mem_bytes: u64 = String::from_utf8_lossy(&output.stdout)
                        .trim()
                        .parse()
                        .unwrap_or(0);
                    if mem_bytes > 0 {
                        // Apple Silicon shares RAM with GPU
                        // Roughly 75% can be used for GPU tasks
                        let gpu_mem_mb = mem_bytes / 1024 / 1024 * 3 / 4;
                        self.gpus = vec![GpuInfo {
                            name: "Apple Silicon (Metal)".to_string(),
                            vram_mb: gpu_mem_mb,
                        }];
                        info!("Detected Apple Silicon: ~{} MB unified memory for GPU", gpu_mem_mb);
                        return;
                    }
                }
            }
        }

        info!("No GPU detected, inference will be CPU-only");
    }

    /// Load a model using a specified backend.
    /// This starts a persistent inference server as a subprocess.
    pub async fn load_model(
        &mut self,
        model_id: &str,
        model_path: &str,
        backend: InferenceBackend,
        port: u16,
    ) -> Result<(), String> {
        if self.backends.contains_key(model_id) {
            return Err(format!("model {} already loaded", model_id));
        }

        let child = match backend {
            InferenceBackend::LlamaCpp => {
                info!("Starting llama.cpp server for {} on port {}", model_id, port);
                Command::new("llama-server")
                    .args([
                        "-m", model_path,
                        "--port", &port.to_string(),
                        "-c", "4096",
                        "--host", "127.0.0.1",
                        "-np", "4", // 4 parallel slots
                    ])
                    .stdout(std::process::Stdio::piped())
                    .stderr(std::process::Stdio::piped())
                    .spawn()
                    .map_err(|e| format!("failed to start llama-server: {}", e))?
            }
            InferenceBackend::Vllm => {
                info!("Starting vLLM server for {} on port {}", model_id, port);
                Command::new("python3")
                    .args([
                        "-m", "vllm.entrypoints.openai.api_server",
                        "--model", model_path,
                        "--port", &port.to_string(),
                        "--host", "127.0.0.1",
                    ])
                    .stdout(std::process::Stdio::piped())
                    .stderr(std::process::Stdio::piped())
                    .spawn()
                    .map_err(|e| format!("failed to start vLLM: {}", e))?
            }
            InferenceBackend::Ollama => {
                // Ollama manages its own server, just pull the model
                info!("Pulling model {} via Ollama", model_id);
                let status = Command::new("ollama")
                    .args(["pull", model_path])
                    .status()
                    .await
                    .map_err(|e| format!("failed to run ollama pull: {}", e))?;
                if !status.success() {
                    return Err("ollama pull failed".to_string());
                }
                // Ollama serves on 11434 by default
                // No child process to manage (ollama serve runs separately)
                self.backends.insert(
                    model_id.to_string(),
                    BackendProcess {
                        model: LoadedModel {
                            model_id: model_id.to_string(),
                            backend: InferenceBackend::Ollama,
                            port: 11434,
                            max_context: 4096,
                            tokens_per_sec: 0.0, // will be measured
                        },
                        process: None,
                        backend_model_name: Some(model_path.to_string()),
                    },
                );
                info!("Model {} ready via Ollama (backend name: {})", model_id, model_path);
                return Ok(());
            }
            InferenceBackend::Custom => {
                // Custom backend: assume already running on the given port
                self.backends.insert(
                    model_id.to_string(),
                    BackendProcess {
                        model: LoadedModel {
                            model_id: model_id.to_string(),
                            backend: InferenceBackend::Custom,
                            port,
                            max_context: 4096,
                            tokens_per_sec: 0.0,
                        },
                        process: None,
                        backend_model_name: None,
                    },
                );
                info!("Registered custom backend for {} on port {}", model_id, port);
                return Ok(());
            }
        };

        self.backends.insert(
            model_id.to_string(),
            BackendProcess {
                model: LoadedModel {
                    model_id: model_id.to_string(),
                    backend,
                    port,
                    max_context: 4096,
                    tokens_per_sec: 0.0, // will be measured on first request
                },
                process: Some(child),
                backend_model_name: None,
            },
        );

        // Wait a moment for the server to start
        tokio::time::sleep(std::time::Duration::from_secs(2)).await;

        info!("Model {} loaded and serving on port {}", model_id, port);
        Ok(())
    }

    /// Unload a model, killing its backend process.
    pub async fn unload_model(&mut self, model_id: &str) -> Result<(), String> {
        if let Some(mut bp) = self.backends.remove(model_id) {
            if let Some(ref mut child) = bp.process {
                let _ = child.kill().await;
            }
            info!("Unloaded model {}", model_id);
            Ok(())
        } else {
            Err(format!("model {} not loaded", model_id))
        }
    }

    /// Forward an inference request to the local backend and return the response.
    pub async fn infer(
        &mut self,
        model_id: &str,
        request: &ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, String> {
        let bp = self
            .backends
            .get(model_id)
            .ok_or_else(|| format!("model {} not loaded", model_id))?;

        let url = match bp.model.backend {
            InferenceBackend::Ollama => {
                format!("http://127.0.0.1:{}/v1/chat/completions", bp.model.port)
            }
            _ => {
                format!("http://127.0.0.1:{}/v1/chat/completions", bp.model.port)
            }
        };

        self.active_requests += 1;

        let client = reqwest::Client::new();
        let result = client
            .post(&url)
            .json(request)
            .send()
            .await
            .map_err(|e| format!("backend request failed: {}", e))?
            .json::<ChatCompletionResponse>()
            .await
            .map_err(|e| format!("failed to parse backend response: {}", e));

        self.active_requests = self.active_requests.saturating_sub(1);
        result
    }

    /// Stream inference — returns chunks via a channel.
    pub async fn infer_stream(
        &mut self,
        model_id: &str,
        request: &ChatCompletionRequest,
    ) -> Result<reqwest::Response, String> {
        let bp = self
            .backends
            .get(model_id)
            .ok_or_else(|| format!("model {} not loaded", model_id))?;

        let url = format!("http://127.0.0.1:{}/v1/chat/completions", bp.model.port);

        // Force stream=true
        let mut req = request.clone();
        req.stream = Some(true);

        self.active_requests += 1;

        let client = reqwest::Client::new();
        let response = client
            .post(&url)
            .json(&req)
            .send()
            .await
            .map_err(|e| format!("backend stream request failed: {}", e))?;

        // Note: caller must decrement active_requests when done
        Ok(response)
    }

    /// Decrement active request count (call after stream completes).
    pub fn request_completed(&mut self) {
        self.active_requests = self.active_requests.saturating_sub(1);
    }

    /// Build a capability announcement for P2P broadcast.
    pub fn capability(&self) -> NodeCapability {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        NodeCapability {
            address: self.node_address.clone(),
            peer_id: self.peer_id.clone(),
            gpus: self.gpus.clone(),
            vram_total_mb: self.gpus.iter().map(|g| g.vram_mb).sum(),
            loaded_models: self.backends.values().map(|bp| bp.model.clone()).collect(),
            max_concurrent: self.max_concurrent,
            queue_depth: self.active_requests,
            timestamp: now,
        }
    }

    /// Check if this node has a specific model loaded.
    pub fn has_model(&self, model_id: &str) -> bool {
        self.backends.contains_key(model_id)
    }

    /// List loaded models.
    pub fn loaded_models(&self) -> Vec<String> {
        self.backends.keys().cloned().collect()
    }

    /// Get backend port for a model (for local proxying).
    pub fn model_port(&self, model_id: &str) -> Option<u16> {
        self.backends.get(model_id).map(|bp| bp.model.port)
    }

    /// Shutdown all backends.
    pub async fn shutdown(&mut self) {
        for (model_id, mut bp) in self.backends.drain() {
            if let Some(ref mut child) = bp.process {
                warn!("Shutting down backend for {}", model_id);
                let _ = child.kill().await;
            }
        }
    }
}
