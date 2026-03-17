//! OpenAI-compatible HTTP API gateway.
//! Listens on a local port, routes requests to inference nodes (local or remote via P2P).

use crate::registry::NodeRegistry;
use crate::types::{
    ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, ChatChoice, ChatMessage,
    InferenceMessage, ModelInfo, ModelList,
};
use axum::{
    extract::State,
    http::StatusCode,
    response::{
        sse::{Event, Sse},
        IntoResponse, Json, Response,
    },
    routing::{get, post},
    Router,
};
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex};

/// Information about a local backend.
#[derive(Debug, Clone)]
pub struct LocalBackendInfo {
    /// Port the backend listens on.
    pub port: u16,
    /// Model name expected by the backend (e.g., "qwen2.5:0.5b" for Ollama).
    pub backend_model_name: Option<String>,
}

/// Shared state for the API gateway.
pub struct GatewayState {
    /// Node capability registry.
    pub registry: Mutex<NodeRegistry>,
    /// Channel to send P2P inference requests.
    pub p2p_request_tx: mpsc::Sender<InferenceMessage>,
    /// This node's address.
    pub node_address: String,
    /// This node's peer_id.
    pub peer_id: String,
    /// Local inference backends (model_id → backend info).
    pub local_backends: Mutex<std::collections::HashMap<String, LocalBackendInfo>>,
    /// Pending remote requests: request_id → response channel.
    pub pending_requests: Arc<Mutex<std::collections::HashMap<String, mpsc::Sender<StreamEvent>>>>,
}

#[derive(Debug, Clone)]
pub enum StreamEvent {
    Chunk(String),
    Done,
    Error(String),
}

/// Create the API gateway router.
pub fn create_router(state: Arc<GatewayState>) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/models", get(list_models))
        .route("/health", get(health))
        .with_state(state)
}

/// Start the API gateway on the given port.
pub async fn start_gateway(state: Arc<GatewayState>, port: u16) -> Result<(), String> {
    let app = create_router(state);
    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{}", port))
        .await
        .map_err(|e| format!("failed to bind gateway: {}", e))?;

    log::info!("API gateway listening on port {}", port);

    tokio::spawn(async move {
        if let Err(e) = axum::serve(listener, app).await {
            log::error!("Gateway server error: {}", e);
        }
    });

    Ok(())
}

async fn health() -> &'static str {
    "ok"
}

async fn list_models(State(state): State<Arc<GatewayState>>) -> Json<ModelList> {
    let registry = state.registry.lock().await;
    let models = registry.available_models();

    Json(ModelList {
        object: "list".to_string(),
        data: models
            .into_iter()
            .map(|(id, count)| ModelInfo {
                id,
                object: "model".to_string(),
                owned_by: "ynet-network".to_string(),
                ynet_nodes: Some(count),
            })
            .collect(),
    })
}

type ApiError = (StatusCode, Json<serde_json::Value>);

fn err_json(status: StatusCode, msg: &str) -> ApiError {
    (status, Json(serde_json::json!({"error": {"message": msg}})))
}

async fn chat_completions(
    State(state): State<Arc<GatewayState>>,
    Json(request): Json<ChatCompletionRequest>,
) -> Result<Response, ApiError> {
    let model_id = &request.model;
    let is_stream = request.stream.unwrap_or(false);

    // 1. Check if we have a local backend for this model
    let local_backend = {
        let backends = state.local_backends.lock().await;
        backends.get(model_id).cloned()
    };

    if let Some(backend_info) = local_backend {
        return handle_local_request(backend_info, request, is_stream).await;
    }

    // 2. Find a remote node via registry
    let target_peer = {
        let registry = state.registry.lock().await;
        let candidates = registry.find_nodes_for_model(model_id);
        candidates.first().map(|c| c.peer_id.clone())
    };

    let target_peer = target_peer.ok_or_else(|| {
        err_json(
            StatusCode::NOT_FOUND,
            &format!("model '{}' not available on any node", model_id),
        )
    })?;

    // 3. Forward request to remote node via P2P
    handle_remote_request(state, target_peer, request, is_stream).await
}

async fn handle_local_request(
    backend_info: LocalBackendInfo,
    request: ChatCompletionRequest,
    is_stream: bool,
) -> Result<Response, ApiError> {
    let url = format!("http://127.0.0.1:{}/v1/chat/completions", backend_info.port);
    // Create client that bypasses system proxy for localhost connections
    let client = reqwest::Client::builder()
        .no_proxy()
        .build()
        .map_err(|e| err_json(StatusCode::INTERNAL_SERVER_ERROR, &format!("failed to create client: {}", e)))?;

    // Map model name if backend requires a specific name
    let mut req = request.clone();
    if let Some(ref backend_model) = backend_info.backend_model_name {
        req.model = backend_model.clone();
    }

    if is_stream {
        req.stream = Some(true);

        let response = client
            .post(&url)
            .json(&req)
            .send()
            .await
            .map_err(|e| err_json(StatusCode::BAD_GATEWAY, &e.to_string()))?;

        let stream = async_stream::stream! {
            use futures::StreamExt;
            let mut byte_stream = response.bytes_stream();
            let mut buffer = String::new();

            while let Some(chunk) = byte_stream.next().await {
                match chunk {
                    Ok(bytes) => {
                        buffer.push_str(&String::from_utf8_lossy(&bytes));
                        while let Some(pos) = buffer.find("\n\n") {
                            let line = buffer[..pos].to_string();
                            buffer = buffer[pos + 2..].to_string();
                            if line.starts_with("data: ") {
                                let data = &line[6..];
                                if data == "[DONE]" {
                                    yield Ok::<Event, std::convert::Infallible>(
                                        Event::default().data("[DONE]")
                                    );
                                } else {
                                    yield Ok(Event::default().data(data.to_string()));
                                }
                            }
                        }
                    }
                    Err(e) => {
                        log::warn!("Stream error: {}", e);
                        break;
                    }
                }
            }
        };

        Ok(Sse::new(stream).into_response())
    } else {
        let response = client
            .post(&url)
            .json(&req)
            .send()
            .await
            .map_err(|e| err_json(StatusCode::BAD_GATEWAY, &e.to_string()))?;

        // Get raw text first for debugging
        let response_text = response.text().await
            .map_err(|e| err_json(StatusCode::BAD_GATEWAY, &format!("failed to read response: {}", e)))?;

        let body: ChatCompletionResponse = serde_json::from_str(&response_text)
            .map_err(|e| {
                log::error!("Failed to parse response: {}. Raw response: {}", e, &response_text[..response_text.len().min(500)]);
                err_json(StatusCode::BAD_GATEWAY, &format!("error decoding response body: {}", e))
            })?;

        Ok(Json(body).into_response())
    }
}

async fn handle_remote_request(
    state: Arc<GatewayState>,
    _target_peer: String,
    request: ChatCompletionRequest,
    is_stream: bool,
) -> Result<Response, ApiError> {
    let request_id = uuid::Uuid::new_v4().to_string();

    // Set up a channel to receive the response
    let (tx, mut rx) = mpsc::channel::<StreamEvent>(64);
    {
        let mut pending = state.pending_requests.lock().await;
        pending.insert(request_id.clone(), tx);
    }

    // Send inference request via P2P
    let msg = InferenceMessage::InferenceRequest {
        request_id: request_id.clone(),
        from_peer: state.peer_id.clone(),
        request: request.clone(),
    };
    state
        .p2p_request_tx
        .send(msg)
        .await
        .map_err(|_| err_json(StatusCode::INTERNAL_SERVER_ERROR, "P2P channel closed"))?;

    if is_stream {
        let pending_requests = Arc::clone(&state.pending_requests);
        let req_id = request_id.clone();

        let stream = async_stream::stream! {
            let timeout = tokio::time::Duration::from_secs(120);
            let start = tokio::time::Instant::now();

            loop {
                let remaining = timeout.saturating_sub(start.elapsed());
                if remaining.is_zero() {
                    yield Ok::<Event, std::convert::Infallible>(
                        Event::default().data(
                            serde_json::json!({"error": "timeout"}).to_string()
                        )
                    );
                    break;
                }

                match tokio::time::timeout(remaining, rx.recv()).await {
                    Ok(Some(StreamEvent::Chunk(data))) => {
                        yield Ok(Event::default().data(data));
                    }
                    Ok(Some(StreamEvent::Done)) => {
                        yield Ok(Event::default().data("[DONE]"));
                        break;
                    }
                    Ok(Some(StreamEvent::Error(e))) => {
                        yield Ok(Event::default().data(
                            serde_json::json!({"error": e}).to_string()
                        ));
                        break;
                    }
                    Ok(None) | Err(_) => break,
                }
            }

            // Cleanup
            let mut pending = pending_requests.lock().await;
            pending.remove(&req_id);
        };

        Ok(Sse::new(stream).into_response())
    } else {
        // Non-streaming: collect all chunks into a single response
        let mut full_content = String::new();
        let timeout = tokio::time::Duration::from_secs(120);

        loop {
            match tokio::time::timeout(timeout, rx.recv()).await {
                Ok(Some(StreamEvent::Chunk(data))) => {
                    if let Ok(chunk) = serde_json::from_str::<ChatCompletionChunk>(&data) {
                        if let Some(choice) = chunk.choices.first() {
                            if let Some(delta) = &choice.delta {
                                full_content.push_str(&delta.content);
                            }
                        }
                    } else {
                        full_content.push_str(&data);
                    }
                }
                Ok(Some(StreamEvent::Done)) => break,
                Ok(Some(StreamEvent::Error(e))) => {
                    let mut pending = state.pending_requests.lock().await;
                    pending.remove(&request_id);
                    return Err(err_json(StatusCode::BAD_GATEWAY, &e));
                }
                Ok(None) | Err(_) => break,
            }
        }

        {
            let mut pending = state.pending_requests.lock().await;
            pending.remove(&request_id);
        }

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Ok(Json(ChatCompletionResponse {
            id: format!("chatcmpl-{}", &request_id[..8]),
            object: "chat.completion".to_string(),
            created: now,
            model: request.model,
            system_fingerprint: None,
            choices: vec![ChatChoice {
                index: 0,
                message: Some(ChatMessage {
                    role: "assistant".to_string(),
                    content: full_content,
                }),
                delta: None,
                finish_reason: Some("stop".to_string()),
            }],
            usage: None,
            ynet_cost: None,
        })
        .into_response())
    }
}
