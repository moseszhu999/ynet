//! yNet node - main entry point
//! Runs P2P network + blockchain + scheduler + IPC bridge to Electron frontend.
//! Block producer elected via deterministic random: hash(prev_hash + address + slot).
//! Lowest score wins. ~3 second block time.

use clap::Parser;
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::path::PathBuf;
use tokio::io::{AsyncBufReadExt, BufReader};
use ynet_chain::{
    Block, Blockchain, EconomicsState, FeeCalculator, ModelPricing, PaymentCurrency,
    Transaction, TxType, Wallet, TARGET_BLOCK_TIME_SECS,
};
use ynet_inference::{
    gateway, InferenceBackend, InferenceMessage, InferenceService,
    NodeRegistry, StreamEvent,
};
use ynet_p2p::NetworkNode;
use ynet_scheduler::{Scheduler, SchedulerMessage, TaskSpec};

#[derive(Parser)]
#[command(name = "ynet-node", about = "yNet decentralized compute node")]
struct Args {
    /// P2P listen port
    #[arg(short, long, default_value_t = 0)]
    port: u16,

    /// Seed node addresses (multiaddr format)
    #[arg(short, long)]
    seed: Vec<String>,

    /// Data directory for wallet and chain storage
    #[arg(short, long)]
    data_dir: Option<PathBuf>,

    /// Accept remote tasks from P2P network (default: false for safety)
    #[arg(long, default_value_t = false)]
    accept_remote_tasks: bool,

    /// Enable API gateway on this port (OpenAI-compatible inference API)
    #[arg(long)]
    api_port: Option<u16>,

    /// Load a model at startup: "model_id:backend:path:port"
    /// e.g., "kimi-k2.5-q4:llamacpp:/models/kimi-k2.5.gguf:8080"
    #[arg(long)]
    load_model: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct Request {
    id: u64,
    method: String,
    #[serde(default)]
    params: serde_json::Value,
}

#[derive(Debug, Serialize)]
struct Response {
    id: u64,
    result: serde_json::Value,
}

fn send_response(resp: &Response) {
    let mut stdout = std::io::stdout().lock();
    let _ = writeln!(stdout, "{}", serde_json::to_string(resp).unwrap());
    let _ = stdout.flush();
}

fn send_event(event_type: &str, data: serde_json::Value) {
    let mut stdout = std::io::stdout().lock();
    let _ = writeln!(
        stdout,
        "{}",
        serde_json::json!({"event": event_type, "data": data})
    );
    let _ = stdout.flush();
}

fn load_or_create_wallet(data_dir: &Option<PathBuf>) -> Wallet {
    if let Some(dir) = data_dir {
        let wallet_file = dir.join("wallet.json");
        if wallet_file.exists() {
            if let Ok(data) = std::fs::read_to_string(&wallet_file) {
                if let Ok(wd) = serde_json::from_str::<ynet_chain::wallet::WalletData>(&data) {
                    if let Ok(w) = Wallet::from_data(&wd) {
                        log::info!("Wallet loaded: {}", w.address());
                        return w;
                    }
                }
            }
        }
        let w = Wallet::generate();
        let _ = std::fs::create_dir_all(dir);
        // Set restrictive permissions on wallet file
        let wallet_json = serde_json::to_string_pretty(&w.to_data()).unwrap();
        let _ = std::fs::write(&wallet_file, &wallet_json);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = std::fs::set_permissions(&wallet_file, std::fs::Permissions::from_mode(0o600));
        }
        log::info!("New wallet created: {}", w.address());
        w
    } else {
        let w = Wallet::generate();
        log::info!("Ephemeral wallet: {}", w.address());
        w
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .target(env_logger::Target::Stderr)
        .init();

    let args = Args::parse();

    log::info!("Starting yNet node on port {} ...", args.port);
    if args.accept_remote_tasks {
        log::info!("Accepting remote tasks from P2P network");
    }

    // Initialize wallet, blockchain, scheduler
    let wallet = load_or_create_wallet(&args.data_dir);
    let my_address = wallet.address();
    let mut blockchain = Blockchain::new();
    let mut scheduler = Scheduler::new(&my_address);
    let mut economics = EconomicsState::new();
    let accept_remote = args.accept_remote_tasks;

    // Start P2P network
    let mut network = NetworkNode::start(args.port, args.seed).await?;
    log::info!("P2P node started: {}", network.local_peer_id());
    log::info!("Wallet address: {}", my_address);

    // ---- Inference layer ----
    let mut inference_service = InferenceService::new(&my_address, network.local_peer_id());
    inference_service.detect_gpus().await;

    let mut inference_registry = NodeRegistry::new();

    // P2P channel for inference requests from the gateway
    let (p2p_inference_tx, mut p2p_inference_rx) = tokio::sync::mpsc::channel::<InferenceMessage>(64);

    // Gateway state (shared with API gateway)
    let gateway_state = std::sync::Arc::new(gateway::GatewayState {
        registry: tokio::sync::Mutex::new(NodeRegistry::new()),
        p2p_request_tx: p2p_inference_tx,
        node_address: my_address.clone(),
        peer_id: network.local_peer_id().to_string(),
        local_backends: tokio::sync::Mutex::new(std::collections::HashMap::new()),
        pending_requests: std::sync::Arc::new(tokio::sync::Mutex::new(std::collections::HashMap::new())),
    });

    // Load models from CLI args
    // Format: model_id:backend:path:port[:backend_model_name]
    // backend_model_name is optional and used when the backend expects a different model name
    // e.g., for Ollama: "qwen:custom::11434:qwen2.5:0.5b" (backend_model_name can contain colons)
    for spec in &args.load_model {
        let parts: Vec<&str> = spec.split(':').collect();
        if parts.len() >= 4 {
            let model_id = parts[0];
            let backend = match parts[1] {
                "llamacpp" | "llama" => InferenceBackend::LlamaCpp,
                "vllm" => InferenceBackend::Vllm,
                "ollama" => InferenceBackend::Ollama,
                _ => InferenceBackend::Custom,
            };
            let model_path = parts[2];
            let port: u16 = parts[3].parse().unwrap_or(8080);
            // Optional 5th+ parts: backend model name (joins remaining parts with ':')
            // This allows model names like "qwen2.5:0.5b" that contain colons
            let backend_model_name = if parts.len() > 4 {
                Some(parts[4..].join(":"))
            } else {
                None
            };

            match inference_service.load_model(model_id, model_path, backend, port).await {
                Ok(()) => {
                    let mut backends = gateway_state.local_backends.lock().await;
                    backends.insert(model_id.to_string(), gateway::LocalBackendInfo {
                        port,
                        backend_model_name,
                    });
                    log::info!("Model {} loaded on port {}", model_id, port);
                }
                Err(e) => log::error!("Failed to load model {}: {}", model_id, e),
            }
        } else {
            log::error!("Invalid --load-model format: '{}'. Expected model_id:backend:path:port[:backend_model_name]", spec);
        }
    }

    // Start API gateway if requested
    if let Some(api_port) = args.api_port {
        if let Err(e) = gateway::start_gateway(gateway_state.clone(), api_port).await {
            log::error!("Failed to start API gateway: {}", e);
        }
    }

    // Capability broadcast timer (every 10 seconds)
    let mut cap_timer = tokio::time::interval(tokio::time::Duration::from_secs(10));
    cap_timer.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    // Emit ready signal for the bridge
    send_event("ready", serde_json::json!({"address": my_address}));

    // IPC from stdin
    let stdin = BufReader::new(tokio::io::stdin());
    let mut lines = stdin.lines();
    let mut stdin_open = true;

    // Block slot timer — check every 500ms for sub-second precision
    let mut slot_timer = tokio::time::interval(tokio::time::Duration::from_millis(500));
    slot_timer.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let mut last_produced_slot: u64 = 0;

    loop {
        tokio::select! {
            line = lines.next_line(), if stdin_open => {
                match line {
                    Ok(Some(line)) if !line.trim().is_empty() => {
                        let req: Request = match serde_json::from_str(&line) {
                            Ok(r) => r,
                            Err(e) => {
                                send_response(&Response {
                                    id: 0,
                                    result: serde_json::json!({"error": e.to_string()}),
                                });
                                continue;
                            }
                        };

                        let result = handle_request(
                            &req, &network, &mut blockchain, &wallet,
                            &mut scheduler, &mut economics,
                            &mut inference_service, &inference_registry,
                        ).await;
                        send_response(&Response { id: req.id, result });
                    }
                    Ok(None) => {
                        log::info!("stdin closed, running in standalone mode");
                        stdin_open = false;
                    }
                    _ => continue,
                }
            }

            event = network.recv_event() => {
                match event {
                    Some(ynet_p2p::NetworkEvent::PeerConnected(peer)) => {
                        send_event("peer_connected", serde_json::json!({"peer_id": peer}));
                    }
                    Some(ynet_p2p::NetworkEvent::PeerDisconnected(peer)) => {
                        send_event("peer_disconnected", serde_json::json!({"peer_id": peer}));
                    }
                    Some(ynet_p2p::NetworkEvent::MessageReceived { from, topic, data }) => {
                        match topic.as_str() {
                            "ynet-chain" => {
                                if let Ok(msg) = serde_json::from_slice::<ChainMessage>(&data) {
                                    handle_chain_message(msg, &mut blockchain, &from);
                                }
                            }
                            "ynet-tasks" => {
                                if let Ok(msg) = serde_json::from_slice::<SchedulerMessage>(&data) {
                                    handle_scheduler_message(
                                        msg, &mut scheduler, &mut blockchain,
                                        &wallet, &network, &my_address, accept_remote,
                                    ).await;
                                }
                            }
                            "ynet-inference" => {
                                if let Ok(msg) = serde_json::from_slice::<InferenceMessage>(&data) {
                                    handle_inference_message(
                                        msg, &mut inference_service, &mut inference_registry,
                                        &network, &gateway_state,
                                    ).await;
                                }
                            }
                            _ => {}
                        }
                        send_event("message", serde_json::json!({
                            "from": from,
                            "topic": topic,
                            "data": String::from_utf8_lossy(&data),
                        }));
                    }
                    None => break,
                }
            }

            // Forward inference requests from API gateway to P2P
            Some(msg) = p2p_inference_rx.recv() => {
                if let Ok(data) = serde_json::to_vec(&msg) {
                    network.broadcast("ynet-inference", data).await;
                }
            }

            // Broadcast inference capabilities periodically
            _ = cap_timer.tick() => {
                if !inference_service.loaded_models().is_empty() {
                    let cap = inference_service.capability();
                    let msg = InferenceMessage::NodeCapability(cap.clone());
                    if let Ok(data) = serde_json::to_vec(&msg) {
                        network.broadcast("ynet-inference", data).await;
                    }
                    // Update our own gateway's registry too
                    let mut reg = gateway_state.registry.lock().await;
                    reg.update(cap);
                }
                // Evict stale entries
                inference_registry.evict_stale();
            }

            // Slot-based block production with deterministic random election
            _ = slot_timer.tick() => {
                let current_slot = Block::current_slot();

                // Only try once per slot, and only if there are pending transactions
                if current_slot > last_produced_slot
                    && !blockchain.get_pending_transactions().is_empty()
                {
                    let prev_hash = &blockchain.latest_block().hash;
                    let my_score = Block::compute_election_score(prev_hash, &my_address, current_slot);

                    // Score-based delay: lower score → shorter wait
                    // First 2 hex chars (0-255) → delay 0-2500ms
                    let score_prefix = u8::from_str_radix(&my_score[..2], 16).unwrap_or(128);
                    let delay_ms = (score_prefix as u64 * 10).min(2500);

                    // Use millisecond precision for elapsed time (#8 fix)
                    let slot_start_ms = current_slot * TARGET_BLOCK_TIME_SECS * 1000;
                    let now_ms = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_millis() as u64;
                    let elapsed_in_slot_ms = now_ms.saturating_sub(slot_start_ms);

                    if elapsed_in_slot_ms >= delay_ms {
                        if blockchain.latest_block().slot < current_slot {
                            last_produced_slot = current_slot;
                            let block = blockchain.produce_block(&my_address);
                            let msg = ChainMessage::NewBlock(block.clone());
                            if let Ok(data) = serde_json::to_vec(&msg) {
                                network.broadcast("ynet-chain", data).await;
                            }
                            send_event("new_block", serde_json::json!({
                                "index": block.index, "hash": block.hash,
                                "producer": block.producer,
                                "slot": block.slot,
                                "election_score": &block.election_score[..16],
                            }));
                        }
                    }
                }
            }
        }
    }

    Ok(())
}

// ---- Chain messages ----

#[derive(Debug, Serialize, Deserialize)]
enum ChainMessage {
    NewBlock(ynet_chain::Block),
    NewTransaction(Transaction),
}

fn handle_chain_message(msg: ChainMessage, blockchain: &mut Blockchain, from: &str) {
    match msg {
        ChainMessage::NewBlock(block) => {
            match blockchain.accept_block(block.clone()) {
                Ok(()) => {
                    log::info!("Accepted block #{} from {}", block.index, from);
                    send_event("new_block", serde_json::json!({
                        "index": block.index, "hash": block.hash,
                        "producer": block.producer, "from": from,
                    }));
                }
                Err(e) => log::warn!("Rejected block #{} from {}: {}", block.index, from, e),
            }
        }
        ChainMessage::NewTransaction(tx) => {
            if let Err(e) = blockchain.add_transaction(tx) {
                log::warn!("Rejected transaction from {}: {}", from, e);
            }
        }
    }
}

// ---- Scheduler messages ----

async fn handle_scheduler_message(
    msg: SchedulerMessage,
    scheduler: &mut Scheduler,
    blockchain: &mut Blockchain,
    wallet: &Wallet,
    network: &NetworkNode,
    my_address: &str,
    accept_remote: bool,
) {
    match msg {
        SchedulerMessage::TaskSubmitted(task) => {
            scheduler.receive_task(task.clone());

            // Only auto-claim remote tasks if explicitly opted in (#1 fix)
            if accept_remote && scheduler.status().running_tasks == 0 {
                // Only execute tasks that require Docker isolation from network
                if task.spec.docker_image.is_some() {
                    if let Some(claimed) = scheduler.claim_task(&task.id) {
                        let claim_msg = SchedulerMessage::TaskClaimed {
                            task_id: claimed.id.clone(),
                            worker_id: my_address.to_string(),
                        };
                        if let Ok(data) = serde_json::to_vec(&claim_msg) {
                            network.broadcast("ynet-tasks", data).await;
                        }

                        execute_and_report(
                            &claimed.id, scheduler, blockchain, wallet, network, my_address,
                        ).await;
                    }
                } else {
                    log::warn!(
                        "Ignoring remote task {} without Docker isolation",
                        task.id
                    );
                }
            }

            send_event("task_received", serde_json::to_value(&task).unwrap());
        }
        SchedulerMessage::TaskClaimed { task_id, worker_id } => {
            scheduler.receive_claim(&task_id, &worker_id);
            send_event("task_claimed", serde_json::json!({
                "task_id": task_id, "worker_id": worker_id,
            }));
        }
        SchedulerMessage::TaskCompleted(result) => {
            let success = result.success;
            let task_id = result.task_id.clone();
            scheduler.receive_result(result);
            send_event("task_completed", serde_json::json!({
                "task_id": task_id, "success": success,
            }));
        }
    }
}

async fn execute_and_report(
    task_id: &str,
    scheduler: &mut Scheduler,
    blockchain: &mut Blockchain,
    wallet: &Wallet,
    network: &NetworkNode,
    my_address: &str,
) {
    if let Some(result) = scheduler.execute_task(task_id).await {
        let msg = SchedulerMessage::TaskCompleted(result.clone());
        if let Ok(data) = serde_json::to_vec(&msg) {
            network.broadcast("ynet-tasks", data).await;
        }

        // Record on chain: ComputeProof (worker signs proving it did the work)
        // #3 fix: worker signs from its own address, recording the proof of computing
        if result.success {
            if let Some(task) = scheduler.get_task(task_id) {
                let tx = Transaction::new(
                    my_address,
                    &task.submitter,
                    task.price,
                    TxType::ComputeProof,
                    wallet.signing_key(),
                );
                let _ = blockchain.add_transaction(tx);
            }
        }

        send_event("task_executed", serde_json::json!({
            "task_id": task_id,
            "success": result.success,
            "exit_code": result.exit_code,
            "duration": result.duration_secs,
            "stdout_preview": &result.stdout[..result.stdout.len().min(200)],
        }));
    }
}

// ---- Inference message handler ----

/// Execute inference and stream chunks back via P2P.
/// This helper function is used by both Phase 1 and Phase 2 inference handlers.
async fn stream_inference_response(
    service: &mut InferenceService,
    model_id: &str,
    request: &ynet_inference::ChatCompletionRequest,
    request_id: &str,
    to_peer: &str,
    network: &NetworkNode,
) {
    match service.infer_stream(model_id, request).await {
        Ok(response) => {
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
                                let done = data == "[DONE]";
                                let msg = InferenceMessage::InferenceChunk {
                                    request_id: request_id.to_string(),
                                    to_peer: to_peer.to_string(),
                                    chunk: data.to_string(),
                                    done,
                                };
                                if let Ok(data) = serde_json::to_vec(&msg) {
                                    network.broadcast("ynet-inference", data).await;
                                }
                                if done {
                                    return;
                                }
                            }
                        }
                    }
                    Err(e) => {
                        let msg = InferenceMessage::InferenceError {
                            request_id: request_id.to_string(),
                            to_peer: to_peer.to_string(),
                            error: e.to_string(),
                        };
                        if let Ok(data) = serde_json::to_vec(&msg) {
                            network.broadcast("ynet-inference", data).await;
                        }
                        return;
                    }
                }
            }
            service.request_completed();
        }
        Err(e) => {
            let msg = InferenceMessage::InferenceError {
                request_id: request_id.to_string(),
                to_peer: to_peer.to_string(),
                error: e,
            };
            if let Ok(data) = serde_json::to_vec(&msg) {
                network.broadcast("ynet-inference", data).await;
            }
        }
    }
}

async fn handle_inference_message(
    msg: InferenceMessage,
    service: &mut InferenceService,
    registry: &mut NodeRegistry,
    network: &NetworkNode,
    gateway_state: &std::sync::Arc<gateway::GatewayState>,
) {
    match msg {
        InferenceMessage::NodeCapability(cap) => {
            log::debug!("Got capability from {}: {} models", cap.peer_id, cap.loaded_models.len());
            registry.update(cap.clone());
            // Also update gateway's registry
            let mut reg = gateway_state.registry.lock().await;
            reg.update(cap);
        }

        InferenceMessage::InferenceRequest {
            request_id,
            from_peer,
            request,
        } => {
            let model_id = &request.model;
            if !service.has_model(model_id) {
                return;
            }

            log::info!("Processing inference request {} for model {}", request_id, model_id);
            stream_inference_response(service, model_id, &request, &request_id, &from_peer, network).await;
        }

        InferenceMessage::InferenceChunk {
            request_id,
            to_peer,
            chunk,
            done,
        } => {
            // This chunk is for us if to_peer matches our peer_id
            if to_peer != network.local_peer_id() {
                return;
            }

            let pending = gateway_state.pending_requests.lock().await;
            if let Some(tx) = pending.get(&request_id) {
                if done {
                    let _ = tx.send(StreamEvent::Done).await;
                } else {
                    let _ = tx.send(StreamEvent::Chunk(chunk)).await;
                }
            }
        }

        InferenceMessage::InferenceError {
            request_id,
            to_peer,
            error,
        } => {
            if to_peer != network.local_peer_id() {
                return;
            }

            let pending = gateway_state.pending_requests.lock().await;
            if let Some(tx) = pending.get(&request_id) {
                let _ = tx.send(StreamEvent::Error(error)).await;
            }
        }

        // === Phase 2: Distributed inference messages ===

        InferenceMessage::NodeAnnouncement(ann) => {
            log::debug!(
                "Got announcement from {}: {} models, load {:.2}",
                ann.peer_id,
                ann.models.len(),
                ann.load
            );
            registry.update_announcement(ann.clone());
            let mut reg = gateway_state.registry.lock().await;
            reg.update_announcement(ann);
        }

        InferenceMessage::NodeHeartbeat {
            peer_id,
            timestamp: _,
            queue_depth,
            load,
        } => {
            log::trace!("Heartbeat from {}: queue={}, load={:.2}", peer_id, queue_depth, load);
            // Update registry with latest status
            if let Some(cap) = registry.get_node(&peer_id) {
                let mut updated = cap.clone();
                updated.queue_depth = queue_depth;
                registry.update(updated.clone());
                let mut reg = gateway_state.registry.lock().await;
                reg.update(updated);
            }
            if let Some(ann) = registry.get_announcement(&peer_id) {
                let mut updated_ann = ann.clone();
                updated_ann.queue_depth = queue_depth;
                updated_ann.load = load;
                registry.update_announcement(updated_ann.clone());
                let mut reg = gateway_state.registry.lock().await;
                reg.update_announcement(updated_ann);
            }
        }

        InferenceMessage::NodeListRequest {
            model_filter,
            max_price,
        } => {
            log::debug!("Node list request: filter={:?}, max_price={:?}", model_filter, max_price);
            // Respond with available nodes matching the filter
            let nodes = if let Some(model_id) = &model_filter {
                registry.find_nodes_with_pricing(model_id, max_price)
            } else {
                // Return all nodes
                registry.get_all_nodes_info()
            };
            let response = InferenceMessage::NodeListResponse { nodes };
            if let Ok(data) = serde_json::to_vec(&response) {
                network.broadcast("ynet-inference", data).await;
            }
        }

        InferenceMessage::NodeListResponse { nodes } => {
            log::debug!("Received node list with {} nodes", nodes.len());
            // Update local registry with received nodes
            // This is informational - the actual announcements are more detailed
            for node in nodes {
                log::trace!(
                    "Available: {} at {} for {} (price: {}/1k)",
                    node.peer_id,
                    node.address,
                    node.model_id,
                    node.price_input_per_1k
                );
            }
        }

        InferenceMessage::RoutedInference {
            request_id,
            target_peer,
            from_peer,
            request,
            max_price,
        } => {
            // Only process if we're the target
            if target_peer != network.local_peer_id() {
                return;
            }

            let model_id = &request.model;
            if !service.has_model(model_id) {
                log::warn!("Routed request {} for model {} but we don't have it", request_id, model_id);
                return;
            }

            log::info!(
                "Processing routed inference {} for model {} from {} (max_price: {:?})",
                request_id,
                model_id,
                from_peer,
                max_price
            );

            stream_inference_response(service, model_id, &request, &request_id, &from_peer, network).await;
        }

        InferenceMessage::InferenceBilling {
            request_id,
            node_address,
            user_address,
            input_tokens,
            output_tokens,
            price_per_1k,
            total_cost_nynet,
        } => {
            log::info!(
                "Billing record for {}: node={}, user={}, cost={} nYNET ({}+{} tokens @ {}/1k)",
                request_id,
                node_address,
                user_address,
                total_cost_nynet,
                input_tokens,
                output_tokens,
                price_per_1k
            );
            // In a full implementation, this would be recorded on-chain
            // For MVP, we just log it
        }
    }
}

// ---- IPC request handler ----

async fn handle_request(
    req: &Request,
    network: &NetworkNode,
    blockchain: &mut Blockchain,
    wallet: &Wallet,
    scheduler: &mut Scheduler,
    economics: &mut EconomicsState,
    inference_service: &mut InferenceService,
    inference_registry: &NodeRegistry,
) -> serde_json::Value {
    match req.method.as_str() {
        "get_status" => {
            let p2p = network.get_status().await;
            let chain = blockchain.status();
            let wallet_info = blockchain.wallet_info(&wallet.address());
            let sched = scheduler.status();
            let econ = economics.summary();
            serde_json::json!({
                "p2p": p2p, "chain": chain,
                "wallet": wallet_info, "scheduler": sched,
                "economics": econ,
            })
        }
        "get_network" => {
            serde_json::to_value(network.get_status().await).unwrap()
        }
        "get_wallet" => {
            let info = blockchain.wallet_info(&wallet.address());
            serde_json::json!({
                "address": info.address,
                "balance": info.balance,
                "chain_height": blockchain.height(),
            })
        }
        "get_chain" => {
            let status = blockchain.status();
            let recent_txs: Vec<_> = blockchain
                .recent_transactions(20)
                .iter()
                .map(|tx| serde_json::to_value(tx).unwrap())
                .collect();
            serde_json::json!({
                "status": status,
                "recent_transactions": recent_txs,
            })
        }
        "transfer" => {
            let to = req.params.get("to").and_then(|v| v.as_str()).unwrap_or("");
            let amount = req.params.get("amount").and_then(|v| v.as_u64()).unwrap_or(0);
            if to.is_empty() || amount == 0 {
                return serde_json::json!({"error": "invalid 'to' or 'amount'"});
            }
            let tx = Transaction::new(
                &wallet.address(), to, amount, TxType::Transfer, wallet.signing_key(),
            );
            match blockchain.add_transaction(tx.clone()) {
                Ok(()) => {
                    let msg = ChainMessage::NewTransaction(tx.clone());
                    if let Ok(data) = serde_json::to_vec(&msg) {
                        network.broadcast("ynet-chain", data).await;
                    }
                    serde_json::json!({"ok": true, "tx_hash": tx.hash})
                }
                Err(e) => serde_json::json!({"error": e}),
            }
        }

        // ---- Scheduler IPC ----
        "get_scheduler" => {
            serde_json::to_value(scheduler.status()).unwrap()
        }
        "submit_task" => {
            let command = req.params.get("command").and_then(|v| v.as_str()).unwrap_or("");
            let docker_image = req.params.get("docker_image").and_then(|v| v.as_str()).map(String::from);
            let timeout = req.params.get("timeout").and_then(|v| v.as_u64()).unwrap_or(300);
            let cpu = req.params.get("cpu_cores").and_then(|v| v.as_u64()).unwrap_or(1) as u32;
            let mem = req.params.get("memory_mb").and_then(|v| v.as_u64()).unwrap_or(256) as u32;
            let price = req.params.get("price").and_then(|v| v.as_u64()).unwrap_or(1);

            if command.is_empty() {
                return serde_json::json!({"error": "command is required"});
            }

            let spec = TaskSpec {
                command: command.to_string(),
                docker_image,
                timeout_secs: timeout,
                cpu_cores: cpu,
                memory_mb: mem,
            };

            let mut task = scheduler.submit_task(spec, &wallet.address());
            task.price = price;

            let msg = SchedulerMessage::TaskSubmitted(task.clone());
            if let Ok(data) = serde_json::to_vec(&msg) {
                network.broadcast("ynet-tasks", data).await;
            }

            serde_json::json!({"ok": true, "task_id": task.id})
        }
        "claim_task" => {
            let task_id = req.params.get("task_id").and_then(|v| v.as_str()).unwrap_or("");
            if task_id.is_empty() {
                return serde_json::json!({"error": "task_id is required"});
            }

            match scheduler.claim_task(task_id) {
                Some(task) => {
                    let msg = SchedulerMessage::TaskClaimed {
                        task_id: task.id.clone(),
                        worker_id: wallet.address(),
                    };
                    if let Ok(data) = serde_json::to_vec(&msg) {
                        network.broadcast("ynet-tasks", data).await;
                    }

                    execute_and_report(
                        task_id, scheduler, blockchain, wallet, network, &wallet.address(),
                    ).await;

                    serde_json::json!({"ok": true})
                }
                None => serde_json::json!({"error": "task not available"}),
            }
        }
        "execute_local" => {
            let command = req.params.get("command").and_then(|v| v.as_str()).unwrap_or("");
            if command.is_empty() {
                return serde_json::json!({"error": "command is required"});
            }

            let spec = TaskSpec {
                command: command.to_string(),
                docker_image: None,
                timeout_secs: 300,
                cpu_cores: 1,
                memory_mb: 256,
            };

            let task = scheduler.submit_task(spec, &wallet.address());
            let task_id = task.id.clone();
            scheduler.claim_task(&task_id);

            if let Some(result) = scheduler.execute_task(&task_id).await {
                serde_json::json!({
                    "ok": true,
                    "task_id": task_id,
                    "success": result.success,
                    "exit_code": result.exit_code,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "duration": result.duration_secs,
                })
            } else {
                serde_json::json!({"error": "execution failed"})
            }
        }
        "list_tasks" => {
            let tasks: Vec<_> = scheduler.list_tasks().iter()
                .map(|t| serde_json::json!({
                    "id": t.id,
                    "command": t.spec.command,
                    "status": t.status,
                    "submitter": t.submitter,
                    "worker": t.worker,
                    "price": t.price,
                    "created_at": t.created_at,
                    "result": t.result.as_ref().map(|r| serde_json::json!({
                        "success": r.success,
                        "exit_code": r.exit_code,
                        "duration": r.duration_secs,
                        "stdout_preview": &r.stdout[..r.stdout.len().min(500)],
                    })),
                }))
                .collect();
            serde_json::json!({"tasks": tasks})
        }

        // ---- Economics IPC ----
        "get_economics" => {
            serde_json::to_value(economics.summary()).unwrap()
        }
        "get_pricing" => {
            let model = req.params.get("model").and_then(|v| v.as_str()).unwrap_or("default");
            let pricing = ModelPricing::find(model);
            serde_json::to_value(pricing).unwrap()
        }
        "list_pricing" => {
            let all = ModelPricing::default_pricing();
            serde_json::to_value(all).unwrap()
        }
        "estimate_cost" => {
            let model = req.params.get("model").and_then(|v| v.as_str()).unwrap_or("default");
            let input_tokens = req.params.get("input_tokens").and_then(|v| v.as_u64()).unwrap_or(0);
            let output_tokens = req.params.get("output_tokens").and_then(|v| v.as_u64()).unwrap_or(0);
            let currency = req.params.get("currency").and_then(|v| v.as_str()).unwrap_or("YNET");
            let pay_currency = match currency {
                "ETH" => PaymentCurrency::ETH,
                "BTC" => PaymentCurrency::BTC,
                "USDT" => PaymentCurrency::USDT,
                "USDC" => PaymentCurrency::USDC,
                _ => PaymentCurrency::YNET,
            };
            let balance = blockchain.get_balance(&wallet.address());
            let fee = FeeCalculator::calculate(model, input_tokens, output_tokens, balance, &pay_currency);
            serde_json::to_value(fee).unwrap()
        }
        "get_reputation" => {
            let my_addr = wallet.address();
            let address = req.params.get("address").and_then(|v| v.as_str())
                .unwrap_or(&my_addr);
            match economics.get_reputation(address) {
                Some(rep) => serde_json::to_value(rep).unwrap(),
                None => serde_json::json!({"error": "no reputation data"}),
            }
        }
        "get_block_info" => {
            let height = blockchain.height();
            serde_json::json!({
                "block_height": height,
                "block_time_target_secs": TARGET_BLOCK_TIME_SECS,
                "bookkeeping_fee": ynet_chain::BLOCK_BOOKKEEPING_FEE,
                "pending_tx_count": blockchain.get_pending_transactions().len(),
            })
        }

        // ---- Inference IPC ----
        "get_inference" => {
            let models = inference_service.loaded_models();
            let cap = inference_service.capability();
            let network_models = inference_registry.available_models();
            serde_json::json!({
                "local_models": models,
                "gpus": cap.gpus,
                "vram_total_mb": cap.vram_total_mb,
                "active_requests": cap.queue_depth,
                "max_concurrent": cap.max_concurrent,
                "network_models": network_models.iter()
                    .map(|(id, count)| serde_json::json!({"model": id, "nodes": count}))
                    .collect::<Vec<_>>(),
                "network_inference_nodes": inference_registry.active_node_count(),
            })
        }
        "load_model" => {
            let model_id = req.params.get("model_id").and_then(|v| v.as_str()).unwrap_or("");
            let model_path = req.params.get("path").and_then(|v| v.as_str()).unwrap_or("");
            let backend_str = req.params.get("backend").and_then(|v| v.as_str()).unwrap_or("llamacpp");
            let port = req.params.get("port").and_then(|v| v.as_u64()).unwrap_or(8080) as u16;

            if model_id.is_empty() || model_path.is_empty() {
                return serde_json::json!({"error": "model_id and path are required"});
            }

            let backend = match backend_str {
                "llamacpp" | "llama" => InferenceBackend::LlamaCpp,
                "vllm" => InferenceBackend::Vllm,
                "ollama" => InferenceBackend::Ollama,
                _ => InferenceBackend::Custom,
            };

            match inference_service.load_model(model_id, model_path, backend, port).await {
                Ok(()) => serde_json::json!({"ok": true, "model_id": model_id, "port": port}),
                Err(e) => serde_json::json!({"error": e}),
            }
        }
        "unload_model" => {
            let model_id = req.params.get("model_id").and_then(|v| v.as_str()).unwrap_or("");
            if model_id.is_empty() {
                return serde_json::json!({"error": "model_id is required"});
            }
            match inference_service.unload_model(model_id).await {
                Ok(()) => serde_json::json!({"ok": true}),
                Err(e) => serde_json::json!({"error": e}),
            }
        }

        "broadcast" => {
            let msg = req.params.get("message").and_then(|v| v.as_str()).unwrap_or("");
            let topic = req.params.get("topic").and_then(|v| v.as_str()).unwrap_or("ynet-global");
            network.broadcast(topic, msg.as_bytes().to_vec()).await;
            serde_json::json!({"ok": true})
        }
        "dial" => {
            let addr = req.params.get("address").and_then(|v| v.as_str()).unwrap_or("");
            network.dial(addr).await;
            serde_json::json!({"ok": true})
        }
        _ => serde_json::json!({"error": "unknown method"}),
    }
}
