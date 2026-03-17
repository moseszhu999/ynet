# yNet - Decentralized Compute Network

A decentralized computing platform with OpenAI-compatible inference API, enabling distributed AI model hosting and inference across a P2P network.

## Features

- **P2P Network**: Libp2p-based peer-to-peer networking for decentralized communication
- **Blockchain**: Built-in blockchain for transaction processing and consensus
- **Task Scheduler**: Distributed task scheduling with Docker isolation support
- **Inference Gateway**: OpenAI-compatible API for LLM inference
- **Multi-Backend Support**: vLLM, llama.cpp, Ollama, and custom backends
- **Economics Model**: Built-in token economics for compute rewards

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     yNet Desktop App                        │
│                    (Electron + React)                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      yNet Core (Rust)                       │
├──────────┬──────────┬──────────┬──────────┬────────────────┤
│   P2P    │  Chain   │Scheduler │Inference │     Node       │
│  (libp2p)│ (Block)  │ (Tasks)  │ (LLM)    │   (Main)       │
└──────────┴──────────┴──────────┴──────────┴────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Inference Backends (GPU Accelerated)           │
├─────────────┬─────────────┬─────────────┬──────────────────┤
│    vLLM     │  llama.cpp  │   Ollama    │    Custom API    │
└─────────────┴─────────────┴─────────────┴──────────────────┘
```

## Quick Start

### Prerequisites

- Rust 1.70+
- Node.js 18+ (for desktop app)
- GPU with CUDA/Metal support (for inference)

### Build

```bash
cd core
cargo build --release
```

### Run Node

```bash
# Start with local inference backend
./target/release/ynet-node \
  --port 0 \
  --api-port 3030 \
  --load-model "qwen:custom::11434:qwen2.5:0.5b" \
  --data-dir ./data

# Or connect to remote Ollama via SSH tunnel
ssh -N -L 11434:localhost:11434 user@remote-host &
./target/release/ynet-node \
  --port 0 \
  --api-port 3030 \
  --load-model "model-id:custom::11434:model-name"
```

### API Usage

yNet provides an OpenAI-compatible API:

```bash
# List models
curl http://localhost:3030/v1/models

# Chat completion (non-streaming)
curl http://localhost:3030/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'

# Chat completion (streaming)
curl http://localhost:3030/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

## Model Configuration

Format: `model_id:backend:path:port[:backend_model_name]`

| Backend | Description | Example |
|---------|-------------|---------|
| `llamacpp` | llama.cpp server | `kimi:llamacpp:/models/kimi.gguf:8080` |
| `vllm` | vLLM server | `qwen:vllm:/models/qwen:8000` |
| `ollama` | Ollama (local) | `qwen:ollama:qwen2.5:0.5b:11434` |
| `custom` | Any OpenAI-compatible API | `qwen:custom::11434:qwen2.5:0.5b` |

## Project Structure

```
ynet/
├── core/                   # Rust backend
│   ├── p2p/               # P2P networking (libp2p)
│   ├── chain/             # Blockchain & wallet
│   ├── scheduler/         # Task scheduling
│   ├── inference/         # LLM inference gateway
│   └── node/              # Main node entry point
├── desktop/               # Electron desktop app
│   ├── src/main/          # Main process
│   ├── src/renderer/      # React UI
│   └── src/bridge/        # IPC bridge
└── scripts/               # Deployment scripts
```

## License

MIT
