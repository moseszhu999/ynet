#!/bin/bash
# ============================================================
# yNet 推理节点一键部署脚本
# 目标：在有 GPU 的台式机上部署 Qwen3-8B + yNet 节点
# 前提：Linux 系统，NVIDIA GPU (RTX 5060 8GB)
# ============================================================

set -e

echo "=============================="
echo "  yNet 推理节点部署"
echo "=============================="
echo ""

# ---- 颜色 ----
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

ok()   { echo -e "${GREEN}[OK]${NC} $1"; }
warn() { echo -e "${YELLOW}[!]${NC} $1"; }
fail() { echo -e "${RED}[FAIL]${NC} $1"; exit 1; }

# ---- 1. 检查 GPU ----
echo "1. 检查 GPU..."
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    ok "检测到 GPU: $GPU_NAME ($GPU_MEM MB)"
else
    fail "未检测到 nvidia-smi。请确认已安装 NVIDIA 驱动。"
fi

# ---- 2. 安装 Ollama ----
echo ""
echo "2. 安装 Ollama..."
if command -v ollama &> /dev/null; then
    OLLAMA_VER=$(ollama --version 2>/dev/null || echo "unknown")
    ok "Ollama 已安装: $OLLAMA_VER"
else
    echo "   正在安装 Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
    ok "Ollama 安装完成"
fi

# 确保 ollama 服务在运行
if ! pgrep -x ollama &> /dev/null; then
    echo "   启动 Ollama 服务..."
    nohup ollama serve > /tmp/ollama.log 2>&1 &
    sleep 3
    ok "Ollama 服务已启动"
else
    ok "Ollama 服务已在运行"
fi

# ---- 3. 下载 Qwen3-8B ----
echo ""
echo "3. 下载 Qwen3-8B 模型 (~5GB)..."

if ollama list 2>/dev/null | grep -q "qwen3:8b"; then
    ok "Qwen3-8B 已下载"
else
    echo "   首次下载，预计 5-10 分钟..."
    ollama pull qwen3:8b
    ok "Qwen3-8B 下载完成"
fi

# ---- 4. 验证模型可用 ----
echo ""
echo "4. 验证模型推理..."
RESPONSE=$(ollama run qwen3:8b "回复OK两个字母" 2>/dev/null | head -5)
if [ -n "$RESPONSE" ]; then
    ok "模型推理正常: $RESPONSE"
else
    warn "模型响应为空，可能需要更多时间加载"
fi

# ---- 5. 安装 Rust (如果需要编译 yNet) ----
echo ""
echo "5. 检查 Rust 工具链..."
if command -v cargo &> /dev/null; then
    RUST_VER=$(rustc --version)
    ok "Rust 已安装: $RUST_VER"
else
    echo "   安装 Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
    ok "Rust 安装完成"
fi

# ---- 6. 编译 yNet Core ----
echo ""
echo "6. 编译 yNet Core..."
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
YNET_ROOT="$(dirname "$SCRIPT_DIR")"
CORE_DIR="$YNET_ROOT/core"

if [ ! -f "$CORE_DIR/Cargo.toml" ]; then
    fail "找不到 $CORE_DIR/Cargo.toml，请确认脚本位于 ynet/scripts/ 目录下"
fi

cd "$CORE_DIR"
cargo build --release 2>&1 | tail -5
YNET_BIN="$CORE_DIR/target/release/ynet-node"

if [ -f "$YNET_BIN" ]; then
    ok "yNet 节点编译完成: $YNET_BIN"
else
    fail "编译失败，找不到 $YNET_BIN"
fi

# ---- 7. 创建数据目录 ----
echo ""
echo "7. 创建数据目录..."
DATA_DIR="$HOME/.ynet"
mkdir -p "$DATA_DIR"
ok "数据目录: $DATA_DIR"

# ---- 8. 创建 systemd 服务 (可选) ----
echo ""
echo "8. 创建启动脚本..."

cat > "$DATA_DIR/start.sh" << 'STARTEOF'
#!/bin/bash
# yNet 推理节点启动脚本

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# 确保 Ollama 在运行
if ! pgrep -x ollama &> /dev/null; then
    echo "启动 Ollama 服务..."
    nohup ollama serve > /tmp/ollama.log 2>&1 &
    sleep 3
fi

# 找到 yNet 二进制
YNET_BIN="$(dirname "$SCRIPT_DIR")/slurm/ynet/core/target/release/ynet-node"
if [ ! -f "$YNET_BIN" ]; then
    # 尝试从当前目录的相对路径找
    YNET_BIN="$(find "$HOME" -name ynet-node -path "*/release/*" 2>/dev/null | head -1)"
fi

if [ -z "$YNET_BIN" ] || [ ! -f "$YNET_BIN" ]; then
    echo "错误: 找不到 ynet-node 二进制文件"
    exit 1
fi

echo "=============================="
echo "  yNet 推理节点"
echo "  模型: qwen3-8b (Ollama)"
echo "  API:  http://0.0.0.0:3000"
echo "  P2P:  自动发现"
echo "=============================="

exec "$YNET_BIN" \
    --data-dir "$HOME/.ynet" \
    --api-port 3000 \
    --load-model "qwen3-8b:ollama:qwen3:8b:11434"
STARTEOF

chmod +x "$DATA_DIR/start.sh"
ok "启动脚本: $DATA_DIR/start.sh"

# ---- 9. 启动 ----
echo ""
echo "=============================="
echo -e "${GREEN}  部署完成！${NC}"
echo "=============================="
echo ""
echo "启动 yNet 推理节点:"
echo "  $DATA_DIR/start.sh"
echo ""
echo "API 端点 (OpenAI 兼容):"
echo "  POST http://$(hostname -I 2>/dev/null | awk '{print $1}' || echo 'localhost'):3000/v1/chat/completions"
echo "  GET  http://$(hostname -I 2>/dev/null | awk '{print $1}' || echo 'localhost'):3000/v1/models"
echo ""
echo "测试:"
echo '  curl http://localhost:3000/v1/chat/completions \'
echo '    -H "Content-Type: application/json" \'
echo "    -d '{\"model\":\"qwen3-8b\",\"messages\":[{\"role\":\"user\",\"content\":\"你好\"}]}'"
echo ""
echo "其他 yNet 节点连接本节点:"
echo "  ynet-node --seed /ip4/$(hostname -I 2>/dev/null | awk '{print $1}' || echo '192.168.x.x')/tcp/<port>"
echo ""

# 询问是否立即启动
read -p "是否现在启动节点? [Y/n] " -n 1 -r
echo
if [[ $REPLY =~ ^[Nn]$ ]]; then
    echo "稍后运行: $DATA_DIR/start.sh"
else
    exec "$DATA_DIR/start.sh"
fi
