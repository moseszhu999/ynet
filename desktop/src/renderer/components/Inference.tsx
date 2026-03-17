import React, { useCallback, useEffect, useState } from 'react';

interface GpuInfo {
  name: string;
  vram_mb: number;
}

interface NetworkModel {
  model: string;
  nodes: number;
}

interface InferenceState {
  local_models: string[];
  gpus: GpuInfo[];
  vram_total_mb: number;
  active_requests: number;
  max_concurrent: number;
  network_models: NetworkModel[];
  network_inference_nodes: number;
}

const Inference: React.FC = () => {
  const [info, setInfo] = useState<InferenceState | null>(null);
  const [modelId, setModelId] = useState('');
  const [modelPath, setModelPath] = useState('');
  const [backend, setBackend] = useState('llamacpp');
  const [port, setPort] = useState('8080');
  const [loadMsg, setLoadMsg] = useState('');
  const [loading, setLoading] = useState(false);

  const refresh = useCallback(() => {
    window.ynet?.getInference().then((data) => {
      if (data && !data.error) setInfo(data);
    }).catch(() => {});
  }, []);

  useEffect(() => {
    refresh();
    const interval = setInterval(refresh, 5000);
    return () => clearInterval(interval);
  }, [refresh]);

  const handleLoad = async () => {
    if (!modelId.trim() || !modelPath.trim()) return;
    setLoading(true);
    setLoadMsg('');
    const result = await window.ynet?.loadModel({
      model_id: modelId.trim(),
      path: modelPath.trim(),
      backend,
      port: parseInt(port) || 8080,
    });
    setLoading(false);
    if (result?.ok) {
      setLoadMsg(`模型 ${modelId} 已加载`);
      setModelId('');
      setModelPath('');
      refresh();
    } else {
      setLoadMsg(`失败: ${result?.error || '未知错误'}`);
    }
  };

  const handleUnload = async (id: string) => {
    const result = await window.ynet?.unloadModel(id);
    if (result?.ok) refresh();
  };

  return (
    <div className="panel">
      <h2>推理</h2>

      {/* GPU info */}
      <div className="card">
        <div className="label">本机 GPU</div>
        {info?.gpus?.length ? (
          info.gpus.map((gpu, i) => (
            <div key={i} style={{ marginTop: 4 }}>
              <span style={{ color: '#00d4aa' }}>{gpu.name}</span>
              <span className="muted" style={{ marginLeft: 8, fontSize: 12 }}>
                {(gpu.vram_mb / 1024).toFixed(1)} GB
              </span>
            </div>
          ))
        ) : (
          <div className="value muted">未检测到 GPU</div>
        )}
      </div>

      {/* Local models */}
      <div className="card">
        <div className="label">
          本机模型 ({info?.local_models?.length ?? 0})
          <span className="muted" style={{ marginLeft: 12, fontSize: 11 }}>
            并发 {info?.active_requests ?? 0}/{info?.max_concurrent ?? 0}
          </span>
        </div>
        {info?.local_models?.length ? (
          info.local_models.map((m, i) => (
            <div key={i} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '6px 0', borderBottom: '1px solid #1a1a2e' }}>
              <span className="mono" style={{ fontSize: 13, color: '#00d4aa' }}>{m}</span>
              <button className="btn btn-sm" style={{ background: '#2a1a1a', color: '#ff6b6b', fontSize: 11 }}
                onClick={() => handleUnload(m)}>
                卸载
              </button>
            </div>
          ))
        ) : (
          <div className="value muted" style={{ marginTop: 4 }}>暂无模型</div>
        )}
      </div>

      {/* Network models */}
      <div className="card">
        <div className="label">
          全网模型
          <span className="muted" style={{ marginLeft: 12, fontSize: 11 }}>
            {info?.network_inference_nodes ?? 0} 个推理节点
          </span>
        </div>
        {info?.network_models?.length ? (
          info.network_models.map((m, i) => (
            <div key={i} style={{ display: 'flex', justifyContent: 'space-between', padding: '6px 0', borderBottom: '1px solid #1a1a2e' }}>
              <span className="mono" style={{ fontSize: 13 }}>{m.model}</span>
              <span className="muted" style={{ fontSize: 12 }}>
                {m.nodes} 节点可用
              </span>
            </div>
          ))
        ) : (
          <div className="value muted" style={{ marginTop: 4 }}>暂无推理节点</div>
        )}
      </div>

      {/* Load model */}
      <div className="card">
        <div className="label">加载模型</div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8, marginTop: 4 }}>
          <input
            className="input"
            placeholder="模型 ID (例: kimi-k2.5-q4)"
            value={modelId}
            onChange={(e) => setModelId(e.target.value)}
          />
          <input
            className="input"
            placeholder="模型路径 (例: /models/kimi-k2.5.Q4_K_M.gguf)"
            value={modelPath}
            onChange={(e) => setModelPath(e.target.value)}
          />
          <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            <select
              className="input"
              style={{ width: 140 }}
              value={backend}
              onChange={(e) => setBackend(e.target.value)}
            >
              <option value="llamacpp">llama.cpp</option>
              <option value="vllm">vLLM</option>
              <option value="ollama">Ollama</option>
              <option value="custom">自定义</option>
            </select>
            <input
              className="input"
              style={{ width: 80 }}
              placeholder="端口"
              type="number"
              value={port}
              onChange={(e) => setPort(e.target.value)}
            />
            <button className="btn btn-sm" onClick={handleLoad} disabled={loading}>
              {loading ? '加载中...' : '加载'}
            </button>
          </div>
          {loadMsg && (
            <div style={{ fontSize: 12, color: loadMsg.startsWith('失败') ? '#ff6b6b' : '#00d4aa' }}>
              {loadMsg}
            </div>
          )}
        </div>
      </div>

      {/* Usage hint */}
      <div className="card">
        <div className="label">API 使用</div>
        <div style={{ fontSize: 12, color: '#888', lineHeight: 1.8 }}>
          启动节点时添加 <code style={{ color: '#00aaff' }}>--api-port 3000</code> 开启 OpenAI 兼容 API
          <br />
          <code style={{ color: '#aaa' }}>POST http://localhost:3000/v1/chat/completions</code>
          <br />
          <code style={{ color: '#aaa' }}>GET  http://localhost:3000/v1/models</code>
          <br />
          支持流式输出 (stream: true)，兼容所有 OpenAI SDK
        </div>
      </div>
    </div>
  );
};

export default Inference;
