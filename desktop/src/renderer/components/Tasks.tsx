import React, { useCallback, useEffect, useState } from 'react';

interface TaskInfo {
  id: string;
  command: string;
  status: string;
  submitter: string;
  worker: string | null;
  price: number;
  created_at: number;
  result?: {
    success: boolean;
    exit_code: number;
    duration: number;
    stdout_preview: string;
  };
}

const statusLabel: Record<string, string> = {
  Pending: '等待中',
  Running: '运行中',
  Completed: '已完成',
  Failed: '失败',
};

const statusColor: Record<string, string> = {
  Pending: '#ffd700',
  Running: '#00aaff',
  Completed: '#00d4aa',
  Failed: '#ff6b6b',
};

const Tasks: React.FC = () => {
  const [scheduler, setScheduler] = useState<any>(null);
  const [tasks, setTasks] = useState<TaskInfo[]>([]);
  const [command, setCommand] = useState('');
  const [price, setPrice] = useState('1');
  const [submitting, setSubmitting] = useState(false);
  const [resultMsg, setResultMsg] = useState('');
  const [expandedTask, setExpandedTask] = useState<string | null>(null);

  const refresh = useCallback(() => {
    window.ynet?.getScheduler().then(setScheduler).catch(() => {});
    window.ynet?.listTasks().then((data) => {
      if (data?.tasks) setTasks(data.tasks);
    }).catch(() => {});
  }, []);

  useEffect(() => {
    refresh();
    const interval = setInterval(refresh, 5000);
    const unsub = window.ynet?.onEvent?.((ev) => {
      if (['task_received', 'task_claimed', 'task_completed', 'task_executed'].includes(ev.event)) {
        refresh();
      }
    });
    return () => { clearInterval(interval); unsub?.(); };
  }, [refresh]);

  const handleSubmit = async () => {
    if (!command.trim()) return;
    setSubmitting(true);
    setResultMsg('');
    const result = await window.ynet?.submitTask({
      command: command.trim(),
      price: parseInt(price) || 1,
    });
    setSubmitting(false);
    if (result?.ok) {
      setResultMsg(`任务已提交: ${result.task_id.slice(0, 8)}...`);
      setCommand('');
      refresh();
    } else {
      setResultMsg(`失败: ${result?.error || '未知错误'}`);
    }
  };

  const handleExecuteLocal = async () => {
    if (!command.trim()) return;
    setSubmitting(true);
    setResultMsg('');
    const result = await window.ynet?.executeLocal(command.trim());
    setSubmitting(false);
    if (result?.ok) {
      setResultMsg(`本地执行完成 (${result.duration.toFixed(2)}s): ${result.stdout.slice(0, 100)}`);
      setCommand('');
      refresh();
    } else {
      setResultMsg(`失败: ${result?.error || result?.stderr || '未知错误'}`);
    }
  };

  return (
    <div className="panel">
      <h2>任务</h2>

      <div className="card" style={{ display: 'flex', gap: 16 }}>
        <div>
          <div className="label">等待中</div>
          <div className="value" style={{ fontSize: 24, color: '#ffd700' }}>{scheduler?.pending_tasks ?? 0}</div>
        </div>
        <div>
          <div className="label">运行中</div>
          <div className="value" style={{ fontSize: 24, color: '#00aaff' }}>{scheduler?.running_tasks ?? 0}</div>
        </div>
        <div>
          <div className="label">已完成</div>
          <div className="value" style={{ fontSize: 24, color: '#00d4aa' }}>{scheduler?.completed_tasks ?? 0}</div>
        </div>
        <div>
          <div className="label">我完成的</div>
          <div className="value" style={{ fontSize: 24 }}>{scheduler?.my_completed ?? 0}</div>
        </div>
      </div>

      <div className="card">
        <div className="label">提交任务</div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8, marginTop: 4 }}>
          <input
            className="input"
            placeholder="命令 (例: echo hello, python3 -c 'print(2**100)')"
            value={command}
            onChange={(e) => setCommand(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSubmit()}
          />
          <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            <input
              className="input"
              style={{ width: 80 }}
              placeholder="价格"
              type="number"
              value={price}
              onChange={(e) => setPrice(e.target.value)}
            />
            <span style={{ fontSize: 12, color: '#666' }}>YNET</span>
            <button className="btn btn-sm" onClick={handleSubmit} disabled={submitting}>
              {submitting ? '提交中...' : '提交到网络'}
            </button>
            <button className="btn btn-sm" style={{ background: '#1a3a4a', color: '#00aaff' }}
              onClick={handleExecuteLocal} disabled={submitting}>
              本地执行
            </button>
          </div>
          {resultMsg && (
            <div style={{ fontSize: 12, color: resultMsg.startsWith('失败') ? '#ff6b6b' : '#00d4aa' }}>
              {resultMsg}
            </div>
          )}
        </div>
      </div>

      <div className="card">
        <div className="label">任务列表</div>
        {tasks.length > 0 ? (
          <div style={{ marginTop: 4 }}>
            {tasks.map((task) => (
              <div
                key={task.id}
                style={{
                  padding: '10px 0',
                  borderBottom: '1px solid #1a1a2e',
                  cursor: 'pointer',
                }}
                onClick={() => setExpandedTask(expandedTask === task.id ? null : task.id)}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <div>
                    <span className="mono" style={{ fontSize: 12 }}>{task.id.slice(0, 8)}...</span>
                    {' '}
                    <span style={{ color: '#aaa' }}>{task.command.slice(0, 40)}{task.command.length > 40 ? '...' : ''}</span>
                  </div>
                  <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                    <span style={{ fontSize: 12, color: '#666' }}>{task.price} YNET</span>
                    <span style={{
                      fontSize: 11, padding: '2px 8px', borderRadius: 4,
                      background: statusColor[task.status] + '22',
                      color: statusColor[task.status],
                    }}>
                      {statusLabel[task.status] ?? task.status}
                    </span>
                  </div>
                </div>

                {expandedTask === task.id && (
                  <div style={{ marginTop: 8, fontSize: 12, color: '#888' }}>
                    <div>提交者: <span className="mono">{task.submitter.slice(0, 20)}...</span></div>
                    {task.worker && <div>执行者: <span className="mono">{task.worker.slice(0, 20)}...</span></div>}
                    {task.result && (
                      <div style={{ marginTop: 4 }}>
                        <div>耗时: {task.result.duration.toFixed(2)}s · 退出码: {task.result.exit_code}</div>
                        {task.result.stdout_preview && (
                          <pre style={{
                            marginTop: 4, padding: 8, background: '#0a0a0f',
                            borderRadius: 4, fontSize: 11, color: '#aaa',
                            whiteSpace: 'pre-wrap', maxHeight: 150, overflow: 'auto',
                          }}>
                            {task.result.stdout_preview}
                          </pre>
                        )}
                      </div>
                    )}
                  </div>
                )}
              </div>
            ))}
          </div>
        ) : (
          <div className="value muted" style={{ marginTop: 4 }}>暂无任务</div>
        )}
      </div>
    </div>
  );
};

export default Tasks;
