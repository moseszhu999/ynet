import React, { useCallback, useEffect, useState } from 'react';

interface PeerInfo {
  peer_id: string;
  addresses: string[];
}

interface NetworkState {
  node_id: string;
  connected_peers: number;
  is_online: boolean;
  listening_addresses: string[];
  peers: PeerInfo[];
}

const Network: React.FC = () => {
  const [network, setNetwork] = useState<NetworkState | null>(null);
  const [events, setEvents] = useState<string[]>([]);
  const [dialAddr, setDialAddr] = useState('');
  const [broadcastMsg, setBroadcastMsg] = useState('');

  const refresh = useCallback(() => {
    window.ynet?.getNetwork().then((data) => {
      if (data && !data.error) setNetwork(data);
    }).catch(() => {});
  }, []);

  useEffect(() => {
    refresh();
    const interval = setInterval(refresh, 5000);

    const unsub = window.ynet?.onEvent?.((ev) => {
      const ts = new Date().toLocaleTimeString();
      if (ev.event === 'peer_connected') {
        setEvents((prev) => [`${ts} + 节点连接: ${ev.data.peer_id.slice(0, 16)}...`, ...prev].slice(0, 50));
        refresh();
      } else if (ev.event === 'peer_disconnected') {
        setEvents((prev) => [`${ts} - 节点断开: ${ev.data.peer_id.slice(0, 16)}...`, ...prev].slice(0, 50));
        refresh();
      } else if (ev.event === 'message') {
        setEvents((prev) => [`${ts} 消息 [${ev.data.topic}]: ${ev.data.data}`, ...prev].slice(0, 50));
      }
    });

    return () => { clearInterval(interval); unsub?.(); };
  }, [refresh]);

  const handleDial = () => {
    if (dialAddr.trim()) {
      window.ynet?.dial(dialAddr.trim());
      setDialAddr('');
    }
  };

  const handleBroadcast = () => {
    if (broadcastMsg.trim()) {
      window.ynet?.broadcast(broadcastMsg.trim());
      setBroadcastMsg('');
    }
  };

  return (
    <div className="panel">
      <h2>网络</h2>

      <div className="card">
        <div className="label">节点 ID</div>
        <div className="value mono">{network?.node_id ?? '---'}</div>
      </div>

      <div className="card">
        <div className="label">监听地址</div>
        {network?.listening_addresses?.length ? (
          network.listening_addresses.map((addr, i) => (
            <div key={i} className="value mono" style={{ fontSize: 12, marginTop: 4 }}>{addr}</div>
          ))
        ) : (
          <div className="value muted">---</div>
        )}
      </div>

      <div className="card">
        <div className="label">已连接节点 ({network?.connected_peers ?? 0})</div>
        {network?.peers?.length ? (
          network.peers.map((p, i) => (
            <div key={i} style={{ marginTop: 8, padding: '8px 0', borderTop: i > 0 ? '1px solid #1a1a2e' : 'none' }}>
              <div className="mono" style={{ fontSize: 12 }}>{p.peer_id}</div>
              {p.addresses.map((a, j) => (
                <div key={j} className="muted" style={{ fontSize: 11, marginLeft: 8 }}>{a}</div>
              ))}
            </div>
          ))
        ) : (
          <div className="value muted">暂无连接</div>
        )}
      </div>

      <div className="card">
        <div className="label">连接节点</div>
        <div style={{ display: 'flex', gap: 8, marginTop: 4 }}>
          <input
            className="input"
            placeholder="/ip4/192.168.1.x/tcp/port"
            value={dialAddr}
            onChange={(e) => setDialAddr(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleDial()}
          />
          <button className="btn btn-sm" onClick={handleDial}>连接</button>
        </div>
      </div>

      <div className="card">
        <div className="label">广播消息</div>
        <div style={{ display: 'flex', gap: 8, marginTop: 4 }}>
          <input
            className="input"
            placeholder="输入消息..."
            value={broadcastMsg}
            onChange={(e) => setBroadcastMsg(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleBroadcast()}
          />
          <button className="btn btn-sm" onClick={handleBroadcast}>发送</button>
        </div>
      </div>

      <div className="card">
        <div className="label">事件日志</div>
        <div className="event-log">
          {events.length ? (
            events.map((e, i) => (
              <div key={i} className="event-line">{e}</div>
            ))
          ) : (
            <div className="value muted">暂无事件</div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Network;
