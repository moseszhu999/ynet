import React, { useEffect, useState } from 'react';
import Wallet from './components/Wallet';
import Tasks from './components/Tasks';
import Network from './components/Network';
import Inference from './components/Inference';

type Tab = 'wallet' | 'tasks' | 'inference' | 'network';

declare global {
  interface Window {
    ynet?: {
      getStatus: () => Promise<any>;
      getNetwork: () => Promise<any>;
      getWallet: () => Promise<any>;
      getChain: () => Promise<any>;
      getScheduler: () => Promise<any>;
      transfer: (to: string, amount: number) => Promise<any>;
      produceBlock: () => Promise<any>;
      submitTask: (params: any) => Promise<any>;
      executeLocal: (command: string) => Promise<any>;
      listTasks: () => Promise<any>;
      claimTask: (task_id: string) => Promise<any>;
      getEconomics: () => Promise<any>;
      getPricing: (model?: string) => Promise<any>;
      listPricing: () => Promise<any>;
      estimateCost: (params: any) => Promise<any>;
      getReputation: (address?: string) => Promise<any>;
      getBlockInfo: () => Promise<any>;
      getInference: () => Promise<any>;
      loadModel: (params: { model_id: string; path: string; backend: string; port: number }) => Promise<any>;
      unloadModel: (model_id: string) => Promise<any>;
      broadcast: (message: string, topic?: string) => Promise<any>;
      dial: (address: string) => Promise<any>;
      onEvent: (callback: (event: { event: string; data: any }) => void) => (() => void) | void;
    };
  }
}

const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<Tab>('wallet');
  const [peerCount, setPeerCount] = useState(0);
  const [isOnline, setIsOnline] = useState(false);
  const [chainHeight, setChainHeight] = useState(0);

  useEffect(() => {
    const poll = setInterval(() => {
      window.ynet?.getStatus().then((s) => {
        if (s && !s.error) {
          setPeerCount(s.p2p?.connected_peers ?? 0);
          setIsOnline(s.p2p?.is_online ?? false);
          setChainHeight(s.chain?.height ?? 0);
        }
      }).catch(() => {});
    }, 3000);

    const unsub = window.ynet?.onEvent?.((ev) => {
      if (ev.event === 'peer_connected' || ev.event === 'peer_disconnected') {
        window.ynet?.getNetwork().then((status) => {
          if (status && !status.error) setPeerCount(status.connected_peers ?? 0);
        });
      }
    });

    return () => { clearInterval(poll); unsub?.(); };
  }, []);

  return (
    <div className="app">
      <header className="app-header">
        <h1 className="logo">yNet</h1>
        <span className="version">v0.1.0</span>
      </header>

      <nav className="tabs">
        {(['wallet', 'tasks', 'inference', 'network'] as Tab[]).map((tab) => (
          <button
            key={tab}
            className={`tab ${activeTab === tab ? 'active' : ''}`}
            onClick={() => setActiveTab(tab)}
          >
            {{ wallet: '钱包', tasks: '任务', inference: '推理', network: '网络' }[tab]}
          </button>
        ))}
      </nav>

      <main className="content">
        {activeTab === 'wallet' && <Wallet />}
        {activeTab === 'tasks' && <Tasks />}
        {activeTab === 'inference' && <Inference />}
        {activeTab === 'network' && <Network />}
      </main>

      <footer className="status-bar">
        <span className={`status-dot ${isOnline ? 'online' : 'offline'}`} />
        <span>
          {isOnline
            ? `在线 · ${peerCount} 节点 · 区块高度 #${chainHeight}`
            : '节点未连接'}
        </span>
      </footer>
    </div>
  );
};

export default App;
