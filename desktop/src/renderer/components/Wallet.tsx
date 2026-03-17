import React, { useCallback, useEffect, useState } from 'react';

interface TxInfo {
  hash: string;
  from: string;
  to: string;
  amount: number;
  tx_type: string;
  timestamp: number;
}

interface PricingInfo {
  model_id: string;
  input_price_per_1k: number;
  output_price_per_1k: number;
  tier: string;
}

const NYNET = 1_000_000_000; // 1 YNET in nYNET

const formatYnet = (nynet: number) => {
  if (nynet >= NYNET) return (nynet / NYNET).toFixed(2);
  if (nynet >= 1_000_000) return (nynet / 1_000_000).toFixed(2) + 'm';
  if (nynet >= 1_000) return (nynet / 1_000).toFixed(2) + 'k';
  return nynet.toString();
};

const txTypeLabel: Record<string, string> = {
  BlockBookkeeping: '记账',
  Transfer: '转账',
  ApiPayment: 'API付费',
  ComputeProof: '算力证明',
  FeeBurn: '销毁',
};

const Wallet: React.FC = () => {
  const [wallet, setWallet] = useState<any>(null);
  const [chain, setChain] = useState<any>(null);
  const [economics, setEconomics] = useState<any>(null);
  const [pricing, setPricing] = useState<PricingInfo[]>([]);
  const [toAddr, setToAddr] = useState('');
  const [amount, setAmount] = useState('');
  const [transferMsg, setTransferMsg] = useState('');

  const refresh = useCallback(() => {
    window.ynet?.getWallet().then(setWallet).catch(() => {});
    window.ynet?.getChain().then(setChain).catch(() => {});
    window.ynet?.getEconomics().then(setEconomics).catch(() => {});
    window.ynet?.listPricing().then((data) => { if (Array.isArray(data)) setPricing(data); }).catch(() => {});
  }, []);

  useEffect(() => {
    refresh();
    const interval = setInterval(refresh, 5000);
    const unsub = window.ynet?.onEvent?.((ev) => {
      if (ev.event === 'new_block') refresh();
    });
    return () => { clearInterval(interval); unsub?.(); };
  }, [refresh]);

  const handleTransfer = async () => {
    if (!toAddr.trim() || !amount.trim()) return;
    setTransferMsg('');
    const result = await window.ynet?.transfer(toAddr.trim(), parseInt(amount));
    if (result?.ok) {
      setTransferMsg(`交易已提交: ${result.tx_hash.slice(0, 16)}...`);
      setToAddr('');
      setAmount('');
      refresh();
    } else {
      setTransferMsg(`失败: ${result?.error || '未知错误'}`);
    }
  };

  const recentTxs: TxInfo[] = chain?.recent_transactions ?? [];

  return (
    <div className="panel">
      <h2>钱包</h2>

      <div className="card">
        <div className="label">地址</div>
        <div className="value mono">{wallet?.address ?? '---'}</div>
      </div>

      <div className="card">
        <div className="label">余额</div>
        <div className="value" style={{ fontSize: 28, fontWeight: 700, color: '#00d4aa' }}>
          {wallet?.balance ?? 0} <span style={{ fontSize: 14, color: '#666' }}>YNET</span>
        </div>
        {wallet?.balance >= 100 && (
          <div style={{ fontSize: 12, color: '#ffd700', marginTop: 4 }}>
            持有折扣: {wallet.balance >= 100000 ? '25%' : wallet.balance >= 10000 ? '15%' : wallet.balance >= 1000 ? '10%' : '5%'}
          </div>
        )}
      </div>

      {/* Economics overview */}
      <div className="card" style={{ display: 'flex', gap: 16, flexWrap: 'wrap' }}>
        <div>
          <div className="label">流通供应</div>
          <div className="value" style={{ fontSize: 14 }}>
            {economics?.circulating_supply?.toLocaleString() ?? '1,000,000,000'} YNET
          </div>
        </div>
        <div>
          <div className="label">已销毁</div>
          <div className="value" style={{ fontSize: 14, color: '#ff6b6b' }}>
            {economics?.total_burned?.toLocaleString() ?? 0}
          </div>
        </div>
        <div>
          <div className="label">API 收入</div>
          <div className="value" style={{ fontSize: 14 }}>
            {economics?.total_api_revenue?.toLocaleString() ?? 0}
          </div>
        </div>
        <div>
          <div className="label">活跃节点</div>
          <div className="value" style={{ fontSize: 14 }}>
            {economics?.active_nodes ?? 0}
          </div>
        </div>
      </div>

      {/* API Pricing */}
      <div className="card">
        <div className="label">API 定价 (每 1000 tokens, nYNET)</div>
        {pricing.length > 0 ? (
          <div style={{ marginTop: 4 }}>
            {pricing.filter(p => p.model_id !== 'default').map((p, i) => (
              <div key={i} style={{ display: 'flex', justifyContent: 'space-between', padding: '4px 0', borderBottom: '1px solid #1a1a2e' }}>
                <span className="mono" style={{ fontSize: 12 }}>{p.model_id}</span>
                <span style={{ fontSize: 12, color: '#888' }}>
                  输入 {formatYnet(p.input_price_per_1k)} · 输出 {formatYnet(p.output_price_per_1k)}
                </span>
              </div>
            ))}
          </div>
        ) : (
          <div className="value muted">加载中...</div>
        )}
      </div>

      {/* Transfer */}
      <div className="card">
        <div className="label">转账</div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8, marginTop: 4 }}>
          <input className="input" placeholder="接收地址" value={toAddr} onChange={(e) => setToAddr(e.target.value)} />
          <div style={{ display: 'flex', gap: 8 }}>
            <input className="input" placeholder="金额" type="number" value={amount} onChange={(e) => setAmount(e.target.value)} />
            <button className="btn btn-sm" onClick={handleTransfer}>发送</button>
          </div>
          {transferMsg && (
            <div style={{ fontSize: 12, color: transferMsg.startsWith('失败') ? '#ff6b6b' : '#00d4aa' }}>
              {transferMsg}
            </div>
          )}
        </div>
      </div>

      {/* Recent transactions */}
      <div className="card">
        <div className="label">最近交易 · 区块高度 #{chain?.status?.height ?? 0} · ~3秒/块</div>
        {recentTxs.length > 0 ? (
          <div className="event-log" style={{ maxHeight: 300 }}>
            {recentTxs.map((tx, i) => (
              <div key={i} className="event-line" style={{ padding: '6px 0' }}>
                <span style={{ color: tx.tx_type === 'BlockBookkeeping' ? '#ffd700' : tx.tx_type === 'ApiPayment' ? '#00aaff' : '#e0e0e0' }}>
                  {txTypeLabel[tx.tx_type] ?? tx.tx_type}
                </span>
                {' '}
                <span className="mono" style={{ fontSize: 11 }}>
                  {tx.from === 'network' ? 'network' : tx.from.slice(0, 12) + '...'}
                </span>
                {' → '}
                <span className="mono" style={{ fontSize: 11 }}>{tx.to.slice(0, 12) + '...'}</span>
                {' '}
                <span style={{ color: '#00d4aa' }}>{tx.amount} YNET</span>
              </div>
            ))}
          </div>
        ) : (
          <div className="value muted">暂无交易</div>
        )}
      </div>
    </div>
  );
};

export default Wallet;
