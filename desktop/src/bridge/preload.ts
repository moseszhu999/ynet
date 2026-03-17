import { contextBridge, ipcRenderer } from 'electron';

contextBridge.exposeInMainWorld('ynet', {
  // Status
  getStatus: () => ipcRenderer.invoke('core:request', 'get_status'),
  getNetwork: () => ipcRenderer.invoke('core:request', 'get_network'),
  getWallet: () => ipcRenderer.invoke('core:request', 'get_wallet'),
  getChain: () => ipcRenderer.invoke('core:request', 'get_chain'),
  getScheduler: () => ipcRenderer.invoke('core:request', 'get_scheduler'),

  // Blockchain
  transfer: (to: string, amount: number) =>
    ipcRenderer.invoke('core:request-with-params', 'transfer', { to, amount }),
  produceBlock: () => ipcRenderer.invoke('core:request-with-params', 'produce_block', {}),

  // Scheduler
  submitTask: (params: { command: string; docker_image?: string; timeout?: number; price?: number }) =>
    ipcRenderer.invoke('core:request-with-params', 'submit_task', params),
  executeLocal: (command: string) =>
    ipcRenderer.invoke('core:request-with-params', 'execute_local', { command }),
  listTasks: () => ipcRenderer.invoke('core:request-with-params', 'list_tasks', {}),
  claimTask: (task_id: string) =>
    ipcRenderer.invoke('core:request-with-params', 'claim_task', { task_id }),

  // Economics
  getEconomics: () => ipcRenderer.invoke('core:request-with-params', 'get_economics', {}),
  getPricing: (model?: string) =>
    ipcRenderer.invoke('core:request-with-params', 'get_pricing', { model }),
  listPricing: () => ipcRenderer.invoke('core:request-with-params', 'list_pricing', {}),
  estimateCost: (params: { model: string; input_tokens: number; output_tokens: number; currency?: string }) =>
    ipcRenderer.invoke('core:request-with-params', 'estimate_cost', params),
  getReputation: (address?: string) =>
    ipcRenderer.invoke('core:request-with-params', 'get_reputation', address ? { address } : {}),
  getBlockInfo: () =>
    ipcRenderer.invoke('core:request-with-params', 'get_block_info', {}),

  // Inference
  getInference: () => ipcRenderer.invoke('core:request-with-params', 'get_inference', {}),
  loadModel: (params: { model_id: string; path: string; backend: string; port: number }) =>
    ipcRenderer.invoke('core:request-with-params', 'load_model', params),
  unloadModel: (model_id: string) =>
    ipcRenderer.invoke('core:request-with-params', 'unload_model', { model_id }),

  // P2P
  broadcast: (message: string, topic?: string) =>
    ipcRenderer.invoke('core:request-with-params', 'broadcast', { message, topic }),
  dial: (address: string) =>
    ipcRenderer.invoke('core:request-with-params', 'dial', { address }),

  // Push events (returns unsubscribe function)
  onEvent: (callback: (event: { event: string; data: any }) => void) => {
    const handler = (_e: any, data: any) => callback(data);
    ipcRenderer.on('core:event', handler);
    return () => ipcRenderer.removeListener('core:event', handler);
  },
});
