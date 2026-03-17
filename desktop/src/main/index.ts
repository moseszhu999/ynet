import { app, BrowserWindow, ipcMain } from 'electron';
import path from 'path';
import { CoreBridge } from '../bridge/core-bridge';

let mainWindow: BrowserWindow | null = null;
let coreBridge: CoreBridge | null = null;

const isDev = !app.isPackaged;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    title: 'yNet',
    webPreferences: {
      preload: path.join(__dirname, '../bridge/preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  if (isDev) {
    mainWindow.loadURL('http://localhost:5173');
    mainWindow.webContents.openDevTools();
  } else {
    mainWindow.loadFile(path.join(__dirname, '../renderer/index.html'));
  }

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

async function startCore() {
  // Port 0 = OS picks a free port
  coreBridge = new CoreBridge(0);
  try {
    await coreBridge.start();
    console.log('yNet core started');
  } catch (e) {
    console.warn('yNet core not available (build with: cd core && cargo build):', e);
  }
}

// IPC handlers
ipcMain.handle('core:request', async (_event, method: string) => {
  if (!coreBridge) return { error: 'Core not running' };
  return coreBridge.request(method);
});

ipcMain.handle('core:request-with-params', async (_event, method: string, params: any) => {
  if (!coreBridge) return { error: 'Core not running' };
  return coreBridge.request(method, params);
});

app.whenReady().then(async () => {
  await startCore();
  createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on('window-all-closed', () => {
  coreBridge?.stop();
  if (process.platform !== 'darwin') {
    app.quit();
  }
});
