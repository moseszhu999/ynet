import { ChildProcess, spawn } from 'child_process';
import path from 'path';
import readline from 'readline';
import { BrowserWindow } from 'electron';

/**
 * Bridge between Electron (TS) and yNet Core (Rust).
 * JSON-line protocol over stdin/stdout.
 * Supports request-response and push events from core.
 */
export class CoreBridge {
  private process: ChildProcess | null = null;
  private requestId = 0;
  private pending = new Map<number, { resolve: (v: any) => void; reject: (e: any) => void }>();
  private rl: readline.Interface | null = null;
  private listenPort: number;
  private seedNodes: string[];

  constructor(listenPort = 0, seedNodes: string[] = []) {
    this.listenPort = listenPort;
    this.seedNodes = seedNodes;
  }

  async start(): Promise<void> {
    const coreBinaryName = process.platform === 'win32' ? 'ynet-node.exe' : 'ynet-node';
    const corePath = path.resolve(__dirname, '../../core/target/debug', coreBinaryName);

    const args = ['--port', String(this.listenPort)];
    for (const seed of this.seedNodes) {
      args.push('--seed', seed);
    }

    return new Promise((resolve, reject) => {
      let resolved = false;
      const done = () => {
        if (!resolved) { resolved = true; resolve(); }
      };

      this.process = spawn(corePath, args, {
        stdio: ['pipe', 'pipe', 'pipe'],
      });

      this.process.on('error', (err) => {
        if (!resolved) { resolved = true; reject(err); }
      });

      this.process.on('exit', (code) => {
        console.log(`yNet core exited with code ${code}`);
        this.process = null;
      });

      if (this.process.stdout) {
        this.rl = readline.createInterface({ input: this.process.stdout });
        this.rl.on('line', (line) => {
          try {
            const msg = JSON.parse(line);

            // Push event from core (no id field)
            if (msg.event) {
              // Resolve on ready signal from core
              if (msg.event === 'ready') done();
              this.forwardEvent(msg.event, msg.data);
              return;
            }

            // Request-response
            const pending = this.pending.get(msg.id);
            if (pending) {
              pending.resolve(msg.result);
              this.pending.delete(msg.id);
            }
          } catch {
            // ignore non-JSON output
          }
        });
      }

      if (this.process.stderr) {
        this.process.stderr.on('data', (data) => {
          console.log(`[core] ${data.toString().trim()}`);
        });
      }

      // Fallback timeout if ready signal never comes
      setTimeout(done, 5000);
    });
  }

  private forwardEvent(eventType: string, data: any) {
    for (const win of BrowserWindow.getAllWindows()) {
      win.webContents.send('core:event', { event: eventType, data });
    }
  }

  async request(method: string, params?: any): Promise<any> {
    if (!this.process?.stdin) {
      throw new Error('Core process not running');
    }

    const id = ++this.requestId;

    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      const msg = JSON.stringify({ id, method, params: params || {} }) + '\n';
      this.process!.stdin!.write(msg);

      setTimeout(() => {
        if (this.pending.has(id)) {
          this.pending.delete(id);
          reject(new Error(`Request ${method} timed out`));
        }
      }, 10000);
    });
  }

  stop() {
    if (this.process) {
      this.process.kill();
      this.process = null;
    }
    this.rl?.close();
  }
}
