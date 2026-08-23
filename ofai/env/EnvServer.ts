/**
 * Vectorized env server: runs K AgentEnv instances and serves batched
 * reset/step over length-prefixed binary TCP frames.
 *
 * Default backend is sequential (oracle / fallback). Pass --workers N>1
 * or set OFAI_ENV_WORKERS to farm complete games across worker threads.
 * Each worker owns its games and advances each game single-threadedly.
 *
 * Protocol (see framing.ts for the wire format):
 *   <- {cmd:"init", envs:[EnvConfig,...]}        -> {type:"inited"} + obs tensors
 *   <- {cmd:"step", actions, nextConfigs?}       -> {type:"step", rewards, dones, infos} + obs tensors
 *   <- {cmd:"reset", index, seed?|config?}       -> {type:"reset"} + obs tensors (K=1)
 *   <- {cmd:"close"}                             -> server exits
 *
 * Run: npx tsx ofai/env/EnvServer.ts --port 8765 [--workers 4]
 */
import net from "node:net";
import {
  encodeInit,
  encodeReset,
  encodeStep,
  EnvBackend,
  parseServerArgs,
  resolveWorkerCount,
  SequentialEnvBackend,
} from "./EnvBackend";
import { encodeFrame, FrameDecoder, frameTensors } from "./framing";
import { ParallelEnvBackend } from "./ParallelBackend";
import { EnvConfig } from "./spec";

class Server {
  private backend: EnvBackend;
  private requestedWorkers: number;
  private initialized = false;

  constructor(requestedWorkers: number) {
    this.requestedWorkers = requestedWorkers;
    this.backend = new SequentialEnvBackend();
  }

  async handle(
    header: Record<string, unknown>,
    blobs: Record<string, { buf: Buffer }>,
  ): Promise<Buffer> {
    const cmd = header.cmd as string;
    switch (cmd) {
      case "init":
        return this.init(header.envs as EnvConfig[]);
      case "step":
        return encodeStep(await this.backend.step(header, blobs));
      case "reset": {
        const result = await this.backend.reset(header);
        return encodeReset(result.index, result.obs);
      }
      case "frame":
        return this.frame(header);
      case "close":
        await this.backend.close();
        setTimeout(() => process.exit(0), 100);
        return encodeFrame({ type: "closed" });
      default:
        return encodeFrame({ type: "error", error: `unknown cmd ${cmd}` });
    }
  }

  private async init(configs: EnvConfig[]): Promise<Buffer> {
    if (this.initialized) {
      await this.backend.close();
    }
    const n = resolveWorkerCount(this.requestedWorkers, configs.length);
    console.error(
      `[env-server] init: ${configs.length} envs requestedWorkers=${this.requestedWorkers} resolved=${n}`,
    );
    const t0 = performance.now();
    let backend: EnvBackend;
    if (n <= 1) {
      backend = new SequentialEnvBackend();
    } else {
      backend = new ParallelEnvBackend(n);
      try {
        const result = await backend.init(configs);
        this.backend = backend;
        this.initialized = true;
        this.logInit(result.k, result.backend, result.workers, t0, result.obs);
        return encodeInit(result);
      } catch (err: unknown) {
        const message = err instanceof Error ? err.message : String(err);
        console.error(
          `[env-server] parallel workers failed (${message}); using sequential oracle`,
        );
        await backend.close();
        backend = new SequentialEnvBackend();
      }
    }
    const result = await backend.init(configs);
    this.backend = backend;
    this.initialized = true;
    this.logInit(result.k, result.backend, result.workers, t0, result.obs);
    return encodeInit(result);
  }

  private logInit(
    k: number,
    backend: string,
    workers: number,
    t0: number,
    obs: { spatial: Float32Array; players: Float32Array },
  ): void {
    console.error(
      `[env-server] obs bytes/env: ${obs.spatial.byteLength / k + obs.players.byteLength / k} (spatial+players)`,
    );
    console.error(
      `[env-server] init done in ${(performance.now() - t0).toFixed(0)}ms backend=${backend} workers=${workers}`,
    );
  }

  private async frame(header: Record<string, unknown>): Promise<Buffer> {
    const f = await this.backend.frame(header);
    return encodeFrame(
      {
        type: "frame",
        width: f.width,
        height: f.height,
        players: f.players,
        units: f.units,
        tick: f.tick,
      },
      { cells: { dtype: "u8", data: f.cells } },
    );
  }
}

const isMain =
  process.argv[1] !== undefined &&
  (process.argv[1].endsWith("EnvServer.ts") ||
    process.argv[1].endsWith("EnvServer.js"));

if (isMain) {
  const { port, workers } = parseServerArgs(process.argv.slice(2));
  const server = new Server(workers);

  net
    .createServer((socket) => {
      console.error("[env-server] client connected");
      const decoder = new FrameDecoder();
      let chain: Promise<void> = Promise.resolve();
      socket.on("data", (chunk) => {
        const frames = decoder.push(chunk);
        for (const frame of frames) {
          const tensors = frameTensors(frame);
          chain = chain.then(async () => {
            try {
              const reply = await server.handle(frame.header, tensors);
              socket.write(reply);
            } catch (err: unknown) {
              const message = err instanceof Error ? err.message : String(err);
              console.error("[env-server] error:", message);
              socket.write(encodeFrame({ type: "error", error: message }));
            }
          });
        }
      });
      socket.on("error", (err) => {
        console.error("[env-server] socket error:", err);
      });
    })
    .listen(port, () => {
      console.error(
        `[env-server] listening on port ${port} workers=${workers}`,
      );
    });

  console.log(`ENV_SERVER_READY port=${port}`);
}
