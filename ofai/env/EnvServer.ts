/**
 * Vectorized env server: runs K AgentEnv instances in one Node process and
 * serves batched reset/step over length-prefixed binary TCP frames.
 *
 * Protocol (see framing.ts for the wire format):
 *   <- {cmd:"init", envs:[EnvConfig,...]}        -> {type:"inited"} + obs tensors
 *   <- {cmd:"step", actions:[[a,t,r,q,u],...]}   -> {type:"step", rewards, dones, infos} + obs tensors
 *   <- {cmd:"reset", index, seed?}               -> {type:"reset"} + obs tensors (K=1)
 *   <- {cmd:"close"}                             -> server exits
 *
 * Run: npx tsx ofai/env/EnvServer.ts --port 8765
 */
import net from "node:net";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { NodeGameMapLoader } from "../../tests/perf/fullgame/NodeGameMapLoader";
import { AgentEnv } from "./AgentEnv";
import { ActionVec } from "./ActionTranslator";
import { BatchObs, stackObs } from "./batchObs";
import { encodeFrame, FrameDecoder, frameTensors } from "./framing";
import { ObsBuffers } from "./ObsExtractor";
import { TerrainCache } from "./TerrainCache";
import { EnvConfig } from "./spec";

const PROJECT_ROOT = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../..",
);

function obsTensors(batch: BatchObs) {
  return {
    spatial: { dtype: "f32" as const, data: batch.spatial },
    players: { dtype: "f32" as const, data: batch.players },
    global: { dtype: "f32" as const, data: batch.global },
    action_mask: { dtype: "u8" as const, data: batch.actionMask },
    // Flattened [K, NUM_ACTION_TYPES, NUM_PLAYER_SLOTS] row-major.
    target_masks: { dtype: "u8" as const, data: batch.targetMasks },
    quantity_mask: { dtype: "u8" as const, data: batch.quantityMask },
    unit_mask: { dtype: "u8" as const, data: batch.unitMask },
    spawn_regions: { dtype: "u8" as const, data: batch.spawnRegions },
    build_regions: { dtype: "u8" as const, data: batch.buildRegions },
    boat_regions: { dtype: "u8" as const, data: batch.boatRegions },
  };
}

class Server {
  private envs: AgentEnv[] = [];
  private batch: BatchObs | null = null;
  private terrain = new TerrainCache(
    new NodeGameMapLoader(path.join(PROJECT_ROOT, "resources/maps")),
  );

  private stacked(obsList: ObsBuffers[]): BatchObs {
    this.batch = stackObs(obsList, this.batch ?? undefined);
    return this.batch;
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
        return this.step(header, blobs);
      case "reset":
        return this.reset(header);
      case "frame":
        return this.frame(header);
      case "close":
        setTimeout(() => process.exit(0), 100);
        return encodeFrame({ type: "closed" });
      default:
        return encodeFrame({ type: "error", error: `unknown cmd ${cmd}` });
    }
  }

  private async init(configs: EnvConfig[]): Promise<Buffer> {
    console.error(`[env-server] init: ${configs.length} envs`);
    const t0 = performance.now();
    this.envs = [];
    // Create sequentially: map bin loading is I/O heavy but cached by Node.
    for (const cfg of configs) {
      this.envs.push(await AgentEnv.create(cfg, this.terrain));
    }
    const obs = this.envs.map((e) => e.peekObs());
    console.error(
      `[env-server] obs bytes/env: ${obs[0].spatial.byteLength + obs[0].players.byteLength} (spatial+players)`,
    );
    console.error(
      `[env-server] init done in ${(performance.now() - t0).toFixed(0)}ms`,
    );
    return encodeFrame(
      { type: "inited", k: this.envs.length },
      obsTensors(this.stacked(obs)),
    );
  }

  private async step(
    header: Record<string, unknown>,
    blobs: Record<string, { buf: Buffer }>,
  ): Promise<Buffer> {
    const actionsBuf = blobs.actions.buf;
    const k = this.envs.length;
    const rewards = new Float32Array(k);
    const dones = new Uint8Array(k);
    const infos: Record<string, unknown>[] = [];
    const obsList: ObsBuffers[] = [];
    for (let i = 0; i < k; i++) {
      const a: ActionVec = {
        actionType: actionsBuf.readInt32LE(i * 20),
        target: actionsBuf.readInt32LE(i * 20 + 4),
        region: actionsBuf.readInt32LE(i * 20 + 8),
        quantity: actionsBuf.readInt32LE(i * 20 + 12),
        unit: actionsBuf.readInt32LE(i * 20 + 16),
      };
      const env = this.envs[i];
      const result = env.step(a);
      rewards[i] = result.reward;
      dones[i] = result.done ? 1 : 0;
      infos.push(result.info);
      if (result.done) {
        // Auto-reset: next episode starts with a fresh seed derived from the
        // old one so streams never repeat within a run.
        const newSeed = `${env.seed}-r${env.resetCount++}`;
        obsList.push(await env.reset(newSeed));
      } else {
        obsList.push(result.obs);
      }
    }
    return encodeFrame(
      { type: "step", rewards: Array.from(rewards), dones: Array.from(dones), infos },
      obsTensors(this.stacked(obsList)),
    );
  }

  private async reset(header: Record<string, unknown>): Promise<Buffer> {
    const index = header.index as number;
    const seed = header.seed as string | undefined;
    const obs = await this.envs[index].reset(seed);
    return encodeFrame(
      { type: "reset", index },
      obsTensors(this.stacked([obs])),
    );
  }

  private frame(header: Record<string, unknown>): Buffer {
    const index = (header.index as number) ?? 0;
    const f = this.envs[index].renderFrame();
    // Units are small and variable-length; send as JSON in the header.
    // The map cells are the big payload and go as a u8 tensor.
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

function parseArgs(argv: string[]): { port: number } {
  let port = 8765;
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === "--port") port = parseInt(argv[++i], 10);
  }
  return { port };
}

const { port } = parseArgs(process.argv.slice(2));
const server = new Server();

net
  .createServer((socket) => {
    console.error("[env-server] client connected");
    const decoder = new FrameDecoder();
    // Serialize handlers: overlapping `data` events must not interleave
    // step/reset on the same AgentEnv instances.
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
    console.error(`[env-server] listening on port ${port}`);
  });

// Marker so the Python spawner knows the server is ready.
console.log(`ENV_SERVER_READY port=${port}`);
