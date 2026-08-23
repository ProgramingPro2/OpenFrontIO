/**
 * Env backends behind the schema-v2 TCP server.
 *
 * SequentialEnvBackend is the correctness oracle and the default/fallback.
 * ParallelEnvBackend (opt-in) farms complete games across worker threads;
 * each worker advances its games single-threadedly.
 */
import path from "node:path";
import { fileURLToPath } from "node:url";
import { NodeGameMapLoader } from "../../tests/perf/fullgame/NodeGameMapLoader";
import { AgentEnv } from "./AgentEnv";
import { ActionVec } from "./ActionTranslator";
import { resolveAutoReset } from "./autoReset";
import { BatchObs, obsTensors } from "./batchObs";
import { encodeFrame } from "./framing";
import { allocOwnedBatch, AttachedBatch } from "./obsLayout";
import { ObsBuffers } from "./ObsExtractor";
import { EnvConfig } from "./spec";
import { TerrainCache } from "./TerrainCache";

export const PROJECT_ROOT = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../..",
);

export type BackendName = "sequential" | "parallel";

export interface StepProfile {
  simMs: number;
  maskObsMs: number;
  serializeMs: number;
  resetMs: number;
  resets: number;
}

export function emptyProfile(): StepProfile {
  return { simMs: 0, maskObsMs: 0, serializeMs: 0, resetMs: 0, resets: 0 };
}

export function addProfile(dst: StepProfile, src: Partial<StepProfile>): void {
  dst.simMs += src.simMs ?? 0;
  dst.maskObsMs += src.maskObsMs ?? 0;
  dst.serializeMs += src.serializeMs ?? 0;
  dst.resetMs += src.resetMs ?? 0;
  dst.resets += src.resets ?? 0;
}

export interface BackendInitResult {
  k: number;
  backend: BackendName;
  workers: number;
  obs: BatchObs;
}

export interface BackendStepResult {
  rewards: Float32Array;
  dones: Uint8Array;
  infos: Record<string, unknown>[];
  profile: StepProfile;
  obs: BatchObs;
}

export interface EnvBackend {
  readonly name: BackendName;
  readonly workers: number;
  init(configs: EnvConfig[]): Promise<BackendInitResult>;
  step(
    header: Record<string, unknown>,
    blobs: Record<string, { buf: Buffer }>,
  ): Promise<BackendStepResult>;
  reset(header: Record<string, unknown>): Promise<{ index: number; obs: ObsBuffers }>;
  frame(
    header: Record<string, unknown>,
  ): Promise<ReturnType<AgentEnv["renderFrame"]>>;
  close(): Promise<void> | void;
}

export function actionFromBlob(buf: Buffer, index: number): ActionVec {
  const o = index * 20;
  return {
    actionType: buf.readInt32LE(o),
    target: buf.readInt32LE(o + 4),
    region: buf.readInt32LE(o + 8),
    quantity: buf.readInt32LE(o + 12),
    unit: buf.readInt32LE(o + 16),
  };
}

export function encodeInit(result: BackendInitResult): Buffer {
  return encodeFrame(
    { type: "inited", k: result.k, backend: result.backend, workers: result.workers },
    obsTensors(result.obs),
  );
}

export function encodeStep(result: BackendStepResult): Buffer {
  return encodeFrame(
    {
      type: "step",
      rewards: Array.from(result.rewards),
      dones: Array.from(result.dones),
      infos: result.infos,
      profile: result.profile,
    },
    obsTensors(result.obs),
  );
}

export function encodeReset(index: number, obs: ObsBuffers): Buffer {
  return encodeFrame({ type: "reset", index }, obsTensors(stackOne(obs)));
}

function stackOne(obs: ObsBuffers): BatchObs {
  return {
    spatial: obs.spatial,
    players: obs.players,
    global: obs.global,
    actionMask: obs.actionMask,
    targetMasks: obs.targetMasks,
    quantityMask: obs.quantityMask,
    unitMask: obs.unitMask,
    spawnRegions: obs.spawnRegions,
    buildRegions: obs.buildRegions,
    boatRegions: obs.boatRegions,
  };
}

export function newTerrainCache(): TerrainCache {
  return new TerrainCache(
    new NodeGameMapLoader(path.join(PROJECT_ROOT, "resources/maps")),
  );
}

/**
 * Default oracle: one thread, each game advanced in index order.
 * Observations are written into fixed per-env views of one batch buffer.
 */
export class SequentialEnvBackend implements EnvBackend {
  readonly name: BackendName = "sequential";
  readonly workers = 1;
  private envs: AgentEnv[] = [];
  private attached: AttachedBatch | null = null;
  private terrain = newTerrainCache();

  async init(configs: EnvConfig[]): Promise<BackendInitResult> {
    this.attached = allocOwnedBatch(configs.length);
    this.envs = [];
    for (let i = 0; i < configs.length; i++) {
      this.envs.push(
        await AgentEnv.create(configs[i], this.terrain, {
          obs: this.attached.envViews[i],
        }),
      );
    }
    return {
      k: this.envs.length,
      backend: this.name,
      workers: this.workers,
      obs: this.attached.batch,
    };
  }

  async step(
    header: Record<string, unknown>,
    blobs: Record<string, { buf: Buffer }>,
  ): Promise<BackendStepResult> {
    if (this.attached === null) throw new Error("backend not initialized");
    const actionsBuf = blobs.actions.buf;
    const k = this.envs.length;
    const rewards = new Float32Array(k);
    const dones = new Uint8Array(k);
    const infos: Record<string, unknown>[] = [];
    const profile = emptyProfile();
    for (let i = 0; i < k; i++) {
      const env = this.envs[i];
      const result = env.step(actionFromBlob(actionsBuf, i));
      profile.simMs += env.lastProfile.simMs;
      profile.maskObsMs += env.lastProfile.maskObsMs;
      rewards[i] = result.reward;
      dones[i] = result.done ? 1 : 0;
      infos.push(result.info);
      if (result.done) {
        const tReset = performance.now();
        await this.resetDone(env, i, header);
        profile.resetMs += performance.now() - tReset;
        profile.resets += 1;
      }
    }
    return { rewards, dones, infos, profile, obs: this.attached.batch };
  }

  private async resetDone(
    env: AgentEnv,
    index: number,
    header: Record<string, unknown>,
  ): Promise<void> {
    const nextConfigs = header.nextConfigs as
      | Array<Partial<EnvConfig> | string | null | undefined>
      | undefined;
    const { spec, fallback } = resolveAutoReset(
      nextConfigs,
      index,
      env.seed,
      env.resetCount,
    );
    if (fallback) {
      env.resetCount += 1;
    }
    await env.reset(spec);
  }

  async reset(
    header: Record<string, unknown>,
  ): Promise<{ index: number; obs: ObsBuffers }> {
    const index = header.index as number;
    const patch = header.config as Partial<EnvConfig> | undefined;
    const seed = header.seed as string | undefined;
    const obs = await this.envs[index].reset(patch ?? seed);
    return { index, obs };
  }

  async frame(
    header: Record<string, unknown>,
  ): Promise<ReturnType<AgentEnv["renderFrame"]>> {
    const index = (header.index as number) ?? 0;
    return this.envs[index].renderFrame();
  }

  close(): void {
    this.envs = [];
    this.attached = null;
  }
}

export function resolveWorkerCount(requested: number, k: number): number {
  if (!Number.isFinite(requested) || requested <= 1) return 1;
  return Math.max(1, Math.min(Math.floor(requested), k, 32));
}

export async function createEnvBackend(
  requestedWorkers: number,
): Promise<EnvBackend> {
  if (requestedWorkers <= 1) return new SequentialEnvBackend();
  const { ParallelEnvBackend } = await import("./ParallelBackend");
  return new ParallelEnvBackend(requestedWorkers);
}

export function parseServerArgs(argv: string[]): { port: number; workers: number } {
  let port = 8765;
  let workers = Number.parseInt(process.env.OFAI_ENV_WORKERS ?? "0", 10);
  if (!Number.isFinite(workers) || workers < 0) workers = 0;
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === "--port") port = parseInt(argv[++i], 10);
    if (argv[i] === "--workers") workers = parseInt(argv[++i], 10);
  }
  return { port, workers };
}
