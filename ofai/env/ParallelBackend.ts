/**
 * Opt-in parallel official-core backend.
 *
 * Each worker owns a disjoint slice of complete games and steps those games
 * single-threadedly. Workers run different games concurrently. Observations
 * live in a SharedArrayBuffer so the parent never copies tensors back.
 *
 * SequentialEnvBackend remains the oracle and the fallback when workers<=1
 * or worker startup fails.
 */
import { existsSync } from "node:fs";
import path from "node:path";
import { Worker } from "node:worker_threads";
import { AgentEnv } from "./AgentEnv";
import {
  addProfile,
  BackendInitResult,
  BackendStepResult,
  emptyProfile,
  EnvBackend,
  PROJECT_ROOT,
  resolveWorkerCount,
  StepProfile,
} from "./EnvBackend";
import { ObsBuffers } from "./ObsExtractor";
import {
  allocSharedActions,
  allocSharedBatch,
  copyActionsFromBlob,
} from "./obsLayout";
import { EnvConfig } from "./spec";

export interface WorkerStepReply {
  type: "step";
  envIndices: number[];
  rewards: number[];
  dones: number[];
  infos: Record<string, unknown>[];
  profile: StepProfile;
}

function actorWorkerPath(): string {
  const fromRoot = path.join(PROJECT_ROOT, "ofai/env/actorWorkerEntry.mjs");
  if (existsSync(fromRoot)) return fromRoot;
  return path.join(process.cwd(), "ofai/env/actorWorkerEntry.mjs");
}

function spawnActorWorker(): Worker {
  return new Worker(actorWorkerPath());
}

function partitionIndices(k: number, workers: number): number[][] {
  const n = resolveWorkerCount(workers, k);
  const groups: number[][] = Array.from({ length: n }, () => []);
  for (let i = 0; i < k; i++) {
    groups[i % n].push(i);
  }
  return groups.filter((g) => g.length > 0);
}

function callWorker<T>(worker: Worker, msg: object): Promise<T> {
  return new Promise((resolve, reject) => {
    const onMessage = (reply: { type?: string; error?: string }) => {
      cleanup();
      if (reply?.type === "error") {
        reject(new Error(reply.error ?? "worker error"));
        return;
      }
      resolve(reply as T);
    };
    const onError = (err: Error) => {
      cleanup();
      reject(err);
    };
    const cleanup = () => {
      worker.off("message", onMessage);
      worker.off("error", onError);
    };
    worker.on("message", onMessage);
    worker.on("error", onError);
    worker.postMessage(msg);
  });
}

export class ParallelEnvBackend implements EnvBackend {
  readonly name = "parallel" as const;
  readonly workers: number;
  private requestedWorkers: number;
  private pool: Worker[] = [];
  private groups: number[][] = [];
  private obsSab: SharedArrayBuffer | null = null;
  private actionsSab: SharedArrayBuffer | null = null;
  private actionsView: Int32Array | null = null;
  private envViews: ObsBuffers[] = [];
  private batch = allocSharedBatch(1).attached.batch;
  private k = 0;

  constructor(requestedWorkers: number) {
    this.requestedWorkers = requestedWorkers;
    this.workers = requestedWorkers;
  }

  async init(configs: EnvConfig[]): Promise<BackendInitResult> {
    this.k = configs.length;
    const groups = partitionIndices(this.k, this.requestedWorkers);
    const { sab, attached } = allocSharedBatch(this.k);
    const actions = allocSharedActions(this.k);
    this.obsSab = sab;
    this.actionsSab = actions.sab;
    this.actionsView = actions.view;
    this.batch = attached.batch;
    this.envViews = attached.envViews;
    this.groups = groups;

    const workers: Worker[] = [];
    try {
      for (let w = 0; w < groups.length; w++) {
        const worker = spawnActorWorker();
        workers.push(worker);
        const envIndices = groups[w];
        await callWorker(worker, {
          type: "init",
          envIndices,
          configs: envIndices.map((i) => configs[i]),
          k: this.k,
          obsSab: sab,
          actionsSab: actions.sab,
        });
      }
    } catch (err) {
      for (const worker of workers) {
        await worker.terminate();
      }
      throw err;
    }
    this.pool = workers;
    return {
      k: this.k,
      backend: this.name,
      workers: this.pool.length,
      obs: this.batch,
    };
  }

  async step(
    header: Record<string, unknown>,
    blobs: Record<string, { buf: Buffer }>,
  ): Promise<BackendStepResult> {
    if (this.actionsView === null) throw new Error("backend not initialized");
    copyActionsFromBlob(blobs.actions.buf, this.actionsView, this.k);
    const replies = await Promise.all(
      this.pool.map((worker) =>
        callWorker<WorkerStepReply>(worker, {
          type: "step",
          nextConfigs: header.nextConfigs,
        }),
      ),
    );
    const rewards = new Float32Array(this.k);
    const dones = new Uint8Array(this.k);
    const infos: Record<string, unknown>[] = new Array(this.k);
    const profile = emptyProfile();
    for (const reply of replies) {
      addProfile(profile, reply.profile);
      for (let i = 0; i < reply.envIndices.length; i++) {
        const gi = reply.envIndices[i];
        rewards[gi] = reply.rewards[i];
        dones[gi] = reply.dones[i];
        infos[gi] = reply.infos[i];
      }
    }
    return { rewards, dones, infos, profile, obs: this.batch };
  }

  async reset(
    header: Record<string, unknown>,
  ): Promise<{ index: number; obs: ObsBuffers }> {
    const index = header.index as number;
    const worker = this.owner(index);
    const patch = header.config as Partial<EnvConfig> | undefined;
    const seed = header.seed as string | undefined;
    await callWorker(worker, {
      type: "reset",
      globalIndex: index,
      spec: patch ?? seed,
    });
    return { index, obs: this.envViews[index] };
  }

  async frame(
    header: Record<string, unknown>,
  ): Promise<ReturnType<AgentEnv["renderFrame"]>> {
    const index = (header.index as number) ?? 0;
    const reply = await callWorker<{
      frame: ReturnType<AgentEnv["renderFrame"]>;
    }>(this.owner(index), { type: "frame", globalIndex: index });
    return reply.frame;
  }

  async close(): Promise<void> {
    await Promise.all(
      this.pool.map(async (worker) => {
        try {
          await callWorker(worker, { type: "close" });
        } catch {
          // ignore
        }
        await worker.terminate();
      }),
    );
    this.pool = [];
  }

  private owner(globalIndex: number): Worker {
    for (let w = 0; w < this.groups.length; w++) {
      if (this.groups[w].includes(globalIndex)) return this.pool[w];
    }
    throw new Error(`no worker owns env ${globalIndex}`);
  }
}

