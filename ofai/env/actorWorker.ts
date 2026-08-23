/**
 * Worker-thread actor: owns a slice of complete AgentEnv games and advances
 * each game on this thread only. Observations are written into SharedArrayBuffer
 * slices allocated by the parent; actions are read from a shared i32 buffer.
 */
import { parentPort } from "node:worker_threads";
import { AgentEnv } from "./AgentEnv";
import { ActionVec } from "./ActionTranslator";
import { resolveAutoReset } from "./autoReset";
import { ACTION_STRIDE, attachBatch } from "./obsLayout";
import { EnvConfig } from "./spec";
import { emptyProfile, newTerrainCache, StepProfile } from "./EnvBackend";

interface InitMsg {
  type: "init";
  envIndices: number[];
  configs: EnvConfig[];
  k: number;
  obsSab: SharedArrayBuffer;
  actionsSab: SharedArrayBuffer;
}

interface StepMsg {
  type: "step";
  nextConfigs?: Array<Partial<EnvConfig> | string | null | undefined>;
}

interface ResetMsg {
  type: "reset";
  globalIndex: number;
  spec?: string | Partial<EnvConfig>;
}

interface FrameMsg {
  type: "frame";
  globalIndex: number;
}

interface CloseMsg {
  type: "close";
}

type WorkerMsg = InitMsg | StepMsg | ResetMsg | FrameMsg | CloseMsg;

const port = parentPort;
if (port === null) {
  throw new Error("actorWorker must run as a worker thread");
}

let envIndices: number[] = [];
let envs: AgentEnv[] = [];
let actionsView: Int32Array | null = null;
const terrain = newTerrainCache();

function actionAt(globalIndex: number): ActionVec {
  if (actionsView === null) throw new Error("actions buffer missing");
  const o = globalIndex * ACTION_STRIDE;
  return {
    actionType: actionsView[o],
    target: actionsView[o + 1],
    region: actionsView[o + 2],
    quantity: actionsView[o + 3],
    unit: actionsView[o + 4],
  };
}

function localOf(globalIndex: number): number {
  const local = envIndices.indexOf(globalIndex);
  if (local < 0) throw new Error(`env ${globalIndex} not owned by this worker`);
  return local;
}

async function handleInit(msg: InitMsg): Promise<void> {
  envIndices = msg.envIndices;
  actionsView = new Int32Array(msg.actionsSab);
  const { envViews } = attachBatch(msg.obsSab, msg.k);
  envs = [];
  for (let i = 0; i < envIndices.length; i++) {
    const gi = envIndices[i];
    envs.push(
      await AgentEnv.create(msg.configs[i], terrain, { obs: envViews[gi] }),
    );
  }
  port.postMessage({ type: "inited", count: envs.length });
}

async function handleStep(msg: StepMsg): Promise<void> {
  const rewards = new Float32Array(envs.length);
  const dones = new Uint8Array(envs.length);
  const infos: Record<string, unknown>[] = [];
  const profile: StepProfile = emptyProfile();
  for (let i = 0; i < envs.length; i++) {
    const env = envs[i];
    const result = env.step(actionAt(envIndices[i]));
    profile.simMs += env.lastProfile.simMs;
    profile.maskObsMs += env.lastProfile.maskObsMs;
    rewards[i] = result.reward;
    dones[i] = result.done ? 1 : 0;
    infos.push(result.info);
    if (result.done) {
      const tReset = performance.now();
      const { spec, fallback } = resolveAutoReset(
        msg.nextConfigs,
        envIndices[i],
        env.seed,
        env.resetCount,
      );
      if (fallback) env.resetCount += 1;
      await env.reset(spec);
      profile.resetMs += performance.now() - tReset;
      profile.resets += 1;
    }
  }
  port.postMessage({
    type: "step",
    envIndices,
    rewards: Array.from(rewards),
    dones: Array.from(dones),
    infos,
    profile,
  });
}

async function handleReset(msg: ResetMsg): Promise<void> {
  const local = localOf(msg.globalIndex);
  await envs[local].reset(msg.spec);
  port.postMessage({ type: "reset", globalIndex: msg.globalIndex });
}

function handleFrame(msg: FrameMsg): void {
  const local = localOf(msg.globalIndex);
  const frame = envs[local].renderFrame();
  port.postMessage({ type: "frame", globalIndex: msg.globalIndex, frame });
}

port.on("message", (msg: WorkerMsg) => {
  const run = async () => {
    switch (msg.type) {
      case "init":
        await handleInit(msg);
        break;
      case "step":
        await handleStep(msg);
        break;
      case "reset":
        await handleReset(msg);
        break;
      case "frame":
        handleFrame(msg);
        break;
      case "close":
        envs = [];
        port.postMessage({ type: "closed" });
        break;
      default:
        port.postMessage({
          type: "error",
          error: `unknown worker cmd ${(msg as { type: string }).type}`,
        });
    }
  };
  run().catch((err: unknown) => {
    const message = err instanceof Error ? err.message : String(err);
    port.postMessage({ type: "error", error: message });
  });
});
