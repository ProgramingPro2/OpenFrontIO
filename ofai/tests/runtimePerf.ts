/**
 * Representative official-runtime throughput bench.
 *
 * Spawns, then steps attack/noop so mask/obs and simulation are live.
 * Compares sequential oracle vs opt-in parallel workers. Not a Python e2e farm.
 *
 *   npx tsx ofai/tests/runtimePerf.ts
 *   OFAI_PERF_K=8 OFAI_PERF_STEPS=12 OFAI_PERF_WORKERS=4 npx tsx ofai/tests/runtimePerf.ts
 */
import { AgentEnv } from "../env/AgentEnv";
import { newTerrainCache, SequentialEnvBackend } from "../env/EnvBackend";
import { ParallelEnvBackend } from "../env/ParallelBackend";
import { EnvConfig, NUM_REGIONS } from "../env/spec";

interface Row {
  name: string;
  backend: string;
  workers: number;
  k: number;
  steps: number;
  wallMs: number;
  stepsPerSec: number;
  simMs: number;
  maskObsMs: number;
  resetMs: number;
}

const DUEL: EnvConfig = {
  map: "Halkidiki",
  mapSize: "Compact",
  nations: "disabled",
  difficulty: "Easy",
  bots: 1,
  seed: "runtime-perf-duel",
  maxTicks: 4000,
  decisionInterval: 10,
  shaping: 0,
};

const NATIONS: EnvConfig = {
  map: "FourIslands",
  mapSize: "Normal",
  nations: 3,
  difficulty: "Easy",
  bots: 4,
  seed: "runtime-perf-nations",
  maxTicks: 4000,
  decisionInterval: 10,
  shaping: 0,
};

function configs(base: EnvConfig, k: number): EnvConfig[] {
  return Array.from({ length: k }, (_, i) => ({
    ...base,
    seed: `${base.seed}-${i}`,
  }));
}

function spawnBlob(spawnRegions: Uint8Array, k: number): Buffer {
  const buf = Buffer.alloc(k * 20);
  for (let i = 0; i < k; i++) {
    let region = 0;
    const off = i * NUM_REGIONS;
    for (let r = 0; r < NUM_REGIONS; r++) {
      if (spawnRegions[off + r] === 1) {
        region = r;
        break;
      }
    }
    buf.writeInt32LE(1, i * 20);
    buf.writeInt32LE(region, i * 20 + 8);
    buf.writeInt32LE(2, i * 20 + 12);
  }
  return buf;
}

function attackBlob(k: number): Buffer {
  const buf = Buffer.alloc(k * 20);
  for (let i = 0; i < k; i++) {
    buf.writeInt32LE(2, i * 20);
    buf.writeInt32LE(0, i * 20 + 4);
    buf.writeInt32LE(4, i * 20 + 12);
  }
  return buf;
}

async function benchBackend(
  name: string,
  backendName: "sequential" | "parallel",
  workers: number,
  base: EnvConfig,
  k: number,
  steps: number,
): Promise<Row> {
  const backend =
    backendName === "parallel"
      ? new ParallelEnvBackend(workers)
      : new SequentialEnvBackend();
  const tInit = performance.now();
  const init = await backend.init(configs(base, k));
  const initMs = performance.now() - tInit;
  await backend.step({}, { actions: { buf: spawnBlob(init.obs.spawnRegions, k) } });
  const blob = { actions: { buf: attackBlob(k) } };
  await backend.step({}, blob);
  let simMs = 0;
  let maskObsMs = 0;
  let resetMs = 0;
  const t0 = performance.now();
  for (let i = 0; i < steps; i++) {
    const result = await backend.step({}, blob);
    simMs += result.profile.simMs;
    maskObsMs += result.profile.maskObsMs;
    resetMs += result.profile.resetMs;
  }
  const wallMs = performance.now() - t0;
  await backend.close();
  const stepsTotal = steps * k;
  console.error(
    `[runtime-perf] ${name} init=${initMs.toFixed(0)}ms wall=${wallMs.toFixed(0)}ms`,
  );
  return {
    name,
    backend: backendName,
    workers: backendName === "parallel" ? workers : 1,
    k,
    steps,
    wallMs,
    stepsPerSec: (stepsTotal / wallMs) * 1000,
    simMs,
    maskObsMs,
    resetMs,
  };
}

async function benchMaskRefill(): Promise<{
  warmMs: number;
  coldMs: number;
  n: number;
}> {
  const terrain = newTerrainCache();
  const env = await AgentEnv.create(
    { ...DUEL, seed: "runtime-perf-mask" },
    terrain,
  );
  const spawn = env.peekObs();
  let region = 0;
  for (let i = 0; i < spawn.spawnRegions.length; i++) {
    if (spawn.spawnRegions[i] === 1) {
      region = i;
      break;
    }
  }
  env.step({
    actionType: 1,
    target: 0,
    region,
    quantity: 2,
    unit: 0,
  });
  const n = 20;
  env.clearTranslatorCachesForTest();
  const tCold = performance.now();
  for (let i = 0; i < n; i++) {
    env.clearTranslatorCachesForTest();
    env.refillMasksForTest();
  }
  const coldMs = (performance.now() - tCold) / n;
  const tWarm = performance.now();
  for (let i = 0; i < n; i++) env.refillMasksForTest();
  const warmMs = (performance.now() - tWarm) / n;
  return { warmMs, coldMs, n };
}

function fmt(row: Row): string {
  return [
    row.name.padEnd(28),
    row.backend.padEnd(11),
    String(row.workers).padStart(2),
    String(row.k).padStart(2),
    row.stepsPerSec.toFixed(2).padStart(8),
    row.wallMs.toFixed(0).padStart(7),
    row.simMs.toFixed(0).padStart(7),
    row.maskObsMs.toFixed(0).padStart(8),
  ].join(" ");
}

async function main(): Promise<void> {
  const k = Number.parseInt(process.env.OFAI_PERF_K ?? "8", 10);
  const steps = Number.parseInt(process.env.OFAI_PERF_STEPS ?? "10", 10);
  const workers = Number.parseInt(process.env.OFAI_PERF_WORKERS ?? "4", 10);
  const rows: Row[] = [];
  for (const [label, cfg] of [
    ["duel", DUEL],
    ["nations", NATIONS],
  ] as const) {
    rows.push(
      await benchBackend(`${label}-sequential`, "sequential", 1, cfg, k, steps),
    );
    rows.push(
      await benchBackend(`${label}-parallel`, "parallel", workers, cfg, k, steps),
    );
  }
  const mask = await benchMaskRefill();
  console.log(
    [
      "name".padEnd(28),
      "backend".padEnd(11),
      "w".padStart(2),
      "k".padStart(2),
      "step/s".padStart(8),
      "wallMs".padStart(7),
      "simMs".padStart(7),
      "maskMs".padStart(8),
    ].join(" "),
  );
  for (const row of rows) console.log(fmt(row));
  const duelX = rows[1].stepsPerSec / rows[0].stepsPerSec;
  const natX = rows[3].stepsPerSec / rows[2].stepsPerSec;
  console.log(
    `mask refill cold ${mask.coldMs.toFixed(2)}ms  warm ${mask.warmMs.toFixed(2)}ms  over ${mask.n} calls`,
  );
  console.log(
    `speedup duel ${duelX.toFixed(2)}x  nations ${natX.toFixed(2)}x  (parallel/${workers} vs sequential, K=${k}, steps=${steps})`,
  );
  console.log(
    JSON.stringify(
      {
        rows,
        maskRefill: mask,
        speedup: { duel: duelX, nations: natX },
        k,
        steps,
        workers,
      },
      null,
      2,
    ),
  );
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
