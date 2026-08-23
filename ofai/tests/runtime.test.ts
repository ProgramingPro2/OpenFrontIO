/**
 * Official-runtime gates: in-place batch views, sequential oracle vs
 * opt-in parallel workers, and golden / randomized digest parity.
 *
 * Run: npx vitest run ofai/tests/runtime.test.ts
 */
import { describe, expect, it } from "vitest";
import { SequentialEnvBackend } from "../env/EnvBackend";
import { ParallelEnvBackend } from "../env/ParallelBackend";
import { EnvConfig } from "../env/spec";
import { actionFromTuple, loadGolden, mulberry32, randomLegalAction, testConfig } from "./fidelity";

function actionsBlob(actions: Array<{ actionType: number; target: number; region: number; quantity: number; unit: number }>): Buffer {
  const buf = Buffer.alloc(actions.length * 20);
  for (let i = 0; i < actions.length; i++) {
    const o = i * 20;
    const a = actions[i];
    buf.writeInt32LE(a.actionType, o);
    buf.writeInt32LE(a.target, o + 4);
    buf.writeInt32LE(a.region, o + 8);
    buf.writeInt32LE(a.quantity, o + 12);
    buf.writeInt32LE(a.unit, o + 16);
  }
  return buf;
}

function compactCfg(overrides: Partial<EnvConfig> = {}): EnvConfig {
  return testConfig({
    map: "Halkidiki",
    mapSize: "Compact",
    nations: "disabled",
    bots: 0,
    seed: "runtime-parity",
    maxTicks: 800,
    enableDigest: true,
    ...overrides,
  });
}

describe("sequential backend in-place batch", () => {
  it("writes bound env views without a stack copy", async () => {
    const backend = new SequentialEnvBackend();
    try {
      const init = await backend.init([
        compactCfg({ seed: "runtime-a" }),
        compactCfg({ seed: "runtime-b" }),
      ]);
      expect(init.backend).toBe("sequential");
      expect(init.k).toBe(2);
      expect(init.obs.spatial.length).toBe(2 * 10 * 64 * 64);
      const noop = {
        actionType: 0,
        target: 0,
        region: 0,
        quantity: 0,
        unit: 0,
      };
      const stepped = await backend.step(
        {},
        { actions: { buf: actionsBlob([noop, noop]) } },
      );
      expect(stepped.rewards.length).toBe(2);
      expect(stepped.obs.spatial.buffer).toBe(init.obs.spatial.buffer);
      expect(stepped.infos[0].digest).toMatch(/^[0-9a-f]{64}$/);
    } finally {
      await backend.close();
    }
  }, 180000);
});

describe("parallel vs sequential oracle", () => {
  it("replays the spawn golden with matching digests on two workers", async () => {
    const golden = loadGolden("spawn");
    const cfg = { ...golden.config, enableDigest: true };
    const configs = [cfg, { ...cfg, seed: `${cfg.seed}-twin` }];
    const seq = new SequentialEnvBackend();
    const par = new ParallelEnvBackend(2);
    try {
      const seqInit = await seq.init(configs);
      const parInit = await par.init(configs);
      expect(parInit.backend).toBe("parallel");
      expect(parInit.workers).toBe(2);
      expect(Array.from(parInit.obs.spatial)).toEqual(Array.from(seqInit.obs.spatial));
      expect(Array.from(parInit.obs.actionMask)).toEqual(
        Array.from(seqInit.obs.actionMask),
      );

      for (const tuple of golden.actions as number[][]) {
        const action = actionFromTuple(tuple as [number, number, number, number, number]);
        const twin = { ...action };
        const blob = actionsBlob([action, twin]);
        const s = await seq.step({}, { actions: { buf: blob } });
        const p = await par.step({}, { actions: { buf: blob } });
        expect(p.rewards[0]).toBe(s.rewards[0]);
        expect(p.dones[0]).toBe(s.dones[0]);
        expect(p.infos[0].digest).toBe(s.infos[0].digest);
        expect(p.infos[0].tileDigest).toBe(s.infos[0].tileDigest);
        expect(p.infos[0].obsDigest).toBe(s.infos[0].obsDigest);
        expect(p.infos[0].terminalCause).toBe(s.infos[0].terminalCause);
        expect(p.infos[0].intentCount).toBe(s.infos[0].intentCount);
        expect(Array.from(p.obs.actionMask)).toEqual(Array.from(s.obs.actionMask));
        if (s.dones[0]) break;
      }
    } finally {
      await seq.close();
      await par.close();
    }
  }, 240000);

  it("keeps two workers byte-equal to sequential on a randomized trace", async () => {
    const cfg = compactCfg({
      seed: "runtime-diff",
      bots: 1,
      maxTicks: 1200,
    });
    const configs = [cfg, { ...cfg, seed: "runtime-diff-b" }];
    const seq = new SequentialEnvBackend();
    const par = new ParallelEnvBackend(2);
    try {
    const seqInit = await seq.init(configs);
    const parInit = await par.init(configs);
    expect(Array.from(parInit.obs.spatial)).toEqual(Array.from(seqInit.obs.spatial));
    const rng = mulberry32(0x51a7);
    let obs0 = seqInit.obs;

    for (let i = 0; i < 8; i++) {
      const a0 = randomLegalAction(
        {
          actionMask: obs0.actionMask.subarray(0, 9),
          targetMasks: obs0.targetMasks.subarray(0, 9 * 16),
          quantityMask: obs0.quantityMask.subarray(0, 5),
          unitMask: obs0.unitMask.subarray(0, 10),
          spawnRegions: obs0.spawnRegions.subarray(0, 1024),
          buildRegions: obs0.buildRegions.subarray(0, 1024),
          boatRegions: obs0.boatRegions.subarray(0, 1024),
        },
        rng,
      );
      const a1 = randomLegalAction(
        {
          actionMask: obs0.actionMask.subarray(9, 18),
          targetMasks: obs0.targetMasks.subarray(9 * 16, 18 * 16),
          quantityMask: obs0.quantityMask.subarray(5, 10),
          unitMask: obs0.unitMask.subarray(10, 20),
          spawnRegions: obs0.spawnRegions.subarray(1024, 2048),
          buildRegions: obs0.buildRegions.subarray(1024, 2048),
          boatRegions: obs0.boatRegions.subarray(1024, 2048),
        },
        rng,
      );
      const blob = actionsBlob([a0, a1]);
      const s = await seq.step({}, { actions: { buf: blob } });
      const p = await par.step({}, { actions: { buf: blob } });
      expect(Array.from(p.rewards)).toEqual(Array.from(s.rewards));
      expect(Array.from(p.dones)).toEqual(Array.from(s.dones));
      expect(p.infos[0].digest).toBe(s.infos[0].digest);
      expect(p.infos[1].digest).toBe(s.infos[1].digest);
      expect(p.infos[0].coreHash ?? p.infos[0].hash).toBe(
        s.infos[0].coreHash ?? s.infos[0].hash,
      );
      expect(Array.from(p.obs.spatial)).toEqual(Array.from(s.obs.spatial));
      expect(Array.from(p.obs.targetMasks)).toEqual(Array.from(s.obs.targetMasks));
      obs0 = s.obs;
      if (s.dones[0] && s.dones[1]) break;
    }
    } finally {
      await seq.close();
      await par.close();
    }
  }, 240000);
});
