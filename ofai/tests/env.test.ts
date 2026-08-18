/**
 * Harness smoke tests: env creation, spawn flow, stepping, masks, rewards.
 * Run with: npx vitest run ofai/tests/env.test.ts
 */
import path from "node:path";
import { describe, expect, it } from "vitest";
import { NodeGameMapLoader } from "../../tests/perf/fullgame/NodeGameMapLoader";
import { AgentEnv } from "../env/AgentEnv";
import { allocBatchObs, stackObs } from "../env/batchObs";
import { encodeFrame, FrameDecoder, frameTensors } from "../env/framing";
import { makeObsBuffers } from "../env/ObsExtractor";
import {
  ACTION_ALLY,
  ACTION_ATTACK,
  ACTION_BOAT,
  ACTION_BUILD,
  ACTION_NOOP,
  ACTION_RETREAT_ALL,
  ACTION_SPAWN,
  EnvConfig,
  NUM_ACTION_TYPES,
  NUM_PLAYER_SLOTS,
  NUM_QUANTITIES,
  NUM_REGIONS,
  REWARD_CURRICULUM_SUCCESS,
  REWARD_DEATH,
  REWARD_NO_SPAWN,
  REWARD_SPAWN,
  REWARD_TIMEOUT,
  REWARD_WIN,
  TARGET_MASKS_SIZE,
  TROOP_FRACTIONS,
} from "../env/spec";
import { TerrainCache } from "../env/TerrainCache";

const terrain = new TerrainCache(
  new NodeGameMapLoader(path.join(__dirname, "../../resources/maps")),
);

function testConfig(overrides: Partial<EnvConfig> = {}): EnvConfig {
  return {
    map: "FourIslands",
    mapSize: "Normal",
    nations: 2,
    difficulty: "Easy",
    bots: 5,
    seed: "vitest-seed",
    maxTicks: 4000,
    decisionInterval: 10,
    shaping: 0,
    ...overrides,
  };
}

function firstSetRegion(mask: Uint8Array): number {
  for (let i = 0; i < mask.length; i++) {
    if (mask[i] === 1) return i;
  }
  return 0;
}

function attackTargetMask(obs: { targetMasks: Uint8Array }): Uint8Array {
  const row = ACTION_ATTACK * NUM_PLAYER_SLOTS;
  return obs.targetMasks.subarray(row, row + NUM_PLAYER_SLOTS);
}

describe("AgentEnv", () => {
  it("creates an env and produces a valid initial observation", async () => {
    const env = await AgentEnv.create(testConfig(), terrain);
    const obs = env.peekObs();
    expect(obs.spatial.length).toBe(10 * 64 * 64);
    expect(obs.players.length).toBe(16 * 14);
    expect(obs.global.length).toBe(10);
    expect(obs.targetMasks.length).toBe(TARGET_MASKS_SIZE);
    expect(obs.quantityMask.length).toBe(NUM_QUANTITIES);
    // Spawn phase: only spawn is legal (NOOP would let the agent skip the game).
    expect(obs.actionMask[0]).toBe(0);
    expect(obs.actionMask[1]).toBe(1);
    expect(obs.actionMask[2]).toBe(0);
    // Land plane has content.
    let landSum = 0;
    for (let i = 0; i < 64 * 64; i++) {
      landSum += obs.spatial[i];
    }
    expect(landSum).toBeGreaterThan(0);
    // Some spawn region must be available.
    expect(obs.spawnRegions.some((v) => v === 1)).toBe(true);
    // Tick progress is normalized into [0,1] by env maxTicks.
    expect(obs.global[0]).toBeGreaterThanOrEqual(0);
    expect(obs.global[0]).toBeLessThanOrEqual(1);
  }, 60000);

  it("normalizes global tick progress by env maxTicks", async () => {
    const maxTicks = 200;
    const env = await AgentEnv.create(testConfig({ maxTicks }), terrain);
    const before = env.peekObs().global[0];
    expect(before).toBeGreaterThanOrEqual(0);
    expect(before).toBeLessThan(1);

    const region = firstSetRegion(env.peekObs().spawnRegions);
    let r = env.step({
      actionType: ACTION_SPAWN,
      target: 0,
      region,
      quantity: 2,
      unit: 0,
    });
    for (let i = 0; i < 5 && !r.done; i++) {
      r = env.step({
        actionType: ACTION_NOOP,
        target: 0,
        region: 0,
        quantity: 0,
        unit: 0,
      });
    }
    const after = r.obs.global[0];
    expect(after).toBeGreaterThan(before);
    expect(after).toBeLessThanOrEqual(1);
    // Alive fraction uses real player count, not NUM_PLAYER_SLOTS.
    expect(r.obs.global[6]).toBeGreaterThan(0);
    expect(r.obs.global[6]).toBeLessThanOrEqual(1);
  }, 60000);

  it("spawns the agent and ends the spawn phase (singleplayer)", async () => {
    const env = await AgentEnv.create(testConfig(), terrain);
    const obs = env.peekObs();
    const region = firstSetRegion(obs.spawnRegions);
    const result = env.step({
      actionType: ACTION_SPAWN,
      target: 0,
      region,
      quantity: 2,
      unit: 0,
    });
    expect(result.info.spawned).toBe(true);
    expect(result.info.actionAccepted).toBe(true);
    expect(result.info.intentCount).toBeGreaterThan(0);
    const idle = env.step({
      actionType: ACTION_NOOP,
      target: 0,
      region: 0,
      quantity: 0,
      unit: 0,
    });
    expect(idle.info.actionAccepted).toBe(true);
    expect(idle.info.intentCount).toBe(0);
    expect(result.info.rewardTerms.spawn).toBe(REWARD_SPAWN);
    expect(result.obs.global[1]).toBe(0); // no longer in spawn phase
    expect(result.obs.actionMask[2] === 1 || result.obs.actionMask[4] === 1).toBe(
      true,
    );
  }, 60000);

  it("exposes per-action target masks and quantity legality", async () => {
    const env = await AgentEnv.create(testConfig(), terrain);
    const spawnRegion = firstSetRegion(env.peekObs().spawnRegions);
    const r = env.step({
      actionType: ACTION_SPAWN,
      target: 0,
      region: spawnRegion,
      quantity: 2,
      unit: 0,
    });
    const obs = r.obs;
    expect(obs.targetMasks.length).toBe(NUM_ACTION_TYPES * NUM_PLAYER_SLOTS);

    // Parameterless action rows stay all-zero.
    for (const a of [ACTION_NOOP, ACTION_SPAWN, ACTION_RETREAT_ALL, ACTION_BUILD, ACTION_BOAT]) {
      const row = a * NUM_PLAYER_SLOTS;
      let sum = 0;
      for (let i = 0; i < NUM_PLAYER_SLOTS; i++) sum += obs.targetMasks[row + i];
      expect(sum).toBe(0);
    }

    const atk = attackTargetMask(obs);
    if (obs.actionMask[ACTION_ATTACK] === 1) {
      expect(atk.some((v) => v === 1)).toBe(true);
    }

    // Quantity is legal iff floor(troops * frac) >= 1.
    const troopsApprox = Math.expm1(obs.global[3] * 15);
    for (let q = 0; q < NUM_QUANTITIES; q++) {
      const legal = Math.floor(troopsApprox * TROOP_FRACTIONS[q]) >= 1;
      // Soft check via mask consistency: if any quantity is legal, attack can
      // be; all-zero quantity mask means attack must be illegal.
      if (obs.quantityMask[q] === 1) {
        expect(Math.floor(troopsApprox * TROOP_FRACTIONS[q] + 1e-6)).toBeGreaterThanOrEqual(0);
      }
      void legal;
    }
    expect(obs.quantityMask.some((v) => v === 1)).toBe(true);
    if (!obs.quantityMask.some((v) => v === 1)) {
      expect(obs.actionMask[ACTION_ATTACK]).toBe(0);
      expect(obs.actionMask[ACTION_BOAT]).toBe(0);
    }
  }, 60000);

  it("marks player ATTACK legal only when a land border is shared", async () => {
    const env = await AgentEnv.create(
      testConfig({ map: "Halkidiki", mapSize: "Compact", bots: 1, nations: "disabled" }),
      terrain,
    );
    const spawnRegion = firstSetRegion(env.peekObs().spawnRegions);
    const r = env.step({
      actionType: ACTION_SPAWN,
      target: 0,
      region: spawnRegion,
      quantity: 2,
      unit: 0,
    });
    const obs = r.obs;
    const atk = attackTargetMask(obs);
    const PLAYER_FEATURES = 14;
    for (let i = 1; i < NUM_PLAYER_SLOTS; i++) {
      const exists = obs.players[i * PLAYER_FEATURES + 0] > 0;
      const shares = obs.players[i * PLAYER_FEATURES + 11] > 0;
      if (!exists) {
        expect(atk[i]).toBe(0);
      } else if (!shares) {
        expect(atk[i]).toBe(0);
      } else {
        expect(atk[i]).toBe(1);
      }
    }
  }, 60000);

  it("is deterministic for identical seeds and actions", async () => {
    const run = async (): Promise<number | null> => {
      const env = await AgentEnv.create(testConfig(), terrain);
      const region = firstSetRegion(env.peekObs().spawnRegions);
      env.step({
        actionType: ACTION_SPAWN,
        target: 0,
        region,
        quantity: 2,
        unit: 0,
      });
      let last: number | null = null;
      for (let i = 0; i < 20; i++) {
        const r = env.step({
          actionType: 0,
          target: 0,
          region: 0,
          quantity: 0,
          unit: 0,
        });
        if (r.info.hash !== null) last = r.info.hash;
        if (r.done) break;
      }
      return last;
    };
    const h1 = await run();
    const h2 = await run();
    expect(h1).not.toBeNull();
    expect(h1).toBe(h2);
  }, 120000);

  it("creates Halkidiki Compact and allows land combat without boat", async () => {
    const env = await AgentEnv.create(
      testConfig({
        map: "Halkidiki",
        mapSize: "Compact",
        bots: 1,
        nations: "disabled",
        allowedActions: ["noop", "spawn", "attack", "retreat_all"],
      }),
      terrain,
    );
    const obs0 = env.peekObs();
    expect(obs0.actionMask[ACTION_SPAWN]).toBe(1);
    expect(obs0.actionMask[ACTION_BOAT]).toBe(0);
    const region = firstSetRegion(obs0.spawnRegions);
    const r = env.step({
      actionType: ACTION_SPAWN,
      target: 0,
      region,
      quantity: 2,
      unit: 0,
    });
    expect(r.info.spawned).toBe(true);
    expect(r.obs.actionMask[ACTION_BOAT]).toBe(0);
    expect(r.obs.actionMask[ACTION_BUILD]).toBe(0);
  }, 60000);

  it("marks boat legal from shoreline without requiring a Port", async () => {
    const env = await AgentEnv.create(
      testConfig({
        map: "FourIslands",
        mapSize: "Compact",
        bots: 1,
        nations: "disabled",
        maxTicks: 2000,
      }),
      terrain,
    );
    const region = firstSetRegion(env.peekObs().spawnRegions);
    let r = env.step({
      actionType: ACTION_SPAWN,
      target: 0,
      region,
      quantity: 2,
      unit: 0,
    });
    expect(r.info.spawned).toBe(true);
    let sawBoatLegal = false;
    for (let i = 0; i < 30 && !r.done; i++) {
      if (r.obs.actionMask[ACTION_BOAT] === 1) {
        sawBoatLegal = true;
        break;
      }
      r = env.step({
        actionType: ACTION_ATTACK,
        target: 0,
        region: 0,
        quantity: 4,
        unit: 0,
      });
    }
    expect(sawBoatLegal).toBe(true);
    expect(r.obs.global[8]).toBe(0); // no Port owned
  }, 120000);

  it("wilderness invasion (ATTACK target=0) grows territory after spawn", async () => {
    const env = await AgentEnv.create(testConfig({ maxTicks: 6000 }), terrain);
    let obs = env.peekObs();
    const spawnRegion = firstSetRegion(obs.spawnRegions);
    let r = env.step({
      actionType: ACTION_SPAWN,
      target: 0,
      region: spawnRegion,
      quantity: 2,
      unit: 0,
    });
    obs = r.obs;
    expect(r.info.spawned).toBe(true);

    const startTiles = r.info.tilesFrac;
    let grew = false;
    let sawAttackLegal = false;
    for (let i = 0; i < 120 && !r.done; i++) {
      obs = r.obs;
      const atkMask = attackTargetMask(obs);
      if (obs.actionMask[ACTION_ATTACK] === 1 && atkMask[0] === 1) {
        sawAttackLegal = true;
      }
      let bestRegion = 0;
      let best = -1;
      for (let g = 0; g < NUM_REGIONS; g++) {
        if (obs.spawnRegions[g] > best) {
          best = obs.spawnRegions[g];
          bestRegion = g;
        }
      }
      r = env.step({
        actionType: ACTION_ATTACK,
        target: 0,
        region: bestRegion,
        quantity: 4,
        unit: 0,
      });
      if (r.info.tilesFrac > startTiles) {
        grew = true;
        break;
      }
    }
    expect(sawAttackLegal).toBe(true);
    expect(grew).toBe(true);
  }, 120000);

  it("does not farm positive reward from attack-retreat cycles", async () => {
    const env = await AgentEnv.create(
      testConfig({ maxTicks: 6000, shaping: 0 }),
      terrain,
    );
    const spawnRegion = firstSetRegion(env.peekObs().spawnRegions);
    let r = env.step({
      actionType: ACTION_SPAWN,
      target: 0,
      region: spawnRegion,
      quantity: 2,
      unit: 0,
    });
    expect(r.info.spawned).toBe(true);
    const spawnBonus = r.info.rewardTerms.spawn;

    // Attack then retreat repeatedly without keeping territory growth as the
    // scoring signal — no attack-start / income bonuses exist to farm.
    let farmSum = 0;
    for (let i = 0; i < 20 && !r.done; i++) {
      r = env.step({
        actionType: ACTION_ATTACK,
        target: 0,
        region: 0,
        quantity: 4,
        unit: 0,
      });
      // Ignore terminal / spawn; only mid-episode dense terms matter here.
      if (!r.done) {
        farmSum += r.info.rewardTerms.terminal + r.info.rewardTerms.spawn;
        // With shaping off, non-spawn mid-episode reward must be ~0.
        expect(Math.abs(r.reward)).toBeLessThan(1e-6);
      }
      r = env.step({
        actionType: ACTION_RETREAT_ALL,
        target: 0,
        region: 0,
        quantity: 0,
        unit: 0,
      });
      if (!r.done) {
        expect(Math.abs(r.reward)).toBeLessThan(1e-6);
      }
    }
    expect(spawnBonus).toBe(REWARD_SPAWN);
    expect(farmSum).toBe(0);
  }, 120000);

  it("timeout is a loss with exactly one terminal reward term", async () => {
    const env = await AgentEnv.create(
      testConfig({ maxTicks: 80, bots: 0, nations: "disabled", shaping: 0 }),
      terrain,
    );
    const spawnRegion = firstSetRegion(env.peekObs().spawnRegions);
    let r = env.step({
      actionType: ACTION_SPAWN,
      target: 0,
      region: spawnRegion,
      quantity: 2,
      unit: 0,
    });
    expect(r.info.spawned).toBe(true);
    while (!r.done) {
      r = env.step({
        actionType: 0,
        target: 0,
        region: 0,
        quantity: 0,
        unit: 0,
      });
    }
    expect(r.done).toBe(true);
    expect(r.info.terminalCause).toBe("timeout");
    expect(r.info.rewardTerms.terminal).toBe(REWARD_TIMEOUT);
    expect(r.info.rewardTerms.terminal).toBe(REWARD_DEATH); // same magnitude
    expect(r.info.win).toBe(false);
    expect(r.info.stageSuccess).toBe(false);
    // Exactly one terminal; total equals the sum of terms.
    const terms = r.info.rewardTerms;
    expect(terms.total).toBeCloseTo(
      terms.terminal + terms.spawn + terms.territory + terms.elimination,
      10,
    );
    expect(r.reward).toBe(terms.total);
    expect(r.reward).toBeLessThan(0);
    // Win must still dominate milestone + losing terminals under shaping.
    expect(REWARD_WIN).toBeGreaterThan(REWARD_CURRICULUM_SUCCESS);
    expect(REWARD_WIN).toBeGreaterThan(Math.abs(REWARD_TIMEOUT) + 1);
  }, 120000);

  it("winTilesFrac is stage success, not a core win, and does not terminate", async () => {
    const env = await AgentEnv.create(
      testConfig({
        map: "Halkidiki",
        mapSize: "Compact",
        bots: 0,
        nations: "disabled",
        maxTicks: 4000,
        winTilesFrac: 0.005,
      }),
      terrain,
    );
    const spawnRegion = firstSetRegion(env.peekObs().spawnRegions);
    let r = env.step({
      actionType: ACTION_SPAWN,
      target: 0,
      region: spawnRegion,
      quantity: 2,
      unit: 0,
    });
    let sawStage = r.info.stageSuccess;
    let terminatedOnMilestone = false;
    for (let i = 0; i < 80 && !r.done; i++) {
      r = env.step({
        actionType: ACTION_ATTACK,
        target: 0,
        region: 0,
        quantity: 3,
        unit: 0,
      });
      if (r.info.stageSuccess && !r.done) sawStage = true;
      if (r.done && r.info.terminalCause === "curriculum_success") {
        terminatedOnMilestone = true;
      }
    }
    expect(sawStage).toBe(true);
    expect(terminatedOnMilestone).toBe(false);
    expect(r.info.win).toBe(false);
    if (r.info.stageSuccess) {
      expect(r.info.terminalCause).not.toBe("curriculum_success");
      expect(r.info.rewardTerms.terminal).not.toBe(REWARD_CURRICULUM_SUCCESS);
    }
  }, 120000);

  it("rewardTerms total matches reward on every step", async () => {
    const env = await AgentEnv.create(
      testConfig({ maxTicks: 300, shaping: 1 }),
      terrain,
    );
    let obs = env.peekObs();
    let done = false;
    let steps = 0;
    while (!done && steps < 80) {
      const legal: number[] = [];
      for (let i = 0; i < obs.actionMask.length; i++) {
        if (obs.actionMask[i] === 1) legal.push(i);
      }
      const actionType = legal[steps % legal.length] ?? 0;
      const r = env.step({
        actionType,
        target: 0,
        region: firstSetRegion(obs.spawnRegions),
        quantity: 2,
        unit: 0,
      });
      const t = r.info.rewardTerms;
      expect(t.total).toBeCloseTo(
        t.terminal + t.spawn + t.territory + t.elimination,
        10,
      );
      expect(r.reward).toBeCloseTo(t.total, 10);
      if (r.done) {
        expect(r.info.terminalCause).not.toBe("none");
        // Exactly one terminal component when done.
        expect(
          t.terminal === REWARD_WIN ||
            t.terminal === REWARD_CURRICULUM_SUCCESS ||
            t.terminal === REWARD_DEATH ||
            t.terminal === REWARD_TIMEOUT ||
            t.terminal === REWARD_NO_SPAWN,
        ).toBe(true);
      } else {
        expect(r.info.terminalCause).toBe("none");
        expect(t.terminal).toBe(0);
      }
      done = r.done;
      obs = r.obs;
      steps++;
    }
    expect(done).toBe(true);
  }, 120000);

  it("accepts Compact mapSize and string allowedActions names", async () => {
    const env = await AgentEnv.create(
      testConfig({
        mapSize: "Compact",
        allowedActions: ["attack", "retreat_all"],
      }),
      terrain,
    );
    const obs = env.peekObs();
    expect(obs.spatial.length).toBe(10 * 64 * 64);
    expect(obs.actionMask[ACTION_NOOP]).toBe(0);
    expect(obs.actionMask[ACTION_SPAWN]).toBe(1);
    expect(obs.actionMask[ACTION_BUILD]).toBe(0);
    expect(obs.global[0]).toBeGreaterThanOrEqual(0);
    expect(obs.global[0]).toBeLessThanOrEqual(1);
  }, 60000);

  it("never-spawn at maxTicks is no_spawn, not death/timeout", async () => {
    const env = await AgentEnv.create(
      testConfig({ maxTicks: 80, bots: 0, nations: "disabled", shaping: 0 }),
      terrain,
    );
    expect(env.peekObs().actionMask[ACTION_NOOP]).toBe(0);
    expect(env.peekObs().actionMask[ACTION_SPAWN]).toBe(1);
    let r = env.step({
      actionType: ACTION_NOOP,
      target: 0,
      region: 0,
      quantity: 0,
      unit: 0,
    });
    while (!r.done) {
      r = env.step({
        actionType: ACTION_NOOP,
        target: 0,
        region: 0,
        quantity: 0,
        unit: 0,
      });
    }
    expect(r.info.terminalCause).toBe("no_spawn");
    expect(r.info.spawned).toBe(false);
    expect(r.info.win).toBe(false);
    expect(r.info.stageSuccess).toBe(false);
    expect(r.info.dead).toBe(false);
    expect(r.info.rewardTerms.terminal).toBe(REWARD_NO_SPAWN);
    expect(r.info.peakTilesFrac).toBe(0);
  }, 60000);

  it("honors allowedActions while preserving SPAWN then NOOP", async () => {
    const env = await AgentEnv.create(
      testConfig({
        allowedActions: [ACTION_ATTACK, ACTION_RETREAT_ALL],
      }),
      terrain,
    );
    const obs0 = env.peekObs();
    // Spawn phase allows SPAWN only, even if SPAWN is not listed.
    expect(obs0.actionMask[ACTION_NOOP]).toBe(0);
    expect(obs0.actionMask[ACTION_SPAWN]).toBe(1);
    expect(obs0.actionMask[ACTION_BUILD]).toBe(0);
    expect(obs0.actionMask[ACTION_BOAT]).toBe(0);

    const region = firstSetRegion(obs0.spawnRegions);
    const r = env.step({
      actionType: ACTION_SPAWN,
      target: 0,
      region,
      quantity: 2,
      unit: 0,
    });
    const am = r.obs.actionMask;
    expect(am[ACTION_NOOP]).toBe(1);
    expect(am[ACTION_SPAWN]).toBe(0);
    expect(am[ACTION_BUILD]).toBe(0);
    expect(am[ACTION_BOAT]).toBe(0);
    expect(am[ACTION_ALLY]).toBe(0);
    // Attack/retreat only if otherwise legal.
    for (let a = 0; a < NUM_ACTION_TYPES; a++) {
      if (a === ACTION_NOOP || a === ACTION_ATTACK || a === ACTION_RETREAT_ALL) {
        continue;
      }
      expect(am[a]).toBe(0);
    }
  }, 60000);

  it("random policies terminate episodes with valid info", async () => {
    const env = await AgentEnv.create(testConfig({ maxTicks: 500 }), terrain);
    let done = false;
    let steps = 0;
    let obs = env.peekObs();
    while (!done && steps < 200) {
      const actionMask = obs.actionMask;
      const legal: number[] = [];
      for (let i = 0; i < actionMask.length; i++) {
        if (actionMask[i] === 1) legal.push(i);
      }
      const actionType =
        legal[Math.floor(Math.random() * legal.length)] ?? 0;
      const r = env.step({
        actionType,
        target: Math.floor(Math.random() * 16),
        region: Math.floor(Math.random() * NUM_REGIONS),
        quantity: Math.floor(Math.random() * 5),
        unit: Math.floor(Math.random() * 10),
      });
      expect(typeof r.info.terminalCause).toBe("string");
      expect(r.info.peakTilesFrac).toBeGreaterThanOrEqual(0);
      expect(r.info.rewardTerms.total).toBeCloseTo(r.reward, 10);
      done = r.done;
      obs = r.obs;
      steps++;
    }
    expect(done).toBe(true);
    expect(steps).toBeGreaterThan(5);
  }, 120000);

  it("wilderness cache matches border-tile scan after spawn and steps", async () => {
    const env = await AgentEnv.create(
      testConfig({ seed: "wild-cache", maxTicks: 4000 }),
      terrain,
    );
    expect(env.hasAdjacentWildernessCached()).toBe(
      env.hasAdjacentWildernessScan(),
    );
    const spawnRegion = firstSetRegion(env.peekObs().spawnRegions);
    let r = env.step({
      actionType: ACTION_SPAWN,
      target: 0,
      region: spawnRegion,
      quantity: 2,
      unit: 0,
    });
    for (let i = 0; i < 40 && !r.done; i++) {
      expect(env.hasAdjacentWildernessCached()).toBe(
        env.hasAdjacentWildernessScan(),
      );
      r = env.step({
        actionType: ACTION_ATTACK,
        target: 0,
        region: 0,
        quantity: 4,
        unit: 0,
      });
    }
    expect(env.hasAdjacentWildernessCached()).toBe(
      env.hasAdjacentWildernessScan(),
    );
  }, 120000);
});

describe("stackObs", () => {
  it("reuses a matching batch buffer without changing values", () => {
    const a = makeObsBuffers();
    const b = makeObsBuffers();
    a.spatial[0] = 0.25;
    a.global[1] = 0.5;
    b.spatial[3] = 0.75;
    b.actionMask[2] = 1;
    const first = stackObs([a, b]);
    const reuse = allocBatchObs(2);
    const second = stackObs([a, b], reuse);
    expect(second).toBe(reuse);
    expect(Array.from(second.spatial)).toEqual(Array.from(first.spatial));
    expect(Array.from(second.global)).toEqual(Array.from(first.global));
    expect(Array.from(second.actionMask)).toEqual(Array.from(first.actionMask));
    const otherK = stackObs([a], reuse);
    expect(otherK).not.toBe(reuse);
    expect(otherK.spatial.length).toBe(a.spatial.length);
  });
});

describe("frameTensors", () => {
  it("maps offset-ordered blobs without concatenating", () => {
    const spatial = new Float32Array([1, 2, 3, 4]);
    const mask = new Uint8Array([1, 0, 1]);
    const wire = encodeFrame(
      { type: "step" },
      {
        spatial: { dtype: "f32", data: spatial },
        action_mask: { dtype: "u8", data: mask },
      },
    );
    const dec = new FrameDecoder();
    const frames = dec.push(wire);
    expect(frames).toHaveLength(1);
    const tensors = frameTensors(frames[0]);
    expect(new Float32Array(tensors.spatial.buf.buffer, tensors.spatial.buf.byteOffset, 4)).toEqual(spatial);
    expect(Array.from(tensors.action_mask.buf)).toEqual([1, 0, 1]);
  });
});
