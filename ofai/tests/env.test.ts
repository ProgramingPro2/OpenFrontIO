/**
 * Harness smoke tests: env creation, spawn flow, stepping, and determinism.
 * Run with: npx vitest run ofai/tests/env.test.ts
 */
import path from "node:path";
import { describe, expect, it } from "vitest";
import { NodeGameMapLoader } from "../../tests/perf/fullgame/NodeGameMapLoader";
import { AgentEnv } from "../env/AgentEnv";
import {
  ACTION_ATTACK,
  ACTION_SPAWN,
  EnvConfig,
  NUM_REGIONS,
  REWARD_ATTACK_START,
  REWARD_DEATH,
  SPATIAL_CHANNELS,
  SPATIAL_SIZE,
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

describe("AgentEnv", () => {
  it("creates an env and produces a valid initial observation", async () => {
    const env = await AgentEnv.create(testConfig(), terrain);
    const obs = env.peekObs();
    expect(obs.spatial.length).toBe(SPATIAL_CHANNELS * SPATIAL_SIZE * SPATIAL_SIZE);
    expect(obs.players.length).toBe(16 * 14);
    expect(obs.global.length).toBe(10);
    // Spawn phase: only noop + spawn are legal.
    expect(obs.actionMask[0]).toBe(1);
    expect(obs.actionMask[1]).toBe(1);
    expect(obs.actionMask[2]).toBe(0);
    // Land plane has content.
    let landSum = 0;
    for (let i = 0; i < SPATIAL_SIZE * SPATIAL_SIZE; i++) {
      landSum += obs.spatial[i];
    }
    expect(landSum).toBeGreaterThan(0);
    // Some spawn region must be available.
    expect(obs.spawnRegions.some((v) => v === 1)).toBe(true);
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
    expect(result.obs.global[1]).toBe(0); // no longer in spawn phase
    expect(result.obs.actionMask[2] === 1 || result.obs.actionMask[4] === 1).toBe(
      true,
    );
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

  it("wilderness invasion (ATTACK target=0) grows territory after spawn", async () => {
    // Regression test for the expansion bug: ATTACK was previously only able
    // to target other players, so the agent could never invade uninhabited
    // land and myTilesFrac stayed 0 forever. target==0 is reinterpreted as
    // TerraNullius (wilderness) and conquers from our border.
    const env = await AgentEnv.create(testConfig({ maxTicks: 6000 }), terrain);
    let obs = env.peekObs();
    // Spawn in a region with unowned land.
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

    // Repeatedly attack wilderness in the region with the most unowned land,
    // using the largest troop fraction, until territory grows.
    const startTiles = r.info.tilesFrac;
    let grew = false;
    let sawAttackLegal = false;
    for (let i = 0; i < 120 && !r.done; i++) {
      obs = r.obs;
      if (obs.actionMask[ACTION_ATTACK] === 1 && obs.targetMask[0] === 1) {
        sawAttackLegal = true;
      }
      // Pick the region with the most spawnable (unowned) land.
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
        target: 0, // wilderness
        region: bestRegion,
        quantity: 4, // 80% of troops
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

  it("pays attack-start once and shaping on growth; no-op does not farm", async () => {
    const env = await AgentEnv.create(
      testConfig({ maxTicks: 6000, shaping: 5 }),
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
    const startTiles = r.info.tilesFrac;

    let startPaid = false;
    let grewWithShaping = false;
    for (let i = 0; i < 120 && !r.done; i++) {
      let bestRegion = 0;
      let best = -1;
      for (let g = 0; g < NUM_REGIONS; g++) {
        if (r.obs.spawnRegions[g] > best) {
          best = r.obs.spawnRegions[g];
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
      if (r.reward >= REWARD_ATTACK_START - 1e-6) startPaid = true;
      if (r.info.tilesFrac > startTiles && r.reward > REWARD_ATTACK_START) {
        grewWithShaping = true;
        break;
      }
    }
    expect(startPaid).toBe(true);
    expect(grewWithShaping).toBe(true);

    const noop = env.step({
      actionType: 0,
      target: 0,
      region: 0,
      quantity: 0,
      unit: 0,
    });
    // Same attack still running: no second start bonus. Incoming/income/shaping
    // can move the number, but it must stay below a fresh attack-start.
    expect(noop.reward).toBeLessThan(REWARD_ATTACK_START);
  }, 120000);

  it("timeout without expansion is death-sized, not -0.25", async () => {
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
    // No-expand timeout uses REWARD_DEATH, not REWARD_TIMEOUT_ALIVE. Last-step
    // incoming/income can nudge it, but it must stay near death, not -0.25.
    expect(r.done).toBe(true);
    expect(r.reward).toBeLessThan(-0.5);
    expect(r.reward).toBeGreaterThan(REWARD_DEATH - 0.2);
  }, 120000);

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
      done = r.done;
      obs = r.obs;
      steps++;
    }
    expect(done).toBe(true);
    expect(steps).toBeGreaterThan(5);
  }, 120000);
});
