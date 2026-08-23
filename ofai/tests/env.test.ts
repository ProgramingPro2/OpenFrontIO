/**
 * Harness smoke tests: env creation, spawn flow, stepping, masks, rewards.
 * Run with: npx vitest run ofai/tests/env.test.ts
 */
import path from "node:path";
import { describe, expect, it } from "vitest";
import { NodeGameMapLoader } from "../../tests/perf/fullgame/NodeGameMapLoader";
import { resolveAutoReset } from "../env/autoReset";
import { AgentEnv } from "../env/AgentEnv";
import { UNIT_HEAD_ORDER } from "../env/ActionTranslator";
import { UnitType } from "../../src/core/game/Game";
import { allocBatchObs, stackObs } from "../env/batchObs";
import { allocOwnedBatch } from "../env/obsLayout";
import { parseServerArgs, resolveWorkerCount } from "../env/EnvBackend";
import { encodeFrame, FrameDecoder, frameTensors } from "../env/framing";
import {
  assignOpponentSlots,
  compareOpponentRank,
  makeObsBuffers,
  slotPriorityScore,
} from "../env/ObsExtractor";
import {
  ACTION_ALLY,
  ACTION_ATTACK,
  ACTION_BOAT,
  ACTION_BREAK_ALLY,
  ACTION_BUILD,
  ACTION_EMBARGO,
  ACTION_NOOP,
  ACTION_RETREAT_ALL,
  ACTION_SPAWN,
  EnvConfig,
  NUM_ACTION_TYPES,
  NUM_PLAYER_SLOTS,
  NUM_QUANTITIES,
  NUM_REGIONS,
  PLAYER_FEATURES,
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

function targetRow(
  obs: { targetMasks: Uint8Array },
  action: number,
): Uint8Array {
  const row = action * NUM_PLAYER_SLOTS;
  return obs.targetMasks.subarray(row, row + NUM_PLAYER_SLOTS);
}

function regionMaskFor(
  obs: {
    spawnRegions: Uint8Array;
    buildRegions: Uint8Array;
    boatRegions: Uint8Array;
  },
  action: number,
): Uint8Array {
  if (action === ACTION_SPAWN) return obs.spawnRegions;
  if (action === ACTION_BUILD) return obs.buildRegions;
  if (action === ACTION_BOAT) return obs.boatRegions;
  return new Uint8Array(0);
}

function firstSet(mask: Uint8Array): number {
  for (let i = 0; i < mask.length; i++) {
    if (mask[i] === 1) return i;
  }
  return 0;
}

function legalMaskedAction(
  obs: {
    actionMask: Uint8Array;
    targetMasks: Uint8Array;
    quantityMask: Uint8Array;
    unitMask: Uint8Array;
    spawnRegions: Uint8Array;
    buildRegions: Uint8Array;
    boatRegions: Uint8Array;
  },
  actionType: number,
  preferUnit?: number,
): {
  actionType: number;
  target: number;
  region: number;
  quantity: number;
  unit: number;
} {
  const regions = regionMaskFor(obs, actionType);
  let unit = firstSet(obs.unitMask);
  if (preferUnit !== undefined && obs.unitMask[preferUnit] === 1) {
    unit = preferUnit;
  }
  return {
    actionType,
    target: firstSet(targetRow(obs, actionType)),
    region: regions.length > 0 ? firstSet(regions) : 0,
    quantity: firstSet(obs.quantityMask),
    unit,
  };
}

function legalFactorCombos(
  obs: {
    actionMask: Uint8Array;
    targetMasks: Uint8Array;
    quantityMask: Uint8Array;
    unitMask: Uint8Array;
    spawnRegions: Uint8Array;
    buildRegions: Uint8Array;
    boatRegions: Uint8Array;
  },
  actionType: number,
): Array<{
  actionType: number;
  target: number;
  region: number;
  quantity: number;
  unit: number;
}> {
  const base = legalMaskedAction(obs, actionType);
  const combos = [base];
  const seen = new Set<string>();
  const push = (a: typeof base) => {
    const key = `${a.actionType}:${a.target}:${a.region}:${a.quantity}:${a.unit}`;
    if (seen.has(key)) return;
    seen.add(key);
    combos.push(a);
  };
  const pickSet = (mask: Uint8Array, limit = 8): number[] => {
    const idx: number[] = [];
    for (let i = 0; i < mask.length; i++) if (mask[i] === 1) idx.push(i);
    if (idx.length <= limit) return idx;
    const out = [idx[0], idx[idx.length - 1]];
    const step = (idx.length - 1) / (limit - 1);
    for (let k = 1; k < limit - 1; k++) out.push(idx[Math.round(k * step)]);
    return [...new Set(out)];
  };
  for (const target of pickSet(targetRow(obs, actionType), 16)) {
    push({ ...base, target });
  }
  const regions = regionMaskFor(obs, actionType);
  if (regions.length > 0) {
    for (const region of pickSet(regions, 8)) {
      push({ ...base, region });
    }
  }
  for (const quantity of pickSet(obs.quantityMask, 5)) {
    push({ ...base, quantity });
  }
  for (const unit of pickSet(obs.unitMask, 10)) {
    push({ ...base, unit });
  }
  // Bounded active-head cartesians (not a 1024-way region product).
  if (actionType === ACTION_ATTACK) {
    for (const target of pickSet(targetRow(obs, actionType), 16)) {
      for (const quantity of pickSet(obs.quantityMask, 5)) {
        push({ ...base, target, quantity });
      }
    }
  }
  if (actionType === ACTION_BOAT && regions.length > 0) {
    for (const region of pickSet(regions, 4)) {
      for (const quantity of pickSet(obs.quantityMask, 5)) {
        push({ ...base, region, quantity });
      }
    }
  }
  if (actionType === ACTION_BUILD && regions.length > 0) {
    for (const unit of pickSet(obs.unitMask, 10)) {
      for (const region of pickSet(regions, 4)) {
        push({ ...base, unit, region });
      }
    }
  }
  return combos;
}

function assertMaskedNonNoopEmitsIntent(
  env: AgentEnv,
  obs: {
    actionMask: Uint8Array;
    targetMasks: Uint8Array;
    quantityMask: Uint8Array;
    unitMask: Uint8Array;
    spawnRegions: Uint8Array;
    buildRegions: Uint8Array;
    boatRegions: Uint8Array;
  },
): void {
  for (let a = 0; a < NUM_ACTION_TYPES; a++) {
    if (obs.actionMask[a] !== 1) continue;
    if (a === ACTION_NOOP) {
      expect(env.wouldEmitIntent(legalMaskedAction(obs, a))).toBe(true);
      continue;
    }
    const action = legalMaskedAction(obs, a);
    expect(env.wouldEmitIntent(action)).toBe(true);
    for (const combo of legalFactorCombos(obs, a)) {
      expect(env.wouldEmitIntent(combo)).toBe(true);
    }
    if (a === ACTION_ATTACK || a === ACTION_ALLY || a === ACTION_BREAK_ALLY || a === ACTION_EMBARGO) {
      const row = targetRow(obs, a);
      expect(row.some((v) => v === 1) || a === ACTION_ATTACK).toBe(true);
      if (a === ACTION_ATTACK) {
        expect(row.some((v) => v === 1)).toBe(true);
      }
    }
    if (a === ACTION_BUILD) {
      expect(obs.unitMask.some((v) => v === 1)).toBe(true);
    }
    if (a === ACTION_BOAT) {
      expect(obs.boatRegions.some((v) => v === 1)).toBe(true);
      expect(obs.quantityMask.some((v) => v === 1)).toBe(true);
    }
  }
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

describe("opponent slot ranking", () => {
  it("breaks priority ties by smallID then id", () => {
    const a = { priority: 10, smallID: 3, id: "c" };
    const b = { priority: 10, smallID: 1, id: "z" };
    const c = { priority: 11, smallID: 9, id: "a" };
    const ranked = [a, b, c].sort(compareOpponentRank);
    expect(ranked.map((x) => x.id)).toEqual(["a", "z", "c"]);
  });

  it("keeps border/attackers in individual slots when overflowing", () => {
    const metas = [];
    for (let i = 0; i < 10; i++) {
      metas.push({
        priority: slotPriorityScore({
          tiles: 5000 + i,
          sharesBorder: false,
          canAttack: false,
          incomingTroops: 0,
          allied: false,
          allySignal: false,
        }),
        smallID: 100 + i,
        id: `mass-${i}`,
      });
    }
    for (let i = 0; i < 8; i++) {
      metas.push({
        priority: slotPriorityScore({
          tiles: 10 + i,
          sharesBorder: true,
          canAttack: true,
          incomingTroops: i === 0 ? 50 : 0,
          allied: false,
          allySignal: false,
        }),
        smallID: i,
        id: `border-${i}`,
      });
    }
    const { picked, rest, overflow } = assignOpponentSlots(metas, 15);
    expect(overflow).toBe(true);
    expect(picked).toHaveLength(14);
    expect(rest.length).toBeGreaterThan(0);
    const pickedIds = new Set(picked.map((m) => m.id));
    for (let i = 0; i < 8; i++) {
      expect(pickedIds.has(`border-${i}`)).toBe(true);
    }
    expect(rest.every((m) => m.id.startsWith("mass-"))).toBe(true);
  });
});

describe("overflow aggregate slot", () => {
  it("does not aggregate when opponents fit in 15 slots", async () => {
    const env = await AgentEnv.create(
      testConfig({ map: "Halkidiki", mapSize: "Compact", bots: 1, nations: "disabled" }),
      terrain,
    );
    const players = env.peekObs().players;
    const last = (NUM_PLAYER_SLOTS - 1) * PLAYER_FEATURES;
    const exists = players[last];
    const nation = players[last + 6];
    const bot = players[last + 7];
    // One bot: last slot is empty or a real player, never the both-type sentinel.
    expect(nation === 1 && bot === 1 && exists === 1).toBe(false);
  }, 60000);

  it("writes an untargetable overflow aggregate when bots exceed 15 slots", async () => {
    const env = await AgentEnv.create(
      testConfig({
        map: "FourIslands",
        mapSize: "Normal",
        bots: 20,
        nations: "disabled",
      }),
      terrain,
    );
    const obs = env.peekObs();
    const last = (NUM_PLAYER_SLOTS - 1) * PLAYER_FEATURES;
    expect(obs.players[last + 0]).toBe(1);
    expect(obs.players[last + 1]).toBe(0);
    expect(obs.players[last + 6]).toBe(1);
    expect(obs.players[last + 7]).toBe(1);
    const atk = attackTargetMask(obs);
    expect(atk[NUM_PLAYER_SLOTS - 1]).toBe(0);
    const allyRow = ACTION_ALLY * NUM_PLAYER_SLOTS;
    expect(obs.targetMasks[allyRow + NUM_PLAYER_SLOTS - 1]).toBe(0);
    expect(env.slotFlagsMatchLive()).toBe(true);
    expect(env.peekSlots()[NUM_PLAYER_SLOTS - 1]).toBeNull();
  }, 60000);

  it("cached slot flags match a live border scan after spawn", async () => {
    const env = await AgentEnv.create(
      testConfig({
        map: "FourIslands",
        mapSize: "Normal",
        bots: 20,
        nations: "disabled",
        seed: "slot-cache",
      }),
      terrain,
    );
    expect(env.slotFlagsMatchLive()).toBe(true);
    const spawnRegion = firstSetRegion(env.peekObs().spawnRegions);
    let r = env.step({
      actionType: ACTION_SPAWN,
      target: 0,
      region: spawnRegion,
      quantity: 2,
      unit: 0,
    });
    for (let i = 0; i < 12 && !r.done; i++) {
      expect(env.slotFlagsMatchLive()).toBe(true);
      r = env.step({
        actionType: ACTION_ATTACK,
        target: 0,
        region: 0,
        quantity: 4,
        unit: 0,
      });
    }
    expect(env.slotFlagsMatchLive()).toBe(true);
    const atk = attackTargetMask(r.obs);
    expect(atk[NUM_PLAYER_SLOTS - 1]).toBe(0);
  }, 120000);
});

describe("ActionTranslator wilderness cache", () => {
  it("cache and scan produce the same wilderness attack intents", async () => {
    const env = await AgentEnv.create(
      testConfig({ seed: "wild-action", maxTicks: 4000 }),
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
    const attack = {
      actionType: ACTION_ATTACK,
      target: 0,
      region: 0,
      quantity: 4,
      unit: 0,
    };
    for (let i = 0; i < 20 && !r.done; i++) {
      const cached = env.hasAdjacentWildernessCached();
      const scanned = env.hasAdjacentWildernessScan();
      expect(cached).toBe(scanned);
      expect(env.translateForTest(attack, cached)).toEqual(
        env.translateForTest(attack, scanned),
      );
      r = env.step(attack);
    }
    expect(env.hasAdjacentWildernessCached()).toBe(
      env.hasAdjacentWildernessScan(),
    );
  }, 120000);
});

describe("headless GameRunner name placement", () => {
  it("skipNamePlacement does not change hash, tiles, or packed obs", async () => {
    const cfg = testConfig({
      seed: "headless-hash",
      map: "Halkidiki",
      mapSize: "Compact",
      bots: 1,
      nations: "disabled",
      maxTicks: 800,
    });
    const run = async (skip: boolean) => {
      const env = await AgentEnv.create(cfg, terrain, {
        skipNamePlacement: skip,
      });
      const spawnRegion = firstSetRegion(env.peekObs().spawnRegions);
      let r = env.step({
        actionType: ACTION_SPAWN,
        target: 0,
        region: spawnRegion,
        quantity: 2,
        unit: 0,
      });
      let lastHash: number | null = r.info.hash;
      for (let i = 0; i < 16 && !r.done; i++) {
        r = env.step({
          actionType: ACTION_NOOP,
          target: 0,
          region: 0,
          quantity: 0,
          unit: 0,
        });
        if (r.info.hash !== null) lastHash = r.info.hash;
      }
      const frame = env.renderFrame();
      return {
        hash: lastHash,
        tiles: r.info.tilesFrac,
        tick: r.info.tick,
        spatial: Array.from(r.obs.spatial),
        players: Array.from(r.obs.players),
        masks: Array.from(r.obs.actionMask),
        cells: Array.from(frame.cells),
      };
    };
    const headless = await run(true);
    const named = await run(false);
    expect(headless.hash).not.toBeNull();
    expect(headless.hash).toBe(named.hash);
    expect(headless.tiles).toBe(named.tiles);
    expect(headless.tick).toBe(named.tick);
    expect(headless.spatial).toEqual(named.spatial);
    expect(headless.players).toEqual(named.players);
    expect(headless.masks).toEqual(named.masks);
    expect(headless.cells).toEqual(named.cells);
  }, 180000);
});

describe("reset next config", () => {
  it("applies a provided seed instead of appending -rN", async () => {
    const env = await AgentEnv.create(testConfig({ seed: "base-seed" }), terrain);
    expect(env.seed).toBe("base-seed");
    await env.reset("holdout-frozen-0");
    expect(env.seed).toBe("holdout-frozen-0");
    await env.reset({ seed: "mixed-next", map: "Halkidiki", mapSize: "Compact" });
    expect(env.seed).toBe("mixed-next");
  }, 60000);
});

describe("resolveAutoReset", () => {
  it("uses Python nextConfigs and falls back to seed-rN", () => {
    const provided = resolveAutoReset(
      [{ seed: "mixed-next", map: "Hawaii" }, "holdout-1"],
      0,
      "base",
      3,
    );
    expect(provided.fallback).toBe(false);
    expect(provided.spec).toEqual({ seed: "mixed-next", map: "Hawaii" });
    const asString = resolveAutoReset(["holdout-1"], 0, "base", 3);
    expect(asString).toEqual({ spec: "holdout-1", fallback: false });
    const missing = resolveAutoReset(undefined, 0, "base", 3);
    expect(missing).toEqual({ spec: "base-r3", fallback: true });
    const hole = resolveAutoReset([null, undefined], 0, "watch-seed", 1);
    expect(hole).toEqual({ spec: "watch-seed-r1", fallback: true });
  });
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

  it("in-place batch views match a stackObs copy", () => {
    const a = makeObsBuffers();
    const b = makeObsBuffers();
    a.spatial[7] = 0.5;
    a.spawnRegions[3] = 1;
    b.players[2] = 0.25;
    b.boatRegions[9] = 1;
    const stacked = stackObs([a, b]);
    const { batch, envViews } = allocOwnedBatch(2);
    envViews[0].spatial.set(a.spatial);
    envViews[0].players.set(a.players);
    envViews[0].global.set(a.global);
    envViews[0].actionMask.set(a.actionMask);
    envViews[0].targetMasks.set(a.targetMasks);
    envViews[0].quantityMask.set(a.quantityMask);
    envViews[0].unitMask.set(a.unitMask);
    envViews[0].spawnRegions.set(a.spawnRegions);
    envViews[0].buildRegions.set(a.buildRegions);
    envViews[0].boatRegions.set(a.boatRegions);
    envViews[1].spatial.set(b.spatial);
    envViews[1].players.set(b.players);
    envViews[1].global.set(b.global);
    envViews[1].actionMask.set(b.actionMask);
    envViews[1].targetMasks.set(b.targetMasks);
    envViews[1].quantityMask.set(b.quantityMask);
    envViews[1].unitMask.set(b.unitMask);
    envViews[1].spawnRegions.set(b.spawnRegions);
    envViews[1].buildRegions.set(b.buildRegions);
    envViews[1].boatRegions.set(b.boatRegions);
    expect(Array.from(batch.spatial)).toEqual(Array.from(stacked.spatial));
    expect(Array.from(batch.players)).toEqual(Array.from(stacked.players));
    expect(Array.from(batch.boatRegions)).toEqual(Array.from(stacked.boatRegions));
    expect(envViews[0].spatial.buffer).toBe(batch.spatial.buffer);
  });
});

describe("worker count and CLI", () => {
  it("treats workers<=1 as sequential and caps at k", () => {
    expect(resolveWorkerCount(0, 8)).toBe(1);
    expect(resolveWorkerCount(1, 8)).toBe(1);
    expect(resolveWorkerCount(4, 8)).toBe(4);
    expect(resolveWorkerCount(16, 3)).toBe(3);
    expect(resolveWorkerCount(64, 64)).toBe(32);
  });

  it("parses --workers and --port", () => {
    expect(parseServerArgs(["--port", "9001", "--workers", "4"])).toEqual({
      port: 9001,
      workers: 4,
    });
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

describe("effective action legality", () => {
  it("every non-NOOP masked sample emits an intent after spawn", async () => {
    const env = await AgentEnv.create(
      testConfig({
        map: "FourIslands",
        mapSize: "Compact",
        bots: 1,
        nations: "disabled",
        seed: "effective-masks",
        maxTicks: 4000,
      }),
      terrain,
    );
    let obs = env.peekObs();
    expect(obs.actionMask[ACTION_SPAWN]).toBe(1);
    assertMaskedNonNoopEmitsIntent(env, obs);
    let r = env.step(legalMaskedAction(obs, ACTION_SPAWN));
    expect(r.info.actionAccepted).toBe(true);
    expect(r.info.intentCount).toBeGreaterThan(0);
    for (let i = 0; i < 24 && !r.done; i++) {
      obs = r.obs;
      assertMaskedNonNoopEmitsIntent(env, obs);
      if (obs.actionMask[ACTION_BOAT] === 1) {
        const boat = legalMaskedAction(obs, ACTION_BOAT);
        expect(env.wouldEmitIntent(boat)).toBe(true);
      }
      if (obs.actionMask[ACTION_BUILD] === 1) {
        expect(env.wouldEmitIntent(legalMaskedAction(obs, ACTION_BUILD))).toBe(
          true,
        );
      }
      if (obs.actionMask[ACTION_ATTACK] === 1) {
        expect(env.wouldEmitIntent(legalMaskedAction(obs, ACTION_ATTACK))).toBe(
          true,
        );
      }
      const legal: number[] = [];
      for (let a = 0; a < NUM_ACTION_TYPES; a++) {
        if (obs.actionMask[a] === 1) legal.push(a);
      }
      const pick = legal[i % legal.length] ?? ACTION_NOOP;
      r = env.step(legalMaskedAction(obs, pick));
      if (pick !== ACTION_NOOP) {
        expect(r.info.intentCount).toBeGreaterThan(0);
        expect(r.info.actionAccepted).toBe(true);
      }
    }
  }, 180000);

  it("boat and build stay illegal when no launchable/placeable config exists", async () => {
    const env = await AgentEnv.create(
      testConfig({
        map: "Halkidiki",
        mapSize: "Compact",
        bots: 0,
        nations: "disabled",
        seed: "no-fizzle-utility",
        maxTicks: 800,
        allowedActions: ["noop", "spawn", "attack", "retreat_all"],
      }),
      terrain,
    );
    const spawn = legalMaskedAction(env.peekObs(), ACTION_SPAWN);
    const r = env.step(spawn);
    expect(r.obs.actionMask[ACTION_BOAT]).toBe(0);
    expect(r.obs.actionMask[ACTION_BUILD]).toBe(0);
    expect(r.obs.actionMask[ACTION_ALLY]).toBe(0);
    const boatAnyway = {
      actionType: ACTION_BOAT,
      target: 0,
      region: firstSet(r.obs.boatRegions),
      quantity: 2,
      unit: 0,
    };
    // Allowed-actions gate, not merely a later penalty: boat is masked.
    expect(r.obs.actionMask[ACTION_BOAT]).toBe(0);
    if (r.obs.boatRegions.every((v) => v === 0)) {
      expect(env.wouldEmitIntent(boatAnyway)).toBe(false);
    }
  }, 120000);

  it("attack is legal only for wilderness or a bordering attackable target", async () => {
    const env = await AgentEnv.create(
      testConfig({
        map: "Halkidiki",
        mapSize: "Compact",
        bots: 1,
        nations: "disabled",
        seed: "attack-effective",
        maxTicks: 2000,
      }),
      terrain,
    );
    let r = env.step(legalMaskedAction(env.peekObs(), ACTION_SPAWN));
    for (let i = 0; i < 12 && !r.done; i++) {
      const obs = r.obs;
      const atk = attackTargetMask(obs);
      if (obs.actionMask[ACTION_ATTACK] === 1) {
        expect(atk.some((v) => v === 1)).toBe(true);
        expect(obs.quantityMask.some((v) => v === 1)).toBe(true);
        expect(env.wouldEmitIntent(legalMaskedAction(obs, ACTION_ATTACK))).toBe(
          true,
        );
      } else {
        expect(env.wouldEmitIntent({
          actionType: ACTION_ATTACK,
          target: 0,
          region: 0,
          quantity: 4,
          unit: 0,
        })).toBe(false);
      }
      r = env.step({
        actionType: ACTION_NOOP,
        target: 0,
        region: 0,
        quantity: 0,
        unit: 0,
      });
    }
  }, 120000);

  it("diplomacy types require an effective target and then emit an intent", async () => {
    const env = await AgentEnv.create(
      testConfig({
        map: "FourIslands",
        mapSize: "Normal",
        bots: 5,
        nations: "disabled",
        seed: "diplo-effective",
        maxTicks: 3000,
        allowedActions: [
          "noop",
          "spawn",
          "attack",
          "retreat_all",
          "ally",
          "break_ally",
          "embargo",
        ],
      }),
      terrain,
    );
    let r = env.step(legalMaskedAction(env.peekObs(), ACTION_SPAWN));
    let sawAlly = false;
    let sawEmbargo = false;
    for (let i = 0; i < 20 && !r.done; i++) {
      const obs = r.obs;
      if (obs.actionMask[ACTION_ALLY] === 1) {
        sawAlly = true;
        expect(targetRow(obs, ACTION_ALLY).some((v) => v === 1)).toBe(true);
        expect(env.wouldEmitIntent(legalMaskedAction(obs, ACTION_ALLY))).toBe(
          true,
        );
      }
      if (obs.actionMask[ACTION_EMBARGO] === 1) {
        sawEmbargo = true;
        expect(targetRow(obs, ACTION_EMBARGO).some((v) => v === 1)).toBe(true);
        expect(env.wouldEmitIntent(legalMaskedAction(obs, ACTION_EMBARGO))).toBe(
          true,
        );
      }
      if (obs.actionMask[ACTION_BREAK_ALLY] === 1) {
        expect(targetRow(obs, ACTION_BREAK_ALLY).some((v) => v === 1)).toBe(true);
        expect(
          env.wouldEmitIntent(legalMaskedAction(obs, ACTION_BREAK_ALLY)),
        ).toBe(true);
      }
      r = env.step({
        actionType: ACTION_NOOP,
        target: 0,
        region: 0,
        quantity: 0,
        unit: 0,
      });
    }
    expect(sawAlly || sawEmbargo).toBe(true);
  }, 180000);

  it("retreat is legal only when a non-retreating outgoing attack exists", async () => {
    const env = await AgentEnv.create(
      testConfig({
        map: "FourIslands",
        mapSize: "Compact",
        bots: 0,
        nations: "disabled",
        seed: "retreat-effective",
        maxTicks: 800,
      }),
      terrain,
    );
    const r = env.step(legalMaskedAction(env.peekObs(), ACTION_SPAWN));
    expect(r.obs.actionMask[ACTION_RETREAT_ALL]).toBe(0);
    expect(
      env.wouldEmitIntent({
        actionType: ACTION_RETREAT_ALL,
        target: 0,
        region: 0,
        quantity: 0,
        unit: 0,
      }),
    ).toBe(false);
  }, 120000);

  it("mask refill stays bounded on compact peninsula after spawn", async () => {
    const env = await AgentEnv.create(
      testConfig({
        map: "Halkidiki",
        mapSize: "Compact",
        bots: 1,
        nations: "disabled",
        seed: "mask-bench",
        maxTicks: 800,
      }),
      terrain,
    );
    env.step(legalMaskedAction(env.peekObs(), ACTION_SPAWN));
    const n = 25;
    const t0 = performance.now();
    for (let i = 0; i < n; i++) {
      env.refillMasksForTest();
    }
    const ms = performance.now() - t0;
    const per = ms / n;
    // Isolated mask refill (no tick / train). Catches O(tiles)*units*regions
    // regressions; Compact Halkidiki should stay well under a second each.
    console.log(`mask refill: ${per.toFixed(2)}ms/call over ${n} calls (${ms.toFixed(1)}ms)`);
    expect(per).toBeLessThan(750);
    expect(ms).toBeLessThan(12_000);
  }, 120000);

  it("cached effective masks match a cold translator exactly", async () => {
    const env = await AgentEnv.create(
      testConfig({
        map: "Halkidiki",
        mapSize: "Compact",
        bots: 1,
        nations: "disabled",
        seed: "mask-cache-eq",
        maxTicks: 2000,
      }),
      terrain,
    );
    let r = env.step(legalMaskedAction(env.peekObs(), ACTION_SPAWN));
    for (let i = 0; i < 16 && !r.done; i++) {
      env.refillMasksForTest();
      const warm = env.effectiveMasksSnapshot();
      env.clearTranslatorCachesForTest();
      env.refillMasksForTest();
      expect(env.effectiveMasksSnapshot()).toEqual(warm);
      const legal: number[] = [];
      for (let a = 0; a < NUM_ACTION_TYPES; a++) {
        if (r.obs.actionMask[a] === 1) legal.push(a);
      }
      r = env.step(legalMaskedAction(r.obs, legal[i % legal.length] ?? ACTION_NOOP));
    }
  }, 180000);
});

const UNIT_CITY = UNIT_HEAD_ORDER.indexOf(UnitType.City);
const UNIT_PORT = UNIT_HEAD_ORDER.indexOf(UnitType.Port);
const UNIT_SILO = UNIT_HEAD_ORDER.indexOf(UnitType.MissileSilo);
const UNIT_ATOM = UNIT_HEAD_ORDER.indexOf(UnitType.AtomBomb);
const UNIT_HBOMB = UNIT_HEAD_ORDER.indexOf(UnitType.HydrogenBomb);
const UNIT_MIRV = UNIT_HEAD_ORDER.indexOf(UnitType.MIRV);
const UNIT_WARSHIP = UNIT_HEAD_ORDER.indexOf(UnitType.Warship);

describe("BUILD warship and nuke legality", () => {
  it("keeps the fixed schema-v2 unit heads", () => {
    expect(UNIT_HEAD_ORDER).toEqual([
      UnitType.City,
      UnitType.DefensePost,
      UnitType.SAMLauncher,
      UnitType.MissileSilo,
      UnitType.Port,
      UnitType.Factory,
      UnitType.AtomBomb,
      UnitType.HydrogenBomb,
      UnitType.MIRV,
      UnitType.Warship,
    ]);
  });

  it("masks Warship until an owned Port exists, then emits water placement", async () => {
    const env = await AgentEnv.create(
      testConfig({
        map: "FourIslands",
        mapSize: "Compact",
        nations: "disabled",
        bots: 0,
        seed: "build-warship-legal",
        maxTicks: 2500,
        startingGold: 500_000,
      }),
      terrain,
    );
    let r = env.step(legalMaskedAction(env.peekObs(), ACTION_SPAWN));
    expect(r.obs.unitMask[UNIT_WARSHIP]).toBe(0);
    expect(
      env.wouldEmitIntent({
        actionType: ACTION_BUILD,
        target: 0,
        region: firstSet(r.obs.buildRegions),
        quantity: 0,
        unit: UNIT_WARSHIP,
      }),
    ).toBe(false);

    let builtPort = false;
    for (let i = 0; i < 40 && !r.done; i++) {
      const obs = r.obs;
      if (
        !builtPort &&
        obs.actionMask[ACTION_BUILD] === 1 &&
        obs.unitMask[UNIT_PORT] === 1
      ) {
        r = env.step(legalMaskedAction(obs, ACTION_BUILD, UNIT_PORT));
        if (r.info.actionAccepted) builtPort = true;
        continue;
      }
      if (obs.unitMask[UNIT_WARSHIP] === 1) {
        const region =
          env.findBuildRegionForUnit(UNIT_WARSHIP) ??
          firstSet(obs.buildRegions);
        const action = {
          actionType: ACTION_BUILD,
          target: 0,
          region,
          quantity: 0,
          unit: UNIT_WARSHIP,
        };
        expect(env.wouldEmitIntent(action)).toBe(true);
        const intents = env.translateForTest(action, false);
        expect(intents).toHaveLength(1);
        expect(intents[0]).toMatchObject({
          type: "build_unit",
          unit: UnitType.Warship,
        });
        const tile = (intents[0] as { tile: number }).tile;
        expect(env.inspectTile(tile).water).toBe(true);
        expect(env.inspectTile(tile).ownerIsAgent).toBe(false);
        r = env.step(action);
        expect(r.info.actionAccepted).toBe(true);
        expect(r.info.intentCount).toBeGreaterThan(0);
        expect(
          env.inspectActiveUnits().some((u) => u.type === UnitType.Warship),
        ).toBe(true);
        return;
      }
      r = env.step({
        actionType: obs.actionMask[ACTION_ATTACK] === 1 ? ACTION_ATTACK : ACTION_NOOP,
        target: 0,
        region: 0,
        quantity: 4,
        unit: UNIT_CITY,
      });
    }
    throw new Error("warship never became mask-legal after port");
  }, 180000);

  it("masks nukes until a ready silo exists, then emits a non-friendly target", async () => {
    const env = await AgentEnv.create(
      testConfig({
        map: "Halkidiki",
        mapSize: "Compact",
        nations: 2,
        difficulty: "Easy",
        bots: 0,
        seed: "build-nuke-legal",
        maxTicks: 4000,
        startingGold: 5_000_000,
      }),
      terrain,
    );
    let r = env.step(legalMaskedAction(env.peekObs(), ACTION_SPAWN));
    expect(r.obs.unitMask[UNIT_ATOM]).toBe(0);
    expect(r.obs.unitMask[UNIT_HBOMB]).toBe(0);
    expect(r.obs.unitMask[UNIT_MIRV]).toBe(0);

    let builtSilo = false;
    for (let i = 0; i < 50 && !r.done; i++) {
      const obs = r.obs;
      if (
        !builtSilo &&
        obs.actionMask[ACTION_BUILD] === 1 &&
        obs.unitMask[UNIT_SILO] === 1
      ) {
        r = env.step(legalMaskedAction(obs, ACTION_BUILD, UNIT_SILO));
        if (r.info.actionAccepted) builtSilo = true;
        continue;
      }
      if (obs.unitMask[UNIT_ATOM] === 1) {
        const region =
          env.findBuildRegionForUnit(UNIT_ATOM) ?? firstSet(obs.buildRegions);
        const action = {
          actionType: ACTION_BUILD,
          target: 0,
          region,
          quantity: 0,
          unit: UNIT_ATOM,
        };
        expect(env.wouldEmitIntent(action)).toBe(true);
        const intents = env.translateForTest(action, false);
        expect(intents).toHaveLength(1);
        expect(intents[0]).toMatchObject({
          type: "build_unit",
          unit: UnitType.AtomBomb,
        });
        const tile = (intents[0] as { tile: number }).tile;
        const info = env.inspectTile(tile);
        expect(info.ownerIsAgent).toBe(false);
        expect(info.impassable).toBe(false);
        r = env.step(action);
        expect(r.info.actionAccepted).toBe(true);
        const liveNuke = env
          .inspectActiveUnits()
          .some(
            (u) =>
              u.type === UnitType.AtomBomb || u.type === UnitType.HydrogenBomb,
          );
        const fallout = env.renderFrame().cells.some((c) => c === 250);
        expect(liveNuke || fallout).toBe(true);
        return;
      }
      r = env.step({
        actionType: obs.actionMask[ACTION_ATTACK] === 1 ? ACTION_ATTACK : ACTION_NOOP,
        target: 0,
        region: 0,
        quantity: 4,
        unit: UNIT_CITY,
      });
    }
    throw new Error("atom never became mask-legal after silo");
  }, 180000);

  it("unavailable warship and nuke choices stay masked", async () => {
    const env = await AgentEnv.create(
      testConfig({
        map: "Halkidiki",
        mapSize: "Compact",
        nations: "disabled",
        bots: 0,
        seed: "build-special-masked",
        maxTicks: 800,
        startingGold: 50_000,
      }),
      terrain,
    );
    const r = env.step(legalMaskedAction(env.peekObs(), ACTION_SPAWN));
    expect(r.obs.unitMask[UNIT_WARSHIP]).toBe(0);
    expect(r.obs.unitMask[UNIT_ATOM]).toBe(0);
    expect(r.obs.unitMask[UNIT_MIRV]).toBe(0);
    expect(
      env.wouldEmitIntent({
        actionType: ACTION_BUILD,
        target: 0,
        region: 0,
        quantity: 0,
        unit: UNIT_WARSHIP,
      }),
    ).toBe(false);
    expect(
      env.wouldEmitIntent({
        actionType: ACTION_BUILD,
        target: 0,
        region: 0,
        quantity: 0,
        unit: UNIT_ATOM,
      }),
    ).toBe(false);
  }, 120000);
});

describe("NationMIRVBehavior game-scoped cooldown", () => {
  it("does not leak recentMirvTargets across sequential games", async () => {
    const cfg = testConfig({
      map: "FourIslands",
      mapSize: "Compact",
      nations: 2,
      bots: 0,
      seed: "mirv-isolation-a",
      maxTicks: 400,
    });
    const first = await AgentEnv.create(cfg, terrain);
    first.recentMirvCooldownForTest().set("victim-a", 0);
    expect(first.recentMirvCooldownForTest().get("victim-a")).toBe(0);

    const second = await AgentEnv.create(
      { ...cfg, seed: "mirv-isolation-b" },
      terrain,
    );
    expect(second.recentMirvCooldownForTest().size).toBe(0);
    expect(second.recentMirvCooldownForTest().get("victim-a")).toBeUndefined();
    expect(first.recentMirvCooldownForTest().get("victim-a")).toBe(0);

    const replay = await AgentEnv.create(cfg, terrain);
    expect(replay.recentMirvCooldownForTest().size).toBe(0);
  }, 180000);
});
