/**
 * Decision-boundary trace recorder / golden comparator.
 *
 * Used by ofai/tests/fidelity.test.ts. Not a training or one-off helper
 * script: this is the committed replay-fidelity utility.
 */
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { AgentEnv } from "../env/AgentEnv";
import { ActionVec, UNIT_HEAD_ORDER } from "../env/ActionTranslator";
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
} from "../env/spec";
import { boundaryRecord, DecisionBoundary } from "../env/stateDigest";
import { TerrainCache } from "../env/TerrainCache";
import { UnitType } from "../../src/core/game/Game";

export const FIXTURE_DIR = path.join(
  path.dirname(fileURLToPath(import.meta.url)),
  "fixtures",
);

export type ActionTuple = [number, number, number, number, number];

export interface GoldenFixture {
  name: string;
  config: EnvConfig;
  actions: ActionTuple[];
  init: Record<string, unknown>;
  boundaries: Array<Record<string, unknown>>;
  /** Feature events observed while recording (oracle verification, not replay input). */
  events?: string[];
}

export const UNIT_CITY = UNIT_HEAD_ORDER.indexOf(UnitType.City);
export const UNIT_DEFENSE = UNIT_HEAD_ORDER.indexOf(UnitType.DefensePost);
export const UNIT_SAM = UNIT_HEAD_ORDER.indexOf(UnitType.SAMLauncher);
export const UNIT_SILO = UNIT_HEAD_ORDER.indexOf(UnitType.MissileSilo);
export const UNIT_PORT = UNIT_HEAD_ORDER.indexOf(UnitType.Port);
export const UNIT_FACTORY = UNIT_HEAD_ORDER.indexOf(UnitType.Factory);
export const UNIT_ATOM = UNIT_HEAD_ORDER.indexOf(UnitType.AtomBomb);
export const UNIT_HBOMB = UNIT_HEAD_ORDER.indexOf(UnitType.HydrogenBomb);
export const UNIT_MIRV = UNIT_HEAD_ORDER.indexOf(UnitType.MIRV);
export const UNIT_WARSHIP = UNIT_HEAD_ORDER.indexOf(UnitType.Warship);

export interface Scenario {
  name: string;
  config: Partial<EnvConfig>;
  /** Preferred legal action types after spawn, first match wins. */
  prefer: number[];
  maxSteps: number;
  /** If set, send this action type even when masked (for no_spawn). */
  forceAction?: number;
  stopOnDone?: boolean;
  /** Fail recording unless this action type appears after spawn. */
  requireAction?: number;
  /** When BUILD is chosen, prefer this unit head if that bit is legal. */
  preferUnit?: number;
  /** Build these unit heads in order, each only when mask-legal. */
  buildOrder?: number[];
  /** Fail recording unless all of these feature events occurred. */
  requireEvents?: string[];
  /** After buildOrder is finished, use this prefer list instead. */
  preferAfterBuild?: number[];
  /** Extra decisions after every required event, so impacts can settle. */
  extraStepsAfterEvents?: number;
  /** Use the highest legal quantity bit (needed for real expansion / boats). */
  preferMaxQuantity?: boolean;
  /**
   * When BUILD is a nuke, aim at an enemy SAM tile so official interception
   * range is satisfied. Recording only; replay uses the stored region.
   */
  preferNukeNearEnemySam?: boolean;
}

export const SCENARIOS: Scenario[] = [
  {
    name: "spawn",
    config: {
      map: "FourIslands",
      mapSize: "Compact",
      nations: "disabled",
      bots: 0,
      seed: "golden-spawn",
      maxTicks: 400,
    },
    prefer: [ACTION_NOOP],
    maxSteps: 4,
  },
  {
    name: "wilderness-combat",
    config: {
      map: "Halkidiki",
      mapSize: "Compact",
      nations: "disabled",
      bots: 0,
      seed: "golden-wilderness",
      maxTicks: 2000,
    },
    prefer: [ACTION_ATTACK, ACTION_NOOP],
    maxSteps: 12,
  },
  {
    name: "player-combat",
    config: {
      map: "Halkidiki",
      mapSize: "Compact",
      nations: "disabled",
      bots: 1,
      seed: "golden-player-combat",
      maxTicks: 3000,
    },
    prefer: [ACTION_ATTACK, ACTION_RETREAT_ALL, ACTION_NOOP],
    maxSteps: 16,
  },
  {
    name: "boats",
    config: {
      map: "FourIslands",
      mapSize: "Normal",
      nations: "disabled",
      bots: 0,
      seed: "golden-boats",
      maxTicks: 4000,
    },
    prefer: [ACTION_BOAT, ACTION_ATTACK, ACTION_NOOP],
    maxSteps: 24,
    requireAction: ACTION_BOAT,
  },
  {
    name: "structures",
    config: {
      map: "Halkidiki",
      mapSize: "Compact",
      nations: "disabled",
      bots: 0,
      seed: "golden-structures",
      maxTicks: 4000,
    },
    prefer: [ACTION_BUILD, ACTION_ATTACK, ACTION_NOOP],
    maxSteps: 80,
    requireAction: ACTION_BUILD,
  },
  {
    name: "diplomacy",
    config: {
      map: "FourIslands",
      mapSize: "Normal",
      nations: "disabled",
      bots: 5,
      seed: "golden-diplomacy",
      maxTicks: 4000,
    },
    prefer: [ACTION_ALLY, ACTION_EMBARGO, ACTION_BREAK_ALLY, ACTION_ATTACK, ACTION_NOOP],
    maxSteps: 20,
  },
  {
    name: "nations",
    config: {
      map: "FourIslands",
      mapSize: "Normal",
      nations: 3,
      bots: 4,
      difficulty: "Easy",
      seed: "golden-nations",
      maxTicks: 3000,
    },
    prefer: [ACTION_ATTACK, ACTION_BOAT, ACTION_BUILD, ACTION_ALLY, ACTION_NOOP],
    maxSteps: 16,
  },
  {
    name: "terminal-no-spawn",
    config: {
      map: "Halkidiki",
      mapSize: "Compact",
      nations: "disabled",
      bots: 0,
      seed: "golden-no-spawn",
      maxTicks: 40,
    },
    prefer: [ACTION_NOOP],
    forceAction: ACTION_NOOP,
    maxSteps: 8,
    stopOnDone: true,
  },
  {
    name: "terminal-timeout",
    config: {
      map: "Halkidiki",
      mapSize: "Compact",
      nations: "disabled",
      bots: 0,
      seed: "golden-timeout",
      maxTicks: 80,
    },
    prefer: [ACTION_NOOP],
    maxSteps: 16,
    stopOnDone: true,
  },
];

/**
 * Later-system official-oracle fixtures. startingGold is the official
 * GameConfig passthrough (unset = 0). Keep it below 3_000_000 unless the
 * scenario needs nation nuke spend — above that nations enter the
 * high-starting-gold SAM-first cooldown.
 *
 * Agent BUILD emits Warship (water + owned Port) and atom/H-bomb/MIRV
 * (legal target region, never friendly land). Schema-v2 heads are unchanged.
 *
 * Doomsday clock grace is 600s on every official speed and is not
 * wire-configurable. Compact fixtures cannot observe it; default stays off.
 */
export const FEATURE_SCENARIOS: Scenario[] = [
  {
    name: "port-factory",
    config: {
      map: "Halkidiki",
      mapSize: "Compact",
      nations: "disabled",
      bots: 0,
      seed: "golden-port-factory",
      maxTicks: 2500,
      startingGold: 500_000,
    },
    prefer: [ACTION_BUILD, ACTION_ATTACK, ACTION_NOOP],
    buildOrder: [UNIT_PORT, UNIT_FACTORY],
    preferMaxQuantity: true,
    maxSteps: 50,
    requireEvents: ["port_complete", "factory_complete"],
    extraStepsAfterEvents: 3,
  },
  {
    name: "rails-trains",
    config: {
      map: "Halkidiki",
      mapSize: "Compact",
      nations: "disabled",
      bots: 0,
      seed: "golden-rails-trains",
      maxTicks: 3000,
      startingGold: 500_000,
    },
    prefer: [ACTION_BUILD, ACTION_ATTACK, ACTION_NOOP],
    buildOrder: [UNIT_CITY, UNIT_FACTORY],
    preferMaxQuantity: true,
    maxSteps: 60,
    requireEvents: ["city_complete", "factory_complete", "rail_station", "train"],
    extraStepsAfterEvents: 4,
  },
  {
    name: "trade-ships",
    config: {
      map: "FourIslands",
      mapSize: "Compact",
      nations: 4,
      difficulty: "Impossible",
      bots: 0,
      seed: "golden-trade-ships",
      maxTicks: 8000,
      startingGold: 2_500_000,
    },
    prefer: [ACTION_ATTACK, ACTION_NOOP],
    buildOrder: [UNIT_PORT],
    preferMaxQuantity: true,
    maxSteps: 200,
    requireEvents: ["port_complete", "trade_ship"],
    extraStepsAfterEvents: 4,
  },
  {
    name: "warships",
    config: {
      map: "FourIslands",
      mapSize: "Compact",
      nations: 4,
      difficulty: "Impossible",
      bots: 0,
      seed: "golden-warships",
      maxTicks: 8000,
      startingGold: 2_500_000,
    },
    prefer: [ACTION_ATTACK, ACTION_NOOP],
    buildOrder: [UNIT_PORT],
    preferMaxQuantity: true,
    maxSteps: 120,
    requireEvents: ["warship", "warship_moved"],
    extraStepsAfterEvents: 12,
  },
  {
    name: "warships-combat",
    config: {
      map: "FourIslands",
      mapSize: "Compact",
      nations: 4,
      difficulty: "Impossible",
      bots: 0,
      seed: "golden-nukes-atom",
      maxTicks: 8000,
      startingGold: 50_000_000,
      disabledUnits: ["Hydrogen Bomb"],
    },
    prefer: [ACTION_ATTACK, ACTION_NOOP],
    preferAfterBuild: [ACTION_NOOP],
    buildOrder: [UNIT_CITY, UNIT_SAM, UNIT_SAM],
    preferMaxQuantity: true,
    maxSteps: 160,
    requireEvents: ["warship", "warship_moved", "warship_combat"],
    extraStepsAfterEvents: 4,
  },
  {
    name: "nukes",
    config: {
      map: "FourIslands",
      mapSize: "Compact",
      nations: 4,
      difficulty: "Impossible",
      bots: 0,
      seed: "golden-nukes",
      maxTicks: 8000,
      startingGold: 50_000_000,
    },
    prefer: [ACTION_ATTACK, ACTION_NOOP],
    buildOrder: [UNIT_SAM, UNIT_CITY, UNIT_SILO],
    preferMaxQuantity: true,
    maxSteps: 220,
    requireEvents: ["hbomb_launch", "hbomb_impact"],
    extraStepsAfterEvents: 6,
  },
  {
    name: "nukes-atom",
    config: {
      map: "FourIslands",
      mapSize: "Compact",
      nations: 4,
      difficulty: "Impossible",
      bots: 0,
      seed: "golden-nukes-atom",
      maxTicks: 8000,
      startingGold: 50_000_000,
      disabledUnits: ["Hydrogen Bomb"],
    },
    prefer: [ACTION_ATTACK, ACTION_NOOP],
    buildOrder: [UNIT_SAM, UNIT_CITY, UNIT_SILO],
    preferMaxQuantity: true,
    maxSteps: 220,
    requireEvents: ["atom_launch", "atom_impact"],
    extraStepsAfterEvents: 6,
  },
  {
    name: "nukes-mirv",
    config: {
      map: "FourIslands",
      mapSize: "Compact",
      nations: 4,
      difficulty: "Easy",
      bots: 0,
      seed: "golden-nukes-mirv",
      maxTicks: 8000,
      startingGold: 50_000_000,
      disabledUnits: ["Atom Bomb", "Hydrogen Bomb", "SAM Launcher"],
    },
    prefer: [ACTION_BUILD, ACTION_ATTACK, ACTION_NOOP],
    preferUnit: UNIT_MIRV,
    buildOrder: [UNIT_SILO],
    preferMaxQuantity: true,
    maxSteps: 200,
    requireEvents: ["mirv_launch", "mirv_impact"],
    extraStepsAfterEvents: 10,
  },
  {
    name: "sam-intercept",
    config: {
      map: "Halkidiki",
      mapSize: "Compact",
      nations: 3,
      difficulty: "Impossible",
      bots: 0,
      seed: "golden-sam-intercept",
      maxTicks: 8000,
      startingGold: 50_000_000,
      disabledUnits: ["Hydrogen Bomb"],
    },
    prefer: [ACTION_ATTACK, ACTION_NOOP],
    buildOrder: [UNIT_SILO],
    preferMaxQuantity: true,
    preferNukeNearEnemySam: true,
    maxSteps: 220,
    requireEvents: ["atom_launch", "nuke_intercepted"],
    extraStepsAfterEvents: 6,
  },
  {
    name: "win",
    config: {
      map: "Halkidiki",
      mapSize: "Compact",
      nations: "disabled",
      bots: 0,
      seed: "golden-win",
      maxTicks: 2000,
      maxTimerValue: 1,
    },
    prefer: [ACTION_ATTACK, ACTION_NOOP],
    preferMaxQuantity: true,
    maxSteps: 80,
    stopOnDone: true,
    requireEvents: ["win"],
  },
];

export function testConfig(overrides: Partial<EnvConfig> = {}): EnvConfig {
  return {
    map: "FourIslands",
    mapSize: "Normal",
    nations: 2,
    difficulty: "Easy",
    bots: 5,
    seed: "fidelity-seed",
    maxTicks: 4000,
    decisionInterval: 10,
    shaping: 0,
    enableDigest: true,
    ...overrides,
  };
}

function firstSet(mask: Uint8Array): number {
  for (let i = 0; i < mask.length; i++) {
    if (mask[i] === 1) return i;
  }
  return 0;
}

function lastSet(mask: Uint8Array): number {
  for (let i = mask.length - 1; i >= 0; i--) {
    if (mask[i] === 1) return i;
  }
  return 0;
}

function targetFor(obs: { targetMasks: Uint8Array }, action: number): number {
  const row = action * NUM_PLAYER_SLOTS;
  for (let i = 0; i < NUM_PLAYER_SLOTS; i++) {
    if (obs.targetMasks[row + i] === 1) return i;
  }
  return 0;
}

function regionFor(
  obs: {
    spawnRegions: Uint8Array;
    buildRegions: Uint8Array;
    boatRegions: Uint8Array;
  },
  action: number,
): number {
  if (action === ACTION_SPAWN) return firstSet(obs.spawnRegions);
  if (action === ACTION_BUILD) return firstSet(obs.buildRegions);
  if (action === ACTION_BOAT) return firstSet(obs.boatRegions);
  return 0;
}

export function legalMaskedAction(
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
  preferMaxQuantity?: boolean,
): ActionVec {
  let unit = firstSet(obs.unitMask);
  if (preferUnit !== undefined && obs.unitMask[preferUnit] === 1) {
    unit = preferUnit;
  }
  return {
    actionType,
    target: targetFor(obs, actionType),
    region: regionFor(obs, actionType),
    quantity: preferMaxQuantity
      ? lastSet(obs.quantityMask)
      : firstSet(obs.quantityMask),
    unit,
  };
}

export function pickPreferredAction(
  obs: {
    actionMask: Uint8Array;
    targetMasks: Uint8Array;
    quantityMask: Uint8Array;
    unitMask: Uint8Array;
    spawnRegions: Uint8Array;
    buildRegions: Uint8Array;
    boatRegions: Uint8Array;
  },
  prefer: number[],
  forceAction?: number,
  preferUnit?: number,
  preferMaxQuantity?: boolean,
): ActionVec {
  if (forceAction !== undefined) {
    return legalMaskedAction(obs, forceAction, preferUnit, preferMaxQuantity);
  }
  if (obs.actionMask[ACTION_SPAWN] === 1) {
    return legalMaskedAction(obs, ACTION_SPAWN, preferUnit, preferMaxQuantity);
  }
  for (const a of prefer) {
    if (obs.actionMask[a] === 1) {
      return legalMaskedAction(obs, a, preferUnit, preferMaxQuantity);
    }
  }
  for (let a = 0; a < NUM_ACTION_TYPES; a++) {
    if (obs.actionMask[a] === 1) {
      return legalMaskedAction(obs, a, preferUnit, preferMaxQuantity);
    }
  }
  return legalMaskedAction(obs, ACTION_NOOP, preferUnit, preferMaxQuantity);
}

export function actionTuple(a: ActionVec): ActionTuple {
  return [a.actionType, a.target, a.region, a.quantity, a.unit];
}

export function actionFromTuple(t: ActionTuple): ActionVec {
  return {
    actionType: t[0],
    target: t[1],
    region: t[2],
    quantity: t[3],
    unit: t[4],
  };
}

export function fixturePath(name: string): string {
  return path.join(FIXTURE_DIR, `golden-${name}.json`);
}

export function loadGolden(name: string): GoldenFixture {
  return JSON.parse(readFileSync(fixturePath(name), "utf8")) as GoldenFixture;
}

export function writeGolden(fixture: GoldenFixture): void {
  mkdirSync(FIXTURE_DIR, { recursive: true });
  writeFileSync(fixturePath(fixture.name), `${JSON.stringify(fixture, null, 2)}\n`);
}

interface UnitSnap {
  id: number;
  type: string;
  tile: number;
  ownerIsAgent: boolean;
  underConstruction: boolean;
  trainStation: boolean;
  health: number;
}

function snapshotUnits(env: AgentEnv): UnitSnap[] {
  return env.inspectActiveUnits();
}

function countFallout(env: AgentEnv): number {
  const cells = env.renderFrame().cells;
  let n = 0;
  for (let i = 0; i < cells.length; i++) {
    if (cells[i] === 250) n++;
  }
  return n;
}

function aimBuildRegion(
  env: AgentEnv,
  obs: Parameters<typeof legalMaskedAction>[0],
  action: ActionVec,
  scenario: Scenario,
): ActionVec {
  const unitType = UNIT_HEAD_ORDER[action.unit];
  const isNuke =
    unitType === UnitType.AtomBomb ||
    unitType === UnitType.HydrogenBomb ||
    unitType === UnitType.MIRV;
  if (scenario.preferNukeNearEnemySam && isNuke) {
    const aim = env.findNukeTileNearEnemySam();
    if (aim !== null) return { ...action, region: aim.region };
  }
  if (unitType === UnitType.Warship || isNuke) {
    const region = env.findBuildRegionForUnit(action.unit);
    if (region !== null) return { ...action, region };
  }
  if (obs.buildRegions[action.region] === 1) return action;
  return action;
}

function pickFeatureAction(
  obs: Parameters<typeof legalMaskedAction>[0],
  scenario: Scenario,
  builtCount: number,
  env: AgentEnv,
): { action: ActionVec; consumedBuild: boolean } {
  const nextUnit = scenario.buildOrder?.[builtCount];
  const preferUnit = nextUnit ?? scenario.preferUnit;
  if (obs.actionMask[ACTION_SPAWN] === 1) {
    return {
      action: legalMaskedAction(obs, ACTION_SPAWN),
      consumedBuild: false,
    };
  }
  const maxQ = scenario.preferMaxQuantity === true;
  if (
    scenario.preferNukeNearEnemySam === true &&
    nextUnit === undefined &&
    obs.actionMask[ACTION_BUILD] === 1 &&
    obs.unitMask[UNIT_ATOM] === 1 &&
    env.findNukeTileNearEnemySam() !== null
  ) {
    return {
      action: aimBuildRegion(
        env,
        obs,
        legalMaskedAction(obs, ACTION_BUILD, UNIT_ATOM, maxQ),
        scenario,
      ),
      consumedBuild: false,
    };
  }
  if (
    nextUnit !== undefined &&
    obs.actionMask[ACTION_BUILD] === 1 &&
    obs.unitMask[nextUnit] === 1
  ) {
    return {
      action: aimBuildRegion(
        env,
        obs,
        legalMaskedAction(obs, ACTION_BUILD, nextUnit, maxQ),
        scenario,
      ),
      consumedBuild: true,
    };
  }
  const prefer =
    nextUnit !== undefined
      ? scenario.prefer.filter((a) => a !== ACTION_BUILD)
      : (scenario.preferAfterBuild ?? scenario.prefer);
  const action = pickPreferredAction(
    obs,
    prefer,
    scenario.forceAction,
    preferUnit,
    maxQ,
  );
  if (action.actionType === ACTION_BUILD) {
    return { action: aimBuildRegion(env, obs, action, scenario), consumedBuild: false };
  }
  return { action, consumedBuild: false };
}

export function collectFeatureEvents(
  env: AgentEnv,
  prev: {
    units: UnitSnap[];
    fallout: number;
    liveNukes: Set<number>;
    samArmed: boolean;
  },
  terminalCause: string,
  win: boolean,
): {
  events: string[];
  units: UnitSnap[];
  fallout: number;
  liveNukes: Set<number>;
  samArmed: boolean;
} {
  const units = snapshotUnits(env);
  const fallout = countFallout(env);
  const events: string[] = [];
  const byType = (t: string, pred?: (u: UnitSnap) => boolean) =>
    units.filter((u) => u.type === t && (pred === undefined || pred(u)));
  const complete = (t: string, agentOnly = true) =>
    byType(t, (u) => (!agentOnly || u.ownerIsAgent) && !u.underConstruction);

  if (complete(UnitType.Port).length > 0) events.push("port_complete");
  if (complete(UnitType.Factory).length > 0) events.push("factory_complete");
  if (complete(UnitType.City).length > 0) events.push("city_complete");
  if (complete(UnitType.MissileSilo).length > 0) events.push("silo_complete");
  if (complete(UnitType.SAMLauncher).length > 0) events.push("sam_complete");
  if (complete(UnitType.Port, false).some((u) => !u.ownerIsAgent)) {
    events.push("nation_port");
  }
  if (complete(UnitType.MissileSilo, false).some((u) => !u.ownerIsAgent)) {
    events.push("nation_silo");
  }
  if (complete(UnitType.SAMLauncher, false).some((u) => !u.ownerIsAgent)) {
    events.push("nation_sam");
  }

  const stations = units.filter((u) => u.trainStation && !u.underConstruction);
  if (stations.length >= 2) events.push("rail_station");
  if (byType(UnitType.Train).length > 0) events.push("train");
  if (byType(UnitType.TradeShip).length > 0) events.push("trade_ship");

  const warships = byType(UnitType.Warship);
  if (warships.length > 0) events.push("warship");
  const prevById = new Map(prev.units.map((u) => [u.id, u]));
  if (warships.some((u) => prevById.get(u.id)?.tile !== undefined && prevById.get(u.id)!.tile !== u.tile)) {
    events.push("warship_moved");
  }
  if (
    byType(UnitType.Shell).length > 0 ||
    warships.some((u) => {
      const before = prevById.get(u.id);
      return before !== undefined && u.health < before.health;
    })
  ) {
    events.push("warship_combat");
  }

  const nukeTypes = [
    [UnitType.AtomBomb, "atom"] as const,
    [UnitType.HydrogenBomb, "hbomb"] as const,
    [UnitType.MIRV, "mirv"] as const,
    [UnitType.MIRVWarhead, "mirv"] as const,
  ];
  const liveNukes = new Set<number>();
  for (const [type, tag] of nukeTypes) {
    const now = byType(type);
    if (now.length > 0) {
      events.push(`${tag}_launch`);
      for (const u of now) liveNukes.add(u.id);
    }
    const vanished = prev.units.filter(
      (u) => u.type === type && !units.some((n) => n.id === u.id),
    );
    if (vanished.length > 0 && fallout > prev.fallout) {
      events.push(`${tag}_impact`);
    }
  }

  const samNow = byType(UnitType.SAMMissile).length > 0;
  if (samNow) events.push("sam_missile");
  const samArmed = prev.samArmed || (samNow && (liveNukes.size > 0 || prev.liveNukes.size > 0));
  const lostNukes = [...prev.liveNukes].filter((id) => !units.some((u) => u.id === id));
  if (lostNukes.length > 0 && samArmed && fallout <= prev.fallout) {
    events.push("nuke_intercepted");
  }
  if (env.inspectBombIntercepts() > 0) {
    events.push("nuke_intercepted");
  }

  if (env.peekSlots().some((p) => p !== null && p.inDoomsdayClock())) {
    events.push("doomsday");
  }
  if (win || terminalCause === "win") events.push("win");

  return { events, units, fallout, liveNukes, samArmed };
}

export async function recordScenario(
  terrain: TerrainCache,
  scenario: Scenario,
): Promise<GoldenFixture> {
  const cfg = testConfig(scenario.config);
  const env = await AgentEnv.create(cfg, terrain);
  const init = boundaryRecord(env.strongStateDigest());
  const actions: ActionTuple[] = [];
  const boundaries: Array<Record<string, unknown>> = [];
  const seen = new Set<string>();
  let obs = env.peekObs();
  let builtCount = 0;
  let extraLeft: number | null = null;
  let prevUnits = snapshotUnits(env);
  let prevFallout = 0;
  let liveNukes = new Set<number>();
  let samArmed = false;
  for (let i = 0; i < scenario.maxSteps; i++) {
    const picked = pickFeatureAction(obs, scenario, builtCount, env);
    const action = picked.action;
    actions.push(actionTuple(action));
    const result = env.step(action);
    if (picked.consumedBuild && result.info.actionAccepted) {
      builtCount++;
    }
    boundaries.push(boundaryRecord(env.strongStateDigest()));
    const collected = collectFeatureEvents(
      env,
      { units: prevUnits, fallout: prevFallout, liveNukes, samArmed },
      result.info.terminalCause,
      result.info.win,
    );
    for (const e of collected.events) seen.add(e);
    prevUnits = collected.units;
    prevFallout = collected.fallout;
    liveNukes = collected.liveNukes;
    samArmed = collected.samArmed;
    obs = result.obs;
    if (scenario.requireEvents && extraLeft === null) {
      const haveAll = scenario.requireEvents.every((e) => seen.has(e));
      if (haveAll) extraLeft = scenario.extraStepsAfterEvents ?? 0;
    }
    if (scenario.stopOnDone !== false && result.done) break;
    if (extraLeft !== null) {
      if (extraLeft <= 0) break;
      extraLeft--;
    }
  }
  if (scenario.requireAction !== undefined) {
    const taken = actions.some((a) => a[0] === scenario.requireAction);
    if (!taken) {
      throw new Error(
        `${scenario.name} never took action ${scenario.requireAction}`,
      );
    }
  }
  if (scenario.requireEvents !== undefined) {
    const missing = scenario.requireEvents.filter((e) => !seen.has(e));
    if (missing.length > 0) {
      const last = boundaries[boundaries.length - 1];
      throw new Error(
        `${scenario.name} missing events ${missing.join(", ")} (saw ${[...seen].join(",") || "none"}; tilesFrac=${String(last?.tilesFrac)}; steps=${actions.length}; tick=${String(last?.tick)}; terminal=${String(last?.terminalCause)})`,
      );
    }
  }
  return {
    name: scenario.name,
    config: cfg,
    actions,
    init,
    boundaries,
    events: [...seen].sort(),
  };
}

export async function replayActions(
  terrain: TerrainCache,
  config: EnvConfig,
  actions: ActionTuple[],
): Promise<{ init: DecisionBoundary; boundaries: DecisionBoundary[] }> {
  const env = await AgentEnv.create({ ...config, enableDigest: true }, terrain);
  const init = env.strongStateDigest();
  const boundaries: DecisionBoundary[] = [];
  for (const tuple of actions) {
    env.step(actionFromTuple(tuple));
    boundaries.push(env.strongStateDigest());
  }
  return { init, boundaries };
}

export function assertBoundariesMatch(
  expected: Array<Record<string, unknown>>,
  actual: DecisionBoundary[],
): void {
  if (expected.length !== actual.length) {
    throw new Error(
      `boundary count ${actual.length} !== golden ${expected.length}`,
    );
  }
  for (let i = 0; i < expected.length; i++) {
    const exp = expected[i];
    const got = boundaryRecord(actual[i]);
    const keys = [
      "tick",
      "coreHash",
      "digest",
      "tileDigest",
      "obsDigest",
      "stepDigest",
      "reward",
      "intentCount",
      "actionAccepted",
      "terminalCause",
      "win",
      "stageSuccess",
    ] as const;
    for (const key of keys) {
      if (exp[key] !== got[key]) {
        throw new Error(
          `step ${i} ${key}: golden=${String(exp[key])} replay=${String(got[key])}`,
        );
      }
    }
  }
}

export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a += 0x6d2b79f5;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function randomLegalAction(
  obs: {
    actionMask: Uint8Array;
    targetMasks: Uint8Array;
    quantityMask: Uint8Array;
    unitMask: Uint8Array;
    spawnRegions: Uint8Array;
    buildRegions: Uint8Array;
    boatRegions: Uint8Array;
  },
  rng: () => number,
): ActionVec {
  const legal: number[] = [];
  for (let a = 0; a < NUM_ACTION_TYPES; a++) {
    if (obs.actionMask[a] === 1) legal.push(a);
  }
  const actionType = legal[Math.floor(rng() * legal.length)] ?? ACTION_NOOP;
  return legalMaskedAction(obs, actionType);
}

/** CLI used by the Rust official-vs-Rust parity test. Not a one-off helper. */
if (process.argv.includes("--replay-stdin")) {
  void (async () => {
    const { NodeGameMapLoader } = await import(
      "../../tests/perf/fullgame/NodeGameMapLoader.ts"
    );
    const input = JSON.parse(readFileSync(0, "utf8")) as {
      config: EnvConfig;
      actions: ActionTuple[];
    };
    const maps = path.join(
      path.dirname(fileURLToPath(import.meta.url)),
      "../../resources/maps",
    );
    const terrain = new TerrainCache(new NodeGameMapLoader(maps));
    const replayed = await replayActions(terrain, input.config, input.actions);
    process.stdout.write(
      `${JSON.stringify({
        init: replayed.init,
        boundaries: replayed.boundaries,
      })}\n`,
    );
    process.exit(0);
  })().catch((err: unknown) => {
    console.error(err);
    process.exit(1);
  });
}
