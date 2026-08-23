/**
 * Shared environment specification between the TypeScript harness and the
 * Python trainer. The Python side (agent/spaces.py) mirrors these constants;
 * keep them in sync.
 */

export const SPATIAL_CHANNELS = 10;
export const SPATIAL_SIZE = 64; // planes are SPATIAL_SIZE x SPATIAL_SIZE
export const NUM_PLAYER_SLOTS = 16; // slot 0 is always the agent
// When more than 15 opponents exist, slots 1-14 stay individually
// targetable (border/attackers first). Slot 15 is an untargetable
// overflow aggregate (both Nation+Bot flags). Residual: >14 simultaneous
// border/attackers still cannot all be targeted; a v3 schema would be
// needed for full observability.
export const PLAYER_FEATURES = 14;
export const GLOBAL_FEATURES = 10;

export const REGION_GRID = 32; // coarse action grid: REGION_GRID x REGION_GRID
export const NUM_REGIONS = REGION_GRID * REGION_GRID;

// Factorized action space sizes (action vector = 5 ints per decision).
export const NUM_ACTION_TYPES = 9;
export const NUM_QUANTITIES = 5;
export const NUM_UNIT_TYPES = 10;

export const ACTION_NOOP = 0;
export const ACTION_SPAWN = 1;
export const ACTION_ATTACK = 2;
export const ACTION_RETREAT_ALL = 3;
export const ACTION_BUILD = 4;
export const ACTION_BOAT = 5;
export const ACTION_ALLY = 6; // request, or accept a pending incoming request
export const ACTION_BREAK_ALLY = 7;
export const ACTION_EMBARGO = 8; // toggle embargo on target

export const ACTION_NAMES = [
  "noop",
  "spawn",
  "attack",
  "retreat_all",
  "build",
  "boat",
  "ally",
  "break_ally",
  "embargo",
] as const;

// Troop fractions for attack/boat, indexed by the quantity head.
export const TROOP_FRACTIONS = [0.05, 0.15, 0.3, 0.5, 0.8];

// Flat target-mask layout: [NUM_ACTION_TYPES, NUM_PLAYER_SLOTS] row-major.
export const TARGET_MASKS_SIZE = NUM_ACTION_TYPES * NUM_PLAYER_SLOTS;

// Terminal outcomes. Exactly one non-`none` cause is set when done=true.
// `win` is a core FFA win only (80% land / last standing). A curriculum
// tile milestone is `curriculum_success`, never `win`.
export type TerminalCause =
  | "none"
  | "win"
  | "curriculum_success"
  | "death"
  | "loss_alive"
  | "timeout"
  | "no_spawn";

// Reward coefficients. Core win dominates every other terminal, including
// the early-stage curriculum milestone. Non-objective terminals are -1 so
// a losing trajectory cannot outscore a win even with dense shaping
// (curriculum keeps shaping <= 1, and shaping is bounded state-potential).
export const REWARD_WIN = 3.0;
export const REWARD_CURRICULUM_SUCCESS = 1.0;
export const REWARD_DEATH = -1.0;
export const REWARD_LOSS_ALIVE = -1.0;
export const REWARD_TIMEOUT = -1.0;
export const REWARD_NO_SPAWN = -1.0;
export const REWARD_SPAWN = 0.1; // one-time when the agent first owns land
// Elimination shaping weight relative to territory delta.
export const ELIMINATION_SHAPING_WEIGHT = 0.25;

/** Per-step reward decomposition reported in StepResult.info. */
export interface RewardTerms {
  terminal: number;
  spawn: number;
  territory: number;
  elimination: number;
  total: number;
}

export interface EnvConfig {
  map: string; // GameMapType enum key, e.g. "FourIslands"
  mapSize: "Normal" | "Compact";
  nations: number | "default" | "disabled";
  difficulty: "Easy" | "Medium" | "Hard" | "Impossible";
  bots: number; // tribe filler bots (0-400)
  seed: string; // deterministic seed (becomes the gameID)
  maxTicks: number; // episode cap in ticks (10 ticks = 1 game second)
  decisionInterval: number; // ticks between agent decisions
  shaping: number; // coefficient for bounded state-potential shaping (0 = off)
  /** When set, only these action types (plus needed SPAWN; NOOP after spawn) are legal. */
  allowedActions?: Array<number | string>;
  /**
   * Optional early-stage competency flag. When the living agent's peak
   * tile fraction reaches this value, `stageSuccess` is set true.
   * This does **not** terminate the episode and is never a core `win`.
   * Terminating on the milestone taught policies to stop expanding.
   */
  winTilesFrac?: number;
  /**
   * Test-only. When true, step info includes the strong decision-boundary
   * digest (tile/obs/mask/reward/intent/terminal). Training leaves this off.
   */
  enableDigest?: boolean;
  /**
   * Official GameConfig.startingGold passthrough. Unset keeps the core
   * default (0). Oracle fixtures set this so later systems stay compact;
   * training configs omit it.
   */
  startingGold?: number;
  /**
   * Official GameConfig.doomsdayClock passthrough. Unset leaves the clock
   * off (core default). Only `enabled` and `speed` are wire-configurable.
   */
  doomsdayClock?: {
    enabled?: boolean;
    speed?: "slow" | "normal" | "fast" | "veryfast";
  };
  /**
   * Official GameConfig.maxTimerValue passthrough (minutes). Unset leaves
   * FFA win at 80% land. Oracle win fixtures set 1 so termination is compact.
   */
  maxTimerValue?: number;
  /**
   * Official GameConfig.disabledUnits passthrough. Unset leaves every unit
   * enabled. Oracle nuke fixtures disable Hydrogen Bomb so nations fire
   * atoms (one third of nations are hydro-only and otherwise erase the
   * agent before an atom is observed).
   */
  disabledUnits?: string[];
}

export const DEFAULT_ENV_CONFIG: EnvConfig = {
  map: "FourIslands",
  mapSize: "Normal",
  nations: "default",
  difficulty: "Medium",
  bots: 20,
  seed: "ofai-seed-0",
  maxTicks: 12000,
  decisionInterval: 10,
  shaping: 0,
};
