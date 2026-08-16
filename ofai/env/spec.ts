/**
 * Shared environment specification between the TypeScript harness and the
 * Python trainer. The Python side (agent/spaces.py) mirrors these constants;
 * keep them in sync.
 */

export const SPATIAL_CHANNELS = 10;
export const SPATIAL_SIZE = 64; // planes are SPATIAL_SIZE x SPATIAL_SIZE
export const NUM_PLAYER_SLOTS = 16; // slot 0 is always the agent
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

// Troop fractions for attack/boat, indexed by the quantity head.
export const TROOP_FRACTIONS = [0.05, 0.15, 0.3, 0.5, 0.8];

// Reward coefficients.
export const REWARD_WIN = 1.0;
export const REWARD_DEATH = -1.0;
export const REWARD_LOSS_ALIVE = -0.5; // someone else won while agent alive
export const REWARD_TIMEOUT_ALIVE = -0.25; // hit maxTicks without a winner
// Paid once when a real attack starts (not when the policy merely clicks).
// Attacks deplete troops, so this is self-limiting and cannot be farmed.
export const REWARD_ATTACK_START = 0.015;
export const REWARD_INCOME = 0.03; // clipped relative troop-regen growth
// NOTE (run6 diagnosis): the boat/build "activity start" bonuses and the
// incoming-attack penalty were removed. The policy farmed the boat/build
// bonuses (30% boat / 18% build / 22% ally) instead of expanding, and the
// incoming penalty punished expansion (more borders -> more attacks -> more
// penalty), teaching the policy to stay small. Territory shaping + survival
// are now the only dense signals, so expansion is the best strategy.
// Timeout-alive is death-sized unless peak tiles exceeded spawn by this much.
export const EXPAND_EPS = 0.002;

export interface EnvConfig {
  map: string; // GameMapType enum key, e.g. "FourIslands"
  mapSize: "Normal" | "Compact";
  nations: number | "default" | "disabled";
  difficulty: "Easy" | "Medium" | "Hard" | "Impossible";
  bots: number; // tribe filler bots (0-400)
  seed: string; // deterministic seed (becomes the gameID)
  maxTicks: number; // episode cap in ticks (10 ticks = 1 game second)
  decisionInterval: number; // ticks between agent decisions
  shaping: number; // coefficient for territory-delta reward shaping (0 = off)
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
