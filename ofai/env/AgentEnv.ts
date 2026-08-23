/**
 * One headless OpenFront game exposing a gym-style reset/step interface.
 *
 * The agent occupies a single Human player slot and acts every
 * `decisionInterval` ticks via factorized actions translated to intents.
 * Opponents are the game's built-in Nations/Tribes, which run inside the
 * deterministic core simulation - no network involved.
 */
import { Config } from "../../src/core/configuration/Config";
import { Executor } from "../../src/core/execution/ExecutionManager";
import {
  Difficulty,
  Game,
  GameMapSize,
  GameMapType,
  GameMode,
  GameType,
  Player,
  PlayerInfo,
  PlayerType,
  UnitType,
} from "../../src/core/game/Game";
import { TileRef } from "../../src/core/game/GameMap";
import { createGame } from "../../src/core/game/GameImpl";
import { createNationsForGame } from "../../src/core/game/NationCreation";
import { recentMirvTargetsFor } from "../../src/core/execution/nation/NationMIRVBehavior";
import {
  ErrorUpdate,
  GameUpdateType,
  GameUpdateViewData,
} from "../../src/core/game/GameUpdates";
import { GameRunner } from "../../src/core/GameRunner";
import { PseudoRandom } from "../../src/core/PseudoRandom";
import { GameConfig, GameStartInfo, Intent } from "../../src/core/Schemas";
import { simpleHash } from "../../src/core/Util";
import { ActionTranslator, ActionVec, UNIT_HEAD_ORDER } from "./ActionTranslator";
import { makeObsBuffers, ObsBuffers, ObsExtractor } from "./ObsExtractor";
import { TerrainCache } from "./TerrainCache";
import {
  ACTION_ALLY,
  ACTION_ATTACK,
  ACTION_BOAT,
  ACTION_BREAK_ALLY,
  ACTION_BUILD,
  ACTION_EMBARGO,
  ACTION_NAMES,
  ACTION_NOOP,
  ACTION_RETREAT_ALL,
  ACTION_SPAWN,
  ELIMINATION_SHAPING_WEIGHT,
  EnvConfig,
  NUM_ACTION_TYPES,
  NUM_PLAYER_SLOTS,
  NUM_QUANTITIES,
  REWARD_DEATH,
  REWARD_LOSS_ALIVE,
  REWARD_NO_SPAWN,
  REWARD_SPAWN,
  REWARD_TIMEOUT,
  REWARD_WIN,
  RewardTerms,
  TerminalCause,
  TROOP_FRACTIONS,
} from "./spec";
import {
  computeDecisionBoundary,
  DecisionBoundary,
  StepDigestInput,
} from "./stateDigest";

export interface StepResult {
  obs: ObsBuffers;
  reward: number;
  done: boolean;
  info: {
    tick: number;
    win: boolean;
    stageSuccess: boolean;
    dead: boolean;
    kills: number;
    tilesFrac: number;
    spawned: boolean;
    hash: number | null;
    terminalCause: TerminalCause;
    peakTilesFrac: number;
    rewardTerms: RewardTerms;
    intentCount: number;
    actionAccepted: boolean;
    map: string;
    mapSize: string;
    seed: string;
    /** Present only when EnvConfig.enableDigest is set (tests / goldens). */
    digest?: string;
    tileDigest?: string;
    obsDigest?: string;
    stepDigest?: string;
  };
}

export interface EnvStepProfile {
  simMs: number;
  maskObsMs: number;
  extractMs: number;
  maskFillMs: number;
}

export interface AgentEnvCreateOptions {
  /**
   * Skip HUD name placement in GameRunner. Default true for training.
   * Packed tiles, hashes, rewards, masks, and renderFrame are unchanged.
   */
  skipNamePlacement?: boolean;
  /**
   * Write observations into this buffer (a view into a fixed batch).
   * Training backends bind per-env slices so stacking does not copy.
   */
  obs?: ObsBuffers;
}

export class AgentEnv {
  readonly agentClientID = "agent";

  private game!: Game;
  private runner!: GameRunner;
  private me!: Player;
  private extractor = new ObsExtractor();
  private translator = new ActionTranslator();
  private skipNamePlacement = true;
  private obs: ObsBuffers = makeObsBuffers();
  private turnNumber = 0;
  private kills = 0;
  private lastHash: number | null = null;
  private fatalError: string | null = null;
  private prevTilesFrac = 0;
  private prevElimProgress = 0;
  private spawnedOnce = false;
  private wasSpawned = false;
  private peakTilesFrac = 0;
  private initialOpponentCount = 1;
  private slots: (Player | null)[] = [];
  private allowedSet: Set<number> | null = null;
  private lastStepMeta: StepDigestInput = {
    reward: 0,
    intentCount: 0,
    actionAccepted: true,
    terminalCause: "none",
    win: false,
    stageSuccess: false,
    dead: false,
    tilesFrac: 0,
    peakTilesFrac: 0,
    kills: 0,
  };
  lastProfile: EnvStepProfile = {
    simMs: 0,
    maskObsMs: 0,
    extractMs: 0,
    maskFillMs: 0,
  };
  resetCount = 0;

  get seed(): string {
    return this.cfg.seed;
  }

  /** The most recently computed observation (no recompute). */
  peekObs(): ObsBuffers {
    return this.obs;
  }

  private constructor(
    private cfg: EnvConfig,
    private terrain: TerrainCache,
  ) {}

  static async create(
    cfg: EnvConfig,
    terrain: TerrainCache,
    options?: AgentEnvCreateOptions,
  ): Promise<AgentEnv> {
    const env = new AgentEnv(cfg, terrain);
    if (options?.skipNamePlacement === false) {
      env.skipNamePlacement = false;
    }
    if (options?.obs !== undefined) {
      env.obs = options.obs;
    }
    await env.reset();
    return env;
  }

  private onUpdate = (gu: GameUpdateViewData | ErrorUpdate): void => {
    if ("errMsg" in gu) {
      this.fatalError = gu.errMsg;
      return;
    }
    // Incremental obs maintenance (guaranteed after init by reset()).
    if (this.me !== undefined) {
      this.extractor.applyTileUpdates(this.game, gu.packedTileUpdates);
    }
    const conquests = gu.updates[GameUpdateType.ConquestEvent];
    if (conquests !== undefined) {
      for (const c of conquests) {
        if (c.conquerorId === this.me?.id()) this.kills++;
      }
    }
    const hashes = gu.updates[GameUpdateType.Hash];
    if (hashes !== undefined && hashes.length > 0) {
      this.lastHash = hashes[hashes.length - 1].hash;
    }
  };

  async reset(next?: string | Partial<EnvConfig>): Promise<ObsBuffers> {
    if (typeof next === "string") {
      this.cfg = { ...this.cfg, seed: next };
    } else if (next !== undefined) {
      const prevActions = this.cfg.allowedActions;
      this.cfg = { ...this.cfg, ...next };
      if (next.allowedActions !== undefined && next.allowedActions !== prevActions) {
        this.allowedSet = null;
      }
    }
    const cfg = this.cfg;
    const gameConfig: GameConfig = {
      gameMap: GameMapType[cfg.map as keyof typeof GameMapType],
      gameMapSize:
        cfg.mapSize === "Compact" ? GameMapSize.Compact : GameMapSize.Normal,
      gameMode: GameMode.FFA,
      gameType: GameType.Singleplayer,
      difficulty: Difficulty[cfg.difficulty],
      nations: cfg.nations,
      donateGold: true,
      donateTroops: true,
      bots: cfg.bots,
      infiniteGold: false,
      infiniteTroops: false,
      instantBuild: false,
      randomSpawn: false,
      ...(cfg.startingGold !== undefined
        ? { startingGold: cfg.startingGold }
        : {}),
      ...(cfg.doomsdayClock !== undefined
        ? { doomsdayClock: cfg.doomsdayClock }
        : {}),
      ...(cfg.maxTimerValue !== undefined
        ? { maxTimerValue: cfg.maxTimerValue }
        : {}),
      ...(cfg.disabledUnits !== undefined
        ? { disabledUnits: cfg.disabledUnits as GameConfig["disabledUnits"] }
        : {}),
    };
    const gameID = cfg.seed;
    // Manual wiring mirroring createGameRunner(), but with fresh GameMapImpl
    // instances from TerrainCache (upstream loadTerrainMap caches mutable
    // map state module-level and would corrupt every game after the first).
    const terrain = await this.terrain.freshTerrain(
      gameConfig.gameMap,
      gameConfig.gameMapSize,
    );
    const config = new Config(gameConfig, null, false);
    const random = new PseudoRandom(simpleHash(gameID));
    const humans = [
      new PlayerInfo("OFAI", PlayerType.Human, this.agentClientID, random.nextID()),
    ];
    const gameStart: GameStartInfo = {
      gameID,
      lobbyCreatedAt: 0,
      config: gameConfig,
      players: [
        { clientID: this.agentClientID, username: "OFAI", clanTag: null },
      ],
    };
    const nations = createNationsForGame(
      gameStart,
      terrain.nations,
      terrain.additionalNations,
      humans.length,
      random,
    );
    this.game = createGame(
      humans,
      nations,
      terrain.gameMap,
      terrain.miniGameMap,
      config,
      terrain.teamGameSpawnAreas,
    );
    this.runner = new GameRunner(
      this.game,
      new Executor(this.game, gameID, this.agentClientID),
      this.onUpdate,
      { skipNamePlacement: this.skipNamePlacement },
    );
    this.runner.init();
    const me = this.game.playerByClientID(this.agentClientID);
    if (me === null) throw new Error("agent player missing after init");
    this.me = me;
    this.cfg.seed = gameID;
    this.turnNumber = 0;
    this.kills = 0;
    this.lastHash = null;
    this.fatalError = null;
    this.prevTilesFrac = 0;
    this.prevElimProgress = 0;
    this.spawnedOnce = false;
    this.wasSpawned = false;
    this.peakTilesFrac = 0;
    this.extractor.buildStatic(this.game, this.me);
    this.translator.clearCaches();
    // Let tribes/nations place their spawns before the agent's first decision.
    this.runTicks(5);
    this.initialOpponentCount = Math.max(
      1,
      this.game.players().filter((p) => p.id() !== this.me.id()).length,
    );
    this.lastStepMeta = {
      reward: 0,
      intentCount: 0,
      actionAccepted: true,
      terminalCause: "none",
      win: false,
      stageSuccess: false,
      dead: false,
      tilesFrac: 0,
      peakTilesFrac: 0,
      kills: 0,
    };
    this.lastProfile = { simMs: 0, maskObsMs: 0, extractMs: 0, maskFillMs: 0 };
    return this.extractObs();
  }

  private runTicks(n: number): void {
    for (let i = 0; i < n; i++) {
      this.runner.addTurn({ turnNumber: this.turnNumber++, intents: [] });
      if (!this.runner.executeNextTick()) break;
      if (this.fatalError !== null) throw new Error(this.fatalError);
    }
  }

  private extractObs(): ObsBuffers {
    const t0 = performance.now();
    this.slots = this.extractor.extract(
      this.game,
      this.me,
      this.obs,
      this.cfg.maxTicks,
    );
    const extractMs = performance.now() - t0;
    this.fillMasks(this.slots);
    const maskFillMs = performance.now() - t0 - extractMs;
    this.lastProfile.extractMs = extractMs;
    this.lastProfile.maskFillMs = maskFillMs;
    this.lastProfile.maskObsMs = extractMs + maskFillMs;
    return this.obs;
  }

  private targetRow(actionType: number): number {
    return actionType * NUM_PLAYER_SLOTS;
  }

  private fillQuantityMask(me: Player): boolean {
    const qm = this.obs.quantityMask;
    qm.fill(0);
    let any = false;
    const troops = me.troops();
    for (let q = 0; q < NUM_QUANTITIES; q++) {
      const frac = TROOP_FRACTIONS[q];
      if (Math.floor(troops * frac) >= 1) {
        qm[q] = 1;
        any = true;
      }
    }
    return any;
  }

  private fillMasks(slots: (Player | null)[]): void {
    const am = this.obs.actionMask;
    am.fill(0);
    const tm = this.obs.targetMasks;
    tm.fill(0);
    this.obs.unitMask.fill(0);
    this.obs.quantityMask.fill(0);

    const game = this.game;
    const me = this.me;
    const spawned = me.hasSpawned();
    const inSpawn = game.inSpawnPhase();

    // Spawn phase waits for the Human agent, so allowing NOOP here lets a
    // greedy policy skip the game forever and never hit no_spawn.
    if (inSpawn && !spawned) {
      const anySpawn = this.translator.fillEffectiveSpawnRegions(
        game,
        me,
        this.obs.spawnRegions,
      );
      am[ACTION_SPAWN] = anySpawn ? 1 : 0;
      this.applyAllowedActionsGate(inSpawn, spawned);
      return;
    }
    am[ACTION_NOOP] = 1;
    if (!spawned) {
      this.applyAllowedActionsGate(inSpawn, spawned);
      return; // dead or waiting; only noop
    }

    const anyQuantity = this.fillQuantityMask(me);

    // ATTACK row: wilderness at slot 0; players we can attack elsewhere.
    // Region is ignored for ATTACK (global wilderness / player targeting).
    const attackRow = this.targetRow(ACTION_ATTACK);
    const bordersWilderness = this.extractor.hasAdjacentWilderness();
    if (bordersWilderness) tm[attackRow + 0] = 1;

    const allyRow = this.targetRow(ACTION_ALLY);
    const breakRow = this.targetRow(ACTION_BREAK_ALLY);
    const embargoRow = this.targetRow(ACTION_EMBARGO);

    let anyAttackTarget = bordersWilderness;
    let anyAllyTarget = false;
    let anyBreakTarget = false;
    let anyEmbargoTarget = false;

    for (let i = 1; i < NUM_PLAYER_SLOTS; i++) {
      const p = slots[i];
      if (p === null || !p.isAlive()) continue;

      // canAttackPlayer is only immunity/friendly — attacks with no shared
      // border translate then immediately retreat. Require a land border.
      // Flags were computed once in extract(); do not rescan borders.
      if (
        this.extractor.slotCanAttackPlayer(i) &&
        this.extractor.slotSharesBorderWith(i)
      ) {
        tm[attackRow + i] = 1;
        anyAttackTarget = true;
      }

      if (this.extractor.slotHasAllySignal(i)) {
        tm[allyRow + i] = 1;
        anyAllyTarget = true;
      }

      if (this.extractor.slotAlliedWith(i)) {
        tm[breakRow + i] = 1;
        anyBreakTarget = true;
      } else {
        // Embargo toggle is meaningful against non-allied opponents.
        tm[embargoRow + i] = 1;
        anyEmbargoTarget = true;
      }
    }

    const anyBuild = this.translator.fillEffectiveBuildMasks(
      game,
      me,
      this.obs.unitMask,
      this.obs.buildRegions,
    );
    // Boat dest scan is pointless without a troop quantity; BOAT stays
    // illegal and region bits remain the extractor over-approx.
    const anyBoatDest =
      anyQuantity &&
      this.translator.fillEffectiveBoatRegions(
        game,
        me,
        this.obs.boatRegions,
      );

    am[ACTION_ATTACK] = anyAttackTarget && anyQuantity ? 1 : 0;
    am[ACTION_RETREAT_ALL] = me.outgoingAttacks().some((a) => !a.retreating())
      ? 1
      : 0;
    am[ACTION_BUILD] = anyBuild ? 1 : 0;
    // Transport ships must have a launchable dest (canBuildTransportShip)
    // and a troop quantity. A Port is not required.
    am[ACTION_BOAT] = anyBoatDest && anyQuantity ? 1 : 0;
    am[ACTION_ALLY] = anyAllyTarget ? 1 : 0;
    am[ACTION_BREAK_ALLY] = anyBreakTarget ? 1 : 0;
    am[ACTION_EMBARGO] = anyEmbargoTarget ? 1 : 0;

    this.applyAllowedActionsGate(inSpawn, spawned);
  }

  /**
   * Honor cfg.allowedActions. SPAWN is always preserved during the spawn
   * window before the agent has land. NOOP is preserved only after that
   * (it is illegal until the agent places).
   */
  private applyAllowedActionsGate(inSpawn: boolean, spawned: boolean): void {
    const allowed = this.cfg.allowedActions;
    if (allowed === undefined) return;
    if (this.allowedSet === null) {
      const allow = new Set<number>();
      for (const a of allowed as Array<number | string>) {
        if (typeof a === "number" && Number.isInteger(a)) {
          allow.add(a);
        } else if (typeof a === "string") {
          const idx = (ACTION_NAMES as readonly string[]).indexOf(a);
          if (idx >= 0) allow.add(idx);
        }
      }
      this.allowedSet = allow;
    }
    const allow = this.allowedSet;
    const forceNoop = !(inSpawn && !spawned);
    const forceSpawn = inSpawn && !spawned;
    const am = this.obs.actionMask;
    for (let a = 0; a < NUM_ACTION_TYPES; a++) {
      const extra =
        (a === ACTION_NOOP && forceNoop) || (a === ACTION_SPAWN && forceSpawn);
      if (!allow.has(a) && !extra) am[a] = 0;
    }
  }

  hasAdjacentWildernessCached(): boolean {
    return this.extractor.hasAdjacentWilderness();
  }

  peekSlots(): (Player | null)[] {
    return this.slots;
  }

  /**
   * Translate using an explicit wilderness flag (cache or scan) so tests
   * can prove ActionTranslator cache-vs-scan equivalence.
   */
  translateForTest(action: ActionVec, bordersWilderness: boolean): Intent[] {
    return this.translator.translate(
      this.game,
      this.me,
      action,
      this.slots,
      bordersWilderness,
    );
  }

  /** Recompute masks from the current extract without advancing the sim. */
  refillMasksForTest(): void {
    this.fillMasks(this.slots);
  }

  /** Drop translator legality caches (tests compare warm vs cold refill). */
  clearTranslatorCachesForTest(): void {
    this.translator.clearCaches();
  }

  effectiveMasksSnapshot(): {
    actionMask: number[];
    unitMask: number[];
    boatRegions: number[];
    buildRegions: number[];
    spawnRegions: number[];
    targetMasks: number[];
  } {
    return {
      actionMask: Array.from(this.obs.actionMask),
      unitMask: Array.from(this.obs.unitMask),
      boatRegions: Array.from(this.obs.boatRegions),
      buildRegions: Array.from(this.obs.buildRegions),
      spawnRegions: Array.from(this.obs.spawnRegions),
      targetMasks: Array.from(this.obs.targetMasks),
    };
  }

  /** True when the current observation's action would emit a core intent. */
  wouldEmitIntent(action: ActionVec): boolean {
    return this.translator.wouldEmitIntent(
      this.game,
      this.me,
      action,
      this.slots,
      this.extractor.hasAdjacentWilderness(),
    );
  }

  /** Cached slot border/attack flags match a live sharesBorderWith scan. */
  slotFlagsMatchLive(): boolean {
    for (let i = 1; i < NUM_PLAYER_SLOTS; i++) {
      const p = this.slots[i];
      const cachedBorder = this.extractor.slotSharesBorderWith(i);
      const cachedAttack = this.extractor.slotCanAttackPlayer(i);
      if (p === null) {
        if (cachedBorder || cachedAttack) return false;
        continue;
      }
      if (cachedBorder !== this.me.sharesBorderWith(p)) return false;
      if (cachedAttack !== this.me.canAttackPlayer(p)) return false;
    }
    return true;
  }

  /**
   * Scan-based wilderness check (reference for cache equality tests).
   * Cardinal neighbors only, matching GameMap.neighbors / neighbors4.
   */
  hasAdjacentWildernessScan(): boolean {
    const map = this.game.map();
    const myID = this.me.smallID();
    for (const tile of this.me.borderTiles()) {
      if (map.ownerID(tile) !== myID) continue;
      for (const n of map.neighbors(tile)) {
        if (map.isLand(n) && !map.isImpassable(n) && !map.hasOwner(n)) {
          return true;
        }
      }
    }
    return false;
  }

  private elimProgress(): number {
    const aliveOpponents = this.game
      .players()
      .filter((p) => p.id() !== this.me.id() && p.isAlive()).length;
    const raw =
      (this.initialOpponentCount - aliveOpponents) / this.initialOpponentCount;
    return Math.max(0, Math.min(1, raw));
  }

  step(action: ActionVec): StepResult {
    const tSim = performance.now();
    const game = this.game;
    const wasAlive = this.me.isAlive();

    // Translate against the slots from the observation the policy saw.
    // Wilderness uses the extractor cache from that same extract.
    const intents = this.translator.translate(
      game,
      this.me,
      action,
      this.slots,
      this.extractor.hasAdjacentWilderness(),
    );
    const stamped = intents.map((i) => ({
      ...i,
      clientID: this.agentClientID,
    }));
    const intentCount = stamped.length;
    // NOOP is a successful empty action; other types fizzle when they
    // produce no core intents (illegal target / no troops / etc.).
    const actionAccepted =
      action.actionType === ACTION_NOOP || intentCount > 0;

    // Feed intents on the first turn of the decision window, then run the
    // window out with empty turns.
    this.runner.addTurn({
      turnNumber: this.turnNumber++,
      intents: stamped,
    });
    if (!this.runner.executeNextTick()) {
      throw new Error(this.fatalError ?? "executeNextTick failed");
    }
    this.runTicks(this.cfg.decisionInterval - 1);

    if (this.me.hasSpawned()) this.spawnedOnce = true;

    const landTiles = Math.max(1, game.numLandTiles());
    const tilesFrac = this.me.numTilesOwned() / landTiles;
    this.peakTilesFrac = Math.max(this.peakTilesFrac, tilesFrac);

    const terms: RewardTerms = {
      terminal: 0,
      spawn: 0,
      territory: 0,
      elimination: 0,
      total: 0,
    };
    let done = false;
    let win = false;
    let stageSuccess = false;
    let dead = false;
    let terminalCause: TerminalCause = "none";

    // Mutually exclusive terminals — exactly one cause and one terminal reward.
    // Spawn phase can linger until the Human places, so also treat maxTicks
    // without a spawn as no_spawn (not timeout/death).
    if (
      !this.spawnedOnce &&
      (!game.inSpawnPhase() || game.ticks() >= this.cfg.maxTicks)
    ) {
      done = true;
      terminalCause = "no_spawn";
      terms.terminal = REWARD_NO_SPAWN;
    } else {
      const winner = game.getWinner();
      const alive = this.me.isAlive();
      const winFrac = this.cfg.winTilesFrac;
      // Competency flag only — do not terminate. Ending the episode at a
      // tile fraction taught policies to stop short of the 80% FFA win.
      if (
        this.spawnedOnce &&
        winFrac !== undefined &&
        winFrac > 0 &&
        this.peakTilesFrac >= winFrac
      ) {
        stageSuccess = true;
      }
      if (winner !== null) {
        const agentWon =
          typeof winner !== "string" && winner.id() === this.me.id();
        if (agentWon) {
          done = true;
          win = true;
          stageSuccess = true;
          terminalCause = "win";
          terms.terminal = REWARD_WIN;
        } else if (!alive) {
          done = true;
          dead = true;
          terminalCause = "death";
          terms.terminal = REWARD_DEATH;
        } else {
          done = true;
          terminalCause = "loss_alive";
          terms.terminal = REWARD_LOSS_ALIVE;
        }
      } else if (wasAlive && !alive) {
        done = true;
        dead = true;
        terminalCause = "death";
        terms.terminal = REWARD_DEATH;
      } else if (game.ticks() >= this.cfg.maxTicks) {
        done = true;
        if (!alive) {
          dead = true;
          terminalCause = "death";
          terms.terminal = REWARD_DEATH;
        } else {
          terminalCause = "timeout";
          terms.terminal = REWARD_TIMEOUT;
        }
      }
    }

    // One-time spawn bonus: without it the territory signal is flat at 0
    // until the first conquest.
    if (!this.wasSpawned && this.spawnedOnce) {
      terms.spawn = REWARD_SPAWN;
    }
    this.wasSpawned = this.spawnedOnce;

    // Bounded state-potential shaping: territory + weighted elimination.
    // Kills remain a metric only (no direct REWARD_KILL).
    const elim = this.elimProgress();
    if (this.cfg.shaping > 0) {
      const territoryDelta = tilesFrac - this.prevTilesFrac;
      const elimDelta = elim - this.prevElimProgress;
      terms.territory = this.cfg.shaping * territoryDelta;
      terms.elimination =
        this.cfg.shaping * ELIMINATION_SHAPING_WEIGHT * elimDelta;
    }
    this.prevTilesFrac = tilesFrac;
    this.prevElimProgress = elim;

    terms.total =
      terms.terminal + terms.spawn + terms.territory + terms.elimination;

    this.lastStepMeta = {
      reward: terms.total,
      intentCount,
      actionAccepted,
      terminalCause,
      win,
      stageSuccess,
      dead,
      tilesFrac,
      peakTilesFrac: this.peakTilesFrac,
      kills: this.kills,
    };
    const simMs = performance.now() - tSim;
    const tObs = performance.now();
    const obs = this.extractObs();
    this.lastProfile.simMs = simMs;
    this.lastProfile.maskObsMs = performance.now() - tObs;

    const info: StepResult["info"] = {
      tick: game.ticks(),
      win,
      stageSuccess,
      dead,
      kills: this.kills,
      tilesFrac,
      spawned: this.spawnedOnce,
      hash: this.lastHash,
      terminalCause,
      peakTilesFrac: this.peakTilesFrac,
      rewardTerms: { ...terms },
      intentCount,
      actionAccepted,
      map: this.cfg.map,
      mapSize: this.cfg.mapSize,
      seed: this.cfg.seed,
    };
    if (this.cfg.enableDigest) {
      const boundary = this.strongStateDigest();
      info.digest = boundary.digest;
      info.tileDigest = boundary.tileDigest;
      info.obsDigest = boundary.obsDigest;
      info.stepDigest = boundary.stepDigest;
    }

    return {
      obs,
      reward: terms.total,
      done,
      info,
    };
  }

  currentHash(): number | null {
    return this.lastHash;
  }

  /**
   * Strong decision-boundary digest: tiles, obs/masks, last step reward,
   * intents, and terminal cause, plus the official 10-tick core hash.
   */
  strongStateDigest(): DecisionBoundary {
    return computeDecisionBoundary(
      this.game,
      this.obs,
      this.lastHash,
      this.lastStepMeta,
    );
  }

  gameTicks(): number {
    return this.game.ticks();
  }

  /**
   * Test-only unit census for oracle fixtures. Does not affect step/reset.
   */
  inspectActiveUnits(): Array<{
    id: number;
    type: string;
    ownerIsAgent: boolean;
    tile: number;
    underConstruction: boolean;
    trainStation: boolean;
    health: number;
  }> {
    const meId = this.me.id();
    const out: Array<{
      id: number;
      type: string;
      ownerIsAgent: boolean;
      tile: number;
      underConstruction: boolean;
      trainStation: boolean;
      health: number;
    }> = [];
    for (const u of this.game.units()) {
      if (!u.isActive()) continue;
      const owner = u.owner();
      out.push({
        id: u.id(),
        type: String(u.type()),
        ownerIsAgent: owner.isPlayer() && owner.id() === meId,
        tile: u.tile(),
        underConstruction: u.isUnderConstruction(),
        trainStation: u.hasTrainStation(),
        health: u.health(),
      });
    }
    return out;
  }

  /**
   * Official bomb-intercept census (BOMB_INDEX_INTERCEPT across all players).
   * Test-only; does not affect step/reset.
   */
  inspectBombIntercepts(): number {
    let n = 0;
    for (const p of this.game.allPlayers()) {
      const stats = this.game.stats().getPlayerStats(p);
      if (stats == null || stats.bombs === undefined) continue;
      for (const arr of Object.values(stats.bombs)) {
        if (arr !== undefined && arr.length > 2) {
          n += Number(arr[2]);
        }
      }
    }
    return n;
  }

  inspectTile(tile: number): {
    water: boolean;
    land: boolean;
    impassable: boolean;
    hasOwner: boolean;
    ownerIsAgent: boolean;
  } {
    const t = tile as TileRef;
    const owner = this.game.hasOwner(t) ? this.game.owner(t) : null;
    return {
      water: this.game.isWater(t),
      land: this.game.isLand(t),
      impassable: this.game.isImpassable(t),
      hasOwner: this.game.hasOwner(t),
      ownerIsAgent:
        owner !== null && owner.isPlayer() && owner.id() === this.me.id(),
    };
  }

  findBuildRegionForUnit(unit: number): number | null {
    const unitType = UNIT_HEAD_ORDER[unit];
    if (unitType === undefined) return null;
    return this.translator.findBuildRegionForUnit(this.game, this.me, unitType);
  }

  findNukeTileNearEnemySam(): { region: number; tile: number } | null {
    for (const sam of this.game.units(UnitType.SAMLauncher)) {
      if (!sam.isActive() || sam.isUnderConstruction()) continue;
      const owner = sam.owner();
      if (!owner.isPlayer() || owner.id() === this.me.id()) continue;
      const tile = sam.tile();
      if (this.me.canBuild(UnitType.AtomBomb, tile) === false) continue;
      return { region: this.translator.regionOfTile(this.game, tile), tile };
    }
    return null;
  }

  /** Game-scoped MIRV pile-on map. Isolation tests compare two envs. */
  recentMirvCooldownForTest(): Map<string, number> {
    return recentMirvTargetsFor(this.game);
  }

  /**
   * Full-resolution render of the real game map for the watch viewer.
   * Returns a color-index buffer (one byte per tile: 0=ocean, 1=impassable,
   * 2=unowned land, 3..=player index+3) plus a roster mapping indices to
   * player names/tiles so Python can build a legend and palette.
   */
  renderFrame(): {
    width: number;
    height: number;
    cells: Uint8Array;
    units: Array<{ x: number; y: number; owner: number; type: number }>;
    players: Array<{
      idx: number;
      name: string;
      tiles: number;
      alive: boolean;
      isAgent: boolean;
    }>;
    tick: number;
  } {
    const game = this.game;
    const map = game.map();
    const w = game.width();
    const h = game.height();
    const cells = new Uint8Array(w * h);

    // Roster: unique owner smallIDs in order of territory size.
    const players = game
      .players()
      .sort((a, b) => b.numTilesOwned() - a.numTilesOwned());
    const idxBySmallID = new Map<number, number>();
    players.forEach((p, i) => idxBySmallID.set(p.smallID(), i));

    for (let ref = 0; ref < w * h; ref++) {
      if (!map.isLand(ref)) {
        cells[ref] = 0; // ocean
      } else if (map.isImpassable(ref)) {
        cells[ref] = 1; // impassable background
      } else {
        const owner = map.ownerID(ref);
        if (owner === 0) {
          cells[ref] = 2; // unowned land
        } else {
          const idx = idxBySmallID.get(owner);
          cells[ref] = idx === undefined ? 2 : 3 + idx;
        }
        if (map.hasFallout(ref)) cells[ref] = 250; // fallout overlay
      }
    }

    const units: Array<{ x: number; y: number; owner: number; type: number }> =
      [];
    for (const u of game.units()) {
      if (!u.isActive()) continue;
      const t = u.tile();
      const owner = u.owner();
      units.push({
        x: t % w,
        y: (t / w) | 0,
        owner:
          owner.isPlayer() && idxBySmallID.has(owner.smallID())
            ? (idxBySmallID.get(owner.smallID()) as number)
            : -1,
        type: u.type() as number,
      });
    }

    return {
      width: w,
      height: h,
      cells,
      units,
      players: players.map((p, i) => ({
        idx: i,
        name: p.displayName(),
        tiles: p.numTilesOwned(),
        alive: p.isAlive(),
        isAgent: p.id() === this.me.id(),
      })),
      tick: game.ticks(),
    };
  }
}
