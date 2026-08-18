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
import { createGame } from "../../src/core/game/GameImpl";
import { createNationsForGame } from "../../src/core/game/NationCreation";
import {
  ErrorUpdate,
  GameUpdateType,
  GameUpdateViewData,
} from "../../src/core/game/GameUpdates";
import { GameRunner } from "../../src/core/GameRunner";
import { PseudoRandom } from "../../src/core/PseudoRandom";
import { GameConfig, GameStartInfo } from "../../src/core/Schemas";
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
  NUM_UNIT_TYPES,
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
  };
}

export class AgentEnv {
  readonly agentClientID = "agent";

  private game!: Game;
  private runner!: GameRunner;
  private me!: Player;
  private extractor = new ObsExtractor();
  private translator = new ActionTranslator();
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

  static async create(cfg: EnvConfig, terrain: TerrainCache): Promise<AgentEnv> {
    const env = new AgentEnv(cfg, terrain);
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

  async reset(seed?: string): Promise<ObsBuffers> {
    const cfg = { ...this.cfg, seed: seed ?? this.cfg.seed };
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
    // Let tribes/nations place their spawns before the agent's first decision.
    this.runTicks(5);
    this.initialOpponentCount = Math.max(
      1,
      this.game.players().filter((p) => p.id() !== this.me.id()).length,
    );
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
    this.slots = this.extractor.extract(
      this.game,
      this.me,
      this.obs,
      this.cfg.maxTicks,
    );
    this.fillMasks(this.slots);
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
      am[ACTION_SPAWN] = 1;
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
      if (me.canAttackPlayer(p) && me.sharesBorderWith(p)) {
        tm[attackRow + i] = 1;
        anyAttackTarget = true;
      }

      const incomingFromThem = me
        .incomingAllianceRequests()
        .some((r) => r.requestor().id() === p.id());
      if (me.canSendAllianceRequest(p) || incomingFromThem) {
        tm[allyRow + i] = 1;
        anyAllyTarget = true;
      }

      if (me.isAlliedWith(p)) {
        tm[breakRow + i] = 1;
        anyBreakTarget = true;
      } else {
        // Embargo toggle is meaningful against non-allied opponents.
        tm[embargoRow + i] = 1;
        anyEmbargoTarget = true;
      }
    }

    const um = this.obs.unitMask;
    let anyUnit = false;
    const gold = me.gold();
    for (let u = 0; u < NUM_UNIT_TYPES; u++) {
      const unitType = UNIT_HEAD_ORDER[u];
      if (game.config().isUnitDisabled(unitType)) continue;
      const cost = game.unitInfo(unitType).cost(game, me);
      if (cost <= gold) {
        um[u] = 1;
        anyUnit = true;
      }
    }

    am[ACTION_ATTACK] = anyAttackTarget && anyQuantity ? 1 : 0;
    am[ACTION_RETREAT_ALL] = me.outgoingAttacks().length > 0 ? 1 : 0;
    am[ACTION_BUILD] = anyUnit && me.numTilesOwned() > 0 ? 1 : 0;
    // Transport ships launch from shoreline (canBuildTransportShip); a Port
    // is not required. Gating on Port made boat never-legal before build.
    am[ACTION_BOAT] =
      this.obs.boatRegions.some((v) => v === 1) && anyQuantity ? 1 : 0;
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
    const game = this.game;
    const wasAlive = this.me.isAlive();

    // Translate against the slots from the observation the policy saw.
    const intents = this.translator.translate(game, this.me, action, this.slots);
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

    return {
      obs: this.extractObs(),
      reward: terms.total,
      done,
      info: {
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
      },
    };
  }

  currentHash(): number | null {
    return this.lastHash;
  }

  gameTicks(): number {
    return this.game.ticks();
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
