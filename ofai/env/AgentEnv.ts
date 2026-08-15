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
  EnvConfig,
  NUM_ACTION_TYPES,
  NUM_PLAYER_SLOTS,
  NUM_UNIT_TYPES,
  REWARD_DEATH,
  REWARD_LOSS_ALIVE,
  REWARD_TIMEOUT_ALIVE,
  REWARD_WIN,
} from "./spec";

export interface StepResult {
  obs: ObsBuffers;
  reward: number;
  done: boolean;
  info: {
    tick: number;
    win: boolean;
    dead: boolean;
    kills: number;
    tilesFrac: number;
    spawned: boolean;
    hash: number | null;
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
  private prevKills = 0;
  private spawnedOnce = false;
  private wasSpawned = false;
  private slots: (Player | null)[] = [];
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
    this.prevKills = 0;
    this.spawnedOnce = false;
    this.wasSpawned = false;
    this.extractor.buildStatic(this.game, this.me);
    // Let tribes/nations place their spawns before the agent's first decision.
    this.runTicks(5);
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
    this.slots = this.extractor.extract(this.game, this.me, this.obs);
    this.fillMasks(this.slots);
    return this.obs;
  }

  private fillMasks(slots: (Player | null)[]): void {
    const am = this.obs.actionMask;
    am.fill(0);
    const game = this.game;
    const me = this.me;
    const spawned = me.hasSpawned();
    const inSpawn = game.inSpawnPhase();

    am[0] = 1; // noop always legal
    if (inSpawn && !spawned) {
      am[1] = 1; // spawn is the only meaningful action
      this.obs.targetMask.fill(0);
      this.obs.unitMask.fill(0);
      return;
    }
    if (!spawned) return; // dead or waiting; only noop

    const tm = this.obs.targetMask;
    tm.fill(0);
    let anyAttackTarget = false;
    let anyAllyTarget = false;
    let anyEmbargoTarget = false;
    for (let i = 1; i < NUM_PLAYER_SLOTS; i++) {
      const p = slots[i];
      if (p === null || !p.isAlive()) continue;
      const incomingFromThem = me
        .incomingAllianceRequests()
        .some((r) => r.requestor().id() === p.id());
      const usable =
        me.canAttackPlayer(p) ||
        me.canSendAllianceRequest(p) ||
        incomingFromThem ||
        me.isAlliedWith(p) ||
        !me.hasEmbargoAgainst(p);
      if (!usable) continue;
      tm[i] = 1;
      if (me.canAttackPlayer(p)) anyAttackTarget = true;
      if (me.canSendAllianceRequest(p) || incomingFromThem) {
        anyAllyTarget = true;
      }
      if (!me.isAlliedWith(p)) anyEmbargoTarget = true;
    }

    const um = this.obs.unitMask;
    um.fill(0);
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

    am[2] = anyAttackTarget ? 1 : 0;
    am[3] = me.outgoingAttacks().length > 0 ? 1 : 0;
    am[4] = anyUnit && me.numTilesOwned() > 0 ? 1 : 0;
    am[5] = this.obs.boatRegions.some((v) => v === 1) && me.troops() > 100 ? 1 : 0;
    am[6] = anyAllyTarget ? 1 : 0;
    am[7] = me.allies().length > 0 ? 1 : 0;
    am[8] = anyEmbargoTarget ? 1 : 0;
    void NUM_ACTION_TYPES;
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

    let reward = 0;
    let done = false;
    let win = false;
    let dead = false;

    // Spawn window ended without the agent ever taking land: the episode is
    // unrecoverable (isAlive() == tiles.size > 0) and would otherwise drift
    // for thousands of ticks as a ghost. Terminate immediately with a death
    // penalty so the policy gets a clean "spawn on land first" gradient
    // instead of a long tail of meaningless -1 returns.
    if (!this.spawnedOnce && !game.inSpawnPhase()) {
      done = true;
      dead = true;
      reward += REWARD_DEATH;
    }

    const winner = game.getWinner();
    if (winner !== null) {
      done = true;
      win = typeof winner !== "string" && winner.id() === this.me.id();
      if (win) reward += REWARD_WIN;
      else if (this.me.isAlive()) reward += REWARD_LOSS_ALIVE;
      else dead = true;
    }
    if (!done && wasAlive && !this.me.isAlive()) {
      done = true;
      dead = true;
      reward += REWARD_DEATH;
    }
    if (!done && game.ticks() >= this.cfg.maxTicks) {
      done = true;
      if (this.me.isAlive()) reward += REWARD_TIMEOUT_ALIVE;
      else {
        dead = true;
        reward += REWARD_DEATH;
      }
    }
    // Reward the spawn step explicitly: random policy flips a coin on
    // spawn/noop during the spawn window, and without a positive signal the
    // shaping gradient is flat (tilesFrac stays 0). +0.1 the first time we
    // actually own land.
    if (!this.wasSpawned && this.spawnedOnce) {
      reward += 0.1;
    }
    this.wasSpawned = this.spawnedOnce;

    // Per-kill bonus, normalized by starting opponent count.
    if (this.kills > this.prevKills) {
      const opponents = Math.max(1, game.allPlayers().length - 1);
      reward += (this.kills - this.prevKills) / opponents;
      this.prevKills = this.kills;
    }
    if (this.cfg.shaping > 0) {
      reward += this.cfg.shaping * (tilesFrac - this.prevTilesFrac);
    }
    this.prevTilesFrac = tilesFrac;

    return {
      obs: this.extractObs(),
      reward,
      done,
      info: {
        tick: game.ticks(),
        win,
        dead,
        kills: this.kills,
        tilesFrac,
        spawned: this.spawnedOnce,
        hash: this.lastHash,
      },
    };
  }

  currentHash(): number | null {
    return this.lastHash;
  }

  gameTicks(): number {
    return this.game.ticks();
  }
}
