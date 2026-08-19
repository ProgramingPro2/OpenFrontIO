/**
 * Builds the fixed-size observation for the agent from live game state.
 *
 * Performance design: the sim emits packed per-tile state deltas every tick
 * (GameUpdateViewData.packedTileUpdates). We keep a full-res mirror of the
 * tile state plus per-bin counters, and apply those deltas incrementally, so
 * the per-decision cost is O(changed tiles) rather than O(map). A full pass
 * happens only on reset() and when the agent's alliance set changes (ally
 * plane reclassification).
 */
import {
  Game,
  Player,
  PlayerType,
  Relation,
  UnitType,
} from "../../src/core/game/Game";
import { GameMap, TileRef } from "../../src/core/game/GameMap";
import {
  NUM_ACTION_TYPES,
  NUM_PLAYER_SLOTS,
  NUM_QUANTITIES,
  NUM_REGIONS,
  NUM_UNIT_TYPES,
  PLAYER_FEATURES,
  GLOBAL_FEATURES,
  REGION_GRID,
  SPATIAL_CHANNELS,
  SPATIAL_SIZE,
  TARGET_MASKS_SIZE,
} from "./spec";

// Spatial channel indices.
export const CH_LAND = 0;
export const CH_MOUNTAIN = 1;
export const CH_MINE = 2;
export const CH_ENEMY = 3;
export const CH_ALLY = 4;
export const CH_MY_BORDER = 5;
export const CH_FALLOUT = 6;
export const CH_DEFENSE = 7;
export const CH_MY_STRUCTURES = 8;
export const CH_ENEMY_STRUCTURES = 9;

const STRUCTURE_TYPES: readonly UnitType[] = [
  UnitType.City,
  UnitType.DefensePost,
  UnitType.SAMLauncher,
  UnitType.MissileSilo,
  UnitType.Port,
  UnitType.Factory,
];

// Naval units share the structure planes: a warship or an inbound transport
// is exactly the kind of "enemy unit presence" the policy must see coming.
const NAVAL_TYPES: readonly UnitType[] = [
  UnitType.Warship,
  UnitType.TransportShip,
];

// Nukes are not painted onto spatial planes; do not scan them each extract.
const RASTER_UNIT_TYPES: readonly UnitType[] = [
  ...STRUCTURE_TYPES,
  ...NAVAL_TYPES,
];

const PLANE = SPATIAL_SIZE * SPATIAL_SIZE;

/** Sort key for opponent slot fill. Higher priority first, then smallID/id. */
export interface OpponentRankKey {
  priority: number;
  smallID: number;
  id: string;
}

/**
 * Priority matches the historical slotPriority formula: tiles, then
 * border-or-attackable, incoming attacks, allied, alliance-signal.
 */
export function slotPriorityScore(args: {
  tiles: number;
  sharesBorder: boolean;
  canAttack: boolean;
  incomingTroops: number;
  allied: boolean;
  allySignal: boolean;
}): number {
  let score = args.tiles;
  if (args.sharesBorder || args.canAttack) score += 1_000_000;
  if (args.incomingTroops > 0) score += 500_000;
  if (args.allied) score += 250_000;
  if (args.allySignal) score += 100_000;
  return score;
}

/** Deterministic descending rank: priority, then smallID, then id. */
export function compareOpponentRank(
  a: OpponentRankKey,
  b: OpponentRankKey,
): number {
  if (b.priority !== a.priority) return b.priority - a.priority;
  if (a.smallID !== b.smallID) return a.smallID - b.smallID;
  if (a.id < b.id) return -1;
  if (a.id > b.id) return 1;
  return 0;
}

/**
 * Fill limited opponent slots. When opponents overflow the individual
 * budget, keep `individualSlots - 1` targetable rows and leave the rest
 * for an untargetable aggregate.
 */
export function assignOpponentSlots<T extends OpponentRankKey>(
  metas: readonly T[],
  individualSlots: number,
): { picked: T[]; rest: T[]; overflow: boolean } {
  const sorted = metas.slice().sort(compareOpponentRank);
  const overflow = sorted.length > individualSlots;
  const n = overflow ? individualSlots - 1 : individualSlots;
  return {
    picked: sorted.slice(0, n),
    rest: overflow ? sorted.slice(n) : [],
    overflow,
  };
}

interface OpponentMeta extends OpponentRankKey {
  player: Player;
  sharesBorder: boolean;
  canAttack: boolean;
  incomingTroops: number;
  allied: boolean;
}

export interface ObsBuffers {
  spatial: Float32Array; // [SPATIAL_CHANNELS, SPATIAL_SIZE, SPATIAL_SIZE]
  players: Float32Array; // [NUM_PLAYER_SLOTS, PLAYER_FEATURES]
  global: Float32Array; // [GLOBAL_FEATURES]
  actionMask: Uint8Array; // [NUM_ACTION_TYPES]
  /** Row-major [NUM_ACTION_TYPES, NUM_PLAYER_SLOTS]; per-action legal targets. */
  targetMasks: Uint8Array;
  quantityMask: Uint8Array; // [NUM_QUANTITIES]
  unitMask: Uint8Array; // [NUM_UNIT_TYPES]
  spawnRegions: Uint8Array; // [NUM_REGIONS]
  buildRegions: Uint8Array; // [NUM_REGIONS]
  boatRegions: Uint8Array; // [NUM_REGIONS]
}

export function makeObsBuffers(): ObsBuffers {
  return {
    spatial: new Float32Array(SPATIAL_CHANNELS * PLANE),
    players: new Float32Array(NUM_PLAYER_SLOTS * PLAYER_FEATURES),
    global: new Float32Array(GLOBAL_FEATURES),
    actionMask: new Uint8Array(NUM_ACTION_TYPES),
    targetMasks: new Uint8Array(TARGET_MASKS_SIZE),
    quantityMask: new Uint8Array(NUM_QUANTITIES),
    unitMask: new Uint8Array(NUM_UNIT_TYPES),
    spawnRegions: new Uint8Array(NUM_REGIONS),
    buildRegions: new Uint8Array(NUM_REGIONS),
    boatRegions: new Uint8Array(NUM_REGIONS),
  };
}

// Per-bin dynamic counters (index by bin = by * SPATIAL_SIZE + bx).
// mine / enemy / ally are ownership fractions; border/fallout/defense are
// counts normalized by bin size at writeout.
class BinState {
  counts = new Float32Array(PLANE); // total tiles per bin (static)
  mine = new Float32Array(PLANE);
  enemy = new Float32Array(PLANE);
  ally = new Float32Array(PLANE);
  myBorder = new Float32Array(PLANE);
  fallout = new Float32Array(PLANE);
  defense = new Float32Array(PLANE);
}

export class ObsExtractor {
  // Static terrain planes, rebuilt on reset.
  private landBins = new Float32Array(PLANE);
  private mountainBins = new Float32Array(PLANE);

  private bins = new BinState();

  // Full-res mirror of the packed tile state (owner/fallout/defense bits).
  private mirror = new Uint16Array(0);

  // Per-region counters backing the three region masks.
  private regionMyTiles = new Int32Array(NUM_REGIONS);
  private regionUnownedLand = new Int32Array(NUM_REGIONS);
  private regionEnemyShore = new Int32Array(NUM_REGIONS);

  private mapW = 0;
  private mapH = 0;
  private mySmallID = 0;
  private allySmallIDs = new Set<number>();
  // Neighbor scratch for border maintenance (N, S, W, E offsets).
  private neighborScratch: TileRef[] = [0, 0, 0, 0];
  // Per-tile "my tile borders unowned passable land" bits + count. Matches
  // AgentEnv.hasAdjacentWildernessScan (borderTiles + neighbors4).
  private wildAdj = new Uint8Array(0);
  private wildAdjCount = 0;
  // Per-slot flags computed once in extract() and reused for player
  // features + AgentEnv target masks. Slot 0 (self) and the overflow
  // aggregate stay false / untargetable.
  private slotSharesBorder = new Uint8Array(NUM_PLAYER_SLOTS);
  private slotCanAttack = new Uint8Array(NUM_PLAYER_SLOTS);

  /** Full rebuild of every maintained structure. Call once per game. */
  buildStatic(game: Game, me: Player): void {
    const map = game.map();
    this.mapW = game.width();
    this.mapH = game.height();
    this.mySmallID = me.smallID();
    this.allySmallIDs = new Set(me.allies().map((a) => a.smallID()));

    this.landBins.fill(0);
    this.mountainBins.fill(0);
    this.bins = new BinState();
    this.regionMyTiles.fill(0);
    this.regionUnownedLand.fill(0);
    this.regionEnemyShore.fill(0);

    const n = this.mapW * this.mapH;
    if (this.mirror.length !== n) this.mirror = new Uint16Array(n);
    if (this.wildAdj.length !== n) this.wildAdj = new Uint8Array(n);
    else this.wildAdj.fill(0);
    this.wildAdjCount = 0;
    const state = map.tileStateBuffer();
    this.mirror.set(state);

    for (let ref = 0; ref < n; ref++) {
      const bin = this.binOf(ref);
      this.bins.counts[bin]++;
      const land = map.isLand(ref);
      if (land) {
        this.landBins[bin]++;
        const t = map.terrainType(ref);
        if (t === 1 || t === 2) this.mountainBins[bin]++; // Highland/Mountain
      }
      this.accumulateInitial(game, ref, this.mirror[ref], land);
    }
    for (let i = 0; i < PLANE; i++) {
      if (this.bins.counts[i] > 0) {
        this.landBins[i] /= this.bins.counts[i];
        this.mountainBins[i] /= this.bins.counts[i];
      }
    }
    for (let ref = 0; ref < n; ref++) {
      this.refreshWilderness(map, ref);
    }
  }

  /** True iff any owned tile borders unowned passable land (ATTACK wilderness). */
  hasAdjacentWilderness(): boolean {
    return this.wildAdjCount > 0;
  }

  private isWilderness(map: GameMap, ref: TileRef): boolean {
    return map.isLand(ref) && !map.isImpassable(ref) && !map.hasOwner(ref);
  }

  private computeWildAdj(map: GameMap, ref: TileRef): boolean {
    if (map.ownerID(ref) !== this.mySmallID) return false;
    const nCount = map.neighbors4(ref, this.neighborScratch);
    for (let j = 0; j < nCount; j++) {
      if (this.isWilderness(map, this.neighborScratch[j])) return true;
    }
    return false;
  }

  private setWildAdj(ref: TileRef, now: boolean): void {
    const was = this.wildAdj[ref] !== 0;
    if (now === was) return;
    this.wildAdj[ref] = now ? 1 : 0;
    this.wildAdjCount += now ? 1 : -1;
  }

  private refreshWilderness(map: GameMap, ref: TileRef): void {
    this.setWildAdj(ref, this.computeWildAdj(map, ref));
  }

  private accumulateInitial(
    game: Game,
    ref: TileRef,
    s: number,
    land: boolean,
  ): void {
    const bin = this.binOf(ref);
    const owner = s & 0x0fff;
    const region = this.regionOf(ref);
    if (owner !== 0) {
      if (owner === this.mySmallID) {
        this.bins.mine[bin]++;
        this.regionMyTiles[region]++;
        if (game.isBorder(ref)) {
          this.bins.myBorder[bin]++;
          this.mirror[ref] |= 0x8000; // border-counted sentinel bit
        }
      } else if (this.allySmallIDs.has(owner)) {
        this.bins.ally[bin]++;
      } else {
        this.bins.enemy[bin]++;
      }
    } else if (land) {
      this.regionUnownedLand[region]++;
    }
    if (s & 0x2000) this.bins.fallout[bin]++;
    if (s & 0x4000) this.bins.defense[bin]++;
    if (owner !== this.mySmallID && game.isShoreline(ref)) {
      this.regionEnemyShore[region]++;
    }
  }

  /**
   * Apply packed tile deltas from one tick. `packed` holds [ref, state]
   * uint32 pairs; state low 16 bits match tileStateBuffer bits.
   */
  applyTileUpdates(game: Game, packed: Uint32Array): void {
    const map = game.map();
    for (let i = 0; i < packed.length; i += 2) {
      const ref = packed[i];
      const s = packed[i + 1] & 0xffff;
      this.transition(game, ref, this.mirror[ref], s);
      this.mirror[ref] = s;
      // Ownership changes can flip border / wilderness status of neighbors.
      const nCount = map.neighbors4(ref, this.neighborScratch);
      const neigh = this.neighborScratch;
      const saved = [neigh[0], neigh[1], neigh[2], neigh[3]];
      for (let j = 0; j < nCount; j++) {
        this.refreshBorder(map, saved[j]);
      }
      this.refreshBorder(map, ref);
      this.refreshWilderness(map, ref);
      for (let j = 0; j < nCount; j++) {
        this.refreshWilderness(map, saved[j]);
      }
    }
  }

  private transition(game: Game, ref: TileRef, oldS: number, newS: number): void {
    if (oldS === newS) return;
    const bin = this.binOf(ref);
    const region = this.regionOf(ref);
    const map = game.map();
    const land = map.isLand(ref);

    // Subtract old classification.
    const oldOwner = oldS & 0x0fff;
    if (oldOwner !== 0) {
      if (oldOwner === this.mySmallID) {
        this.bins.mine[bin]--;
        this.regionMyTiles[region]--;
      } else if (this.allySmallIDs.has(oldOwner)) {
        this.bins.ally[bin]--;
      } else {
        this.bins.enemy[bin]--;
      }
    } else if (land) {
      this.regionUnownedLand[region]--;
    }
    if (oldS & 0x2000) this.bins.fallout[bin]--;
    if (oldS & 0x4000) this.bins.defense[bin]--;
    const oldShore =
      oldOwner !== this.mySmallID && land && map.isShoreline(ref);
    if (oldShore) this.regionEnemyShore[region]--;

    // Add new classification (border handled separately via refreshBorder).
    const newOwner = newS & 0x0fff;
    if (newOwner !== 0) {
      if (newOwner === this.mySmallID) {
        this.bins.mine[bin]++;
        this.regionMyTiles[region]++;
      } else if (this.allySmallIDs.has(newOwner)) {
        this.bins.ally[bin]++;
      } else {
        this.bins.enemy[bin]++;
      }
    } else if (land) {
      this.regionUnownedLand[region]++;
    }
    if (newS & 0x2000) this.bins.fallout[bin]++;
    if (newS & 0x4000) this.bins.defense[bin]++;
    const newShore =
      newOwner !== this.mySmallID && land && map.isShoreline(ref);
    if (newShore) this.regionEnemyShore[region]++;
  }

  /** Recompute my-border membership for one tile after ownership changed. */
  private refreshBorder(map: GameMap, ref: TileRef): void {
    const s = this.mirror[ref];
    const owner = s & 0x0fff;
    const bin = this.binOf(ref);
    // Border status for "my" tiles only. The counted state is tracked in the
    // mirror's spare high bit (bit 15 is unused by the game).
    const nowBorder = owner === this.mySmallID && owner !== 0 && map.isBorder(ref);
    const counted = (s & 0x8000) !== 0;
    if (nowBorder && !counted) {
      this.mirror[ref] = s | 0x8000;
      this.bins.myBorder[bin]++;
    } else if (!nowBorder && counted) {
      this.mirror[ref] = s & ~0x8000;
      this.bins.myBorder[bin]--;
    }
  }

  private binOf(ref: TileRef): number {
    const x = ref % this.mapW;
    const y = (ref / this.mapW) | 0;
    const bx = Math.min(SPATIAL_SIZE - 1, ((x * SPATIAL_SIZE) / this.mapW) | 0);
    const by = Math.min(SPATIAL_SIZE - 1, ((y * SPATIAL_SIZE) / this.mapH) | 0);
    return by * SPATIAL_SIZE + bx;
  }

  private regionOf(ref: TileRef): number {
    const x = ref % this.mapW;
    const y = (ref / this.mapW) | 0;
    const rx = Math.min(REGION_GRID - 1, ((x * REGION_GRID) / this.mapW) | 0);
    const ry = Math.min(REGION_GRID - 1, ((y * REGION_GRID) / this.mapH) | 0);
    return ry * REGION_GRID + rx;
  }

  slotSharesBorderWith(slot: number): boolean {
    return this.slotSharesBorder[slot] !== 0;
  }

  slotCanAttackPlayer(slot: number): boolean {
    return this.slotCanAttack[slot] !== 0;
  }

  private incomingTroopsFrom(me: Player, p: Player): number {
    let incomingTroops = 0;
    for (const atk of p.outgoingAttacks()) {
      const t = atk.target();
      if (t.isPlayer() && t.id() === me.id()) incomingTroops += atk.troops();
    }
    return incomingTroops;
  }

  /**
   * One opponent scan: border, attackability, incoming troops, and rank.
   * Callers must not re-invoke sharesBorderWith for the same extract.
   */
  private opponentMeta(me: Player, p: Player): OpponentMeta {
    const sharesBorder = me.sharesBorderWith(p);
    const canAttack = me.canAttackPlayer(p);
    const incomingTroops = this.incomingTroopsFrom(me, p);
    const allied = me.isAlliedWith(p);
    const incomingAlly = me
      .incomingAllianceRequests()
      .some((r) => r.requestor().id() === p.id());
    const allySignal = incomingAlly || me.canSendAllianceRequest(p);
    return {
      player: p,
      priority: slotPriorityScore({
        tiles: p.numTilesOwned(),
        sharesBorder,
        canAttack,
        incomingTroops,
        allied,
        allySignal,
      }),
      smallID: p.smallID(),
      id: p.id(),
      sharesBorder,
      canAttack,
      incomingTroops,
      allied,
    };
  }

  /**
   * Write the full observation for `me` into `out`.
   * Returns the ordered list of players placed into slots (slot 0 = me).
   * @param maxTicks episode tick cap used to normalize global[0] into [0,1]
   */
  extract(
    game: Game,
    me: Player,
    out: ObsBuffers,
    maxTicks: number,
  ): (Player | null)[] {
    // Alliance changes reclassify tiles between ally/enemy planes; rebuild.
    const alliesNow = new Set(me.allies().map((a) => a.smallID()));
    if (
      alliesNow.size !== this.allySmallIDs.size ||
      ![...alliesNow].every((id) => this.allySmallIDs.has(id))
    ) {
      this.buildStatic(game, me);
    }

    const spatial = out.spatial;
    spatial.set(this.landBins, CH_LAND * PLANE);
    spatial.set(this.mountainBins, CH_MOUNTAIN * PLANE);
    const b = this.bins;
    for (let i = 0; i < PLANE; i++) {
      const c = b.counts[i] > 0 ? b.counts[i] : 1;
      spatial[CH_MINE * PLANE + i] = b.mine[i] / c;
      spatial[CH_ENEMY * PLANE + i] = b.enemy[i] / c;
      spatial[CH_ALLY * PLANE + i] = b.ally[i] / c;
      spatial[CH_MY_BORDER * PLANE + i] = b.myBorder[i] / c;
      spatial[CH_FALLOUT * PLANE + i] = b.fallout[i] / c;
      spatial[CH_DEFENSE * PLANE + i] = b.defense[i] / c;
    }
    spatial.fill(0, CH_MY_STRUCTURES * PLANE, (CH_MY_STRUCTURES + 1) * PLANE);
    spatial.fill(
      0,
      CH_ENEMY_STRUCTURES * PLANE,
      (CH_ENEMY_STRUCTURES + 1) * PLANE,
    );

    // Rasterize units: structures and naval units into mine/enemy planes.
    const mySmallID = me.smallID();
    for (const unit of game.units(...RASTER_UNIT_TYPES)) {
      if (!unit.isActive()) continue;
      const bin = this.binOf(unit.tile());
      const owner = unit.owner();
      const isMine = owner.isPlayer() && owner.smallID() === mySmallID;
      const ch = isMine ? CH_MY_STRUCTURES : CH_ENEMY_STRUCTURES;
      spatial[ch * PLANE + bin] = Math.min(1, spatial[ch * PLANE + bin] + 0.34);
    }

    // Region masks from counters. Semantically active for spawn/build/boat
    // only; ATTACK ignores region (global wilderness / player targeting).
    for (let r = 0; r < NUM_REGIONS; r++) {
      out.spawnRegions[r] = this.regionUnownedLand[r] > 0 ? 1 : 0;
      out.buildRegions[r] = this.regionMyTiles[r] > 0 ? 1 : 0;
      out.boatRegions[r] = this.regionEnemyShore[r] > 0 ? 1 : 0;
    }

    // Player slots: 0 = me, then others by action priority then territory.
    // When opponents exceed 15, keep 14 individually targetable slots and
    // write an untargetable aggregate into the last slot (null player).
    // Border / attackability is computed once per opponent and reused.
    this.slotSharesBorder.fill(0);
    this.slotCanAttack.fill(0);
    const metas: OpponentMeta[] = [];
    for (const p of game.players()) {
      if (p.id() === me.id()) continue;
      metas.push(this.opponentMeta(me, p));
    }
    const assigned = assignOpponentSlots(metas, NUM_PLAYER_SLOTS - 1);
    const slots: (Player | null)[] = [me, ...assigned.picked.map((m) => m.player)];
    while (slots.length < NUM_PLAYER_SLOTS) slots.push(null);

    const landTiles = Math.max(1, game.numLandTiles());
    const players = out.players;
    players.fill(0);
    this.writeSelfFeatures(players, me, landTiles);
    for (let i = 0; i < assigned.picked.length; i++) {
      const meta = assigned.picked[i];
      const slot = i + 1;
      this.slotSharesBorder[slot] = meta.sharesBorder ? 1 : 0;
      this.slotCanAttack[slot] = meta.canAttack ? 1 : 0;
      this.writePlayerFeatures(players, slot, me, meta, landTiles);
    }
    if (assigned.rest.length > 0) {
      this.writeAggregate(players, NUM_PLAYER_SLOTS - 1, me, assigned.rest, landTiles);
    }

    const g = out.global;
    let myIncoming = 0;
    for (const atk of me.incomingAttacks()) myIncoming += atk.troops();
    const allPlayers = game.players();
    const aliveCount = allPlayers.filter((p) => p.isAlive()).length;
    const tickProgress = game.ticks() / Math.max(1, maxTicks);
    g[0] = Math.max(0, Math.min(1, tickProgress));
    g[1] = game.inSpawnPhase() ? 1 : 0;
    g[2] = me.numTilesOwned() / landTiles;
    g[3] = Math.log1p(me.troops()) / 15;
    g[4] = Math.log1p(Number(me.gold())) / 20;
    g[5] = Math.log1p(myIncoming) / 15;
    g[6] = aliveCount / Math.max(1, allPlayers.length);
    g[7] = Math.log1p(landTiles) / 15;
    g[8] = me.units(UnitType.Port).length > 0 ? 1 : 0;
    g[9] = game.getWinner() !== null ? 1 : 0;

    return slots;
  }

  private writeSelfFeatures(
    players: Float32Array,
    me: Player,
    landTiles: number,
  ): void {
    const incomingTroops = this.incomingTroopsFrom(me, me);
    const base = 0;
    players[base + 0] = 1;
    players[base + 1] = 1;
    players[base + 2] = me.numTilesOwned() / landTiles;
    players[base + 3] = Math.log1p(me.troops()) / 15;
    players[base + 4] = Math.log1p(Number(me.gold())) / 20;
    players[base + 5] = me.isAlive() ? 1 : 0;
    players[base + 6] = me.type() === PlayerType.Nation ? 1 : 0;
    players[base + 7] = me.type() === PlayerType.Bot ? 1 : 0;
    players[base + 8] = 0;
    players[base + 9] = 0;
    players[base + 10] = Math.log1p(incomingTroops) / 15;
    players[base + 11] = 0;
    players[base + 12] = Relation.Friendly / 3;
    players[base + 13] = me.hasSpawned() ? 1 : 0;
  }

  private writePlayerFeatures(
    players: Float32Array,
    slot: number,
    me: Player,
    meta: OpponentMeta,
    landTiles: number,
  ): void {
    const p = meta.player;
    const base = slot * PLAYER_FEATURES;
    players[base + 0] = 1;
    players[base + 1] = 0;
    players[base + 2] = p.numTilesOwned() / landTiles;
    players[base + 3] = Math.log1p(p.troops()) / 15;
    players[base + 4] = Math.log1p(Number(p.gold())) / 20;
    players[base + 5] = p.isAlive() ? 1 : 0;
    players[base + 6] = p.type() === PlayerType.Nation ? 1 : 0;
    players[base + 7] = p.type() === PlayerType.Bot ? 1 : 0;
    players[base + 8] = meta.allied ? 1 : 0;
    players[base + 9] = me.hasEmbargoAgainst(p) ? 1 : 0;
    players[base + 10] = Math.log1p(meta.incomingTroops) / 15;
    players[base + 11] = meta.sharesBorder ? 1 : 0;
    players[base + 12] = me.relation(p) / 3;
    players[base + 13] = p.hasSpawned() ? 1 : 0;
  }

  /**
   * Summary of opponents that did not fit in individual slots.
   * Both type flags are set (impossible for a real player) so the policy
   * can recognize leftover mass. The slot stays null / untargetable.
   */
  private writeAggregate(
    players: Float32Array,
    slot: number,
    me: Player,
    rest: OpponentMeta[],
    landTiles: number,
  ): void {
    let tiles = 0;
    let troops = 0;
    let gold = 0;
    let incoming = 0;
    let alive = 0;
    let spawned = 0;
    let nationTiles = 0;
    let botTiles = 0;
    let border = 0;
    let relationSum = 0;
    for (const meta of rest) {
      const p = meta.player;
      const owned = p.numTilesOwned();
      tiles += owned;
      troops += p.troops();
      gold += Number(p.gold());
      if (p.isAlive()) alive += 1;
      if (p.hasSpawned()) spawned += 1;
      if (p.type() === PlayerType.Nation) nationTiles += owned;
      if (p.type() === PlayerType.Bot) botTiles += owned;
      if (meta.sharesBorder) border += 1;
      relationSum += me.relation(p);
      incoming += meta.incomingTroops;
    }
    const base = slot * PLAYER_FEATURES;
    players[base + 0] = 1;
    players[base + 1] = 0;
    players[base + 2] = tiles / landTiles;
    players[base + 3] = Math.log1p(troops) / 15;
    players[base + 4] = Math.log1p(gold) / 20;
    players[base + 5] = alive > 0 ? 1 : 0;
    // Sentinel: a real player is never both Nation and Bot.
    players[base + 6] = 1;
    players[base + 7] = 1;
    players[base + 8] = 0;
    players[base + 9] = 0;
    players[base + 10] = Math.log1p(incoming) / 15;
    players[base + 11] = border > 0 ? 1 : 0;
    players[base + 12] = rest.length > 0 ? relationSum / rest.length / 3 : 0;
    players[base + 13] = spawned > 0 ? 1 : 0;
  }
}
