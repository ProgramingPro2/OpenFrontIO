/**
 * Builds the fixed-size observation for the agent from live game state.
 *
 * Spatial planes are produced by a single pass over the map's packed tile
 * state buffer (ownerID bits 0-11, fallout bit 13, defense bit 14), area-
 * averaged into SPATIAL_SIZE x SPATIAL_SIZE bins. Terrain planes (land,
 * mountain) are static per game and cached per reset.
 *
 * Everything is written into caller-owned buffers so repeated steps allocate
 * nothing.
 */
import {
  Game,
  Player,
  PlayerType,
  Relation,
  UnitType,
} from "../../src/core/game/Game";
import { TileRef } from "../../src/core/game/GameMap";
import {
  NUM_PLAYER_SLOTS,
  NUM_REGIONS,
  PLAYER_FEATURES,
  GLOBAL_FEATURES,
  REGION_GRID,
  SPATIAL_CHANNELS,
  SPATIAL_SIZE,
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

const NUKE_TYPES: readonly UnitType[] = [
  UnitType.AtomBomb,
  UnitType.HydrogenBomb,
  UnitType.MIRV,
  UnitType.MIRVWarhead,
];

export interface ObsBuffers {
  spatial: Float32Array; // [SPATIAL_CHANNELS, SPATIAL_SIZE, SPATIAL_SIZE]
  players: Float32Array; // [NUM_PLAYER_SLOTS, PLAYER_FEATURES]
  global: Float32Array; // [GLOBAL_FEATURES]
  actionMask: Uint8Array; // [NUM_ACTION_TYPES]
  targetMask: Uint8Array; // [NUM_PLAYER_SLOTS]
  unitMask: Uint8Array; // [NUM_UNIT_TYPES]
  spawnRegions: Uint8Array; // [NUM_REGIONS]
  buildRegions: Uint8Array; // [NUM_REGIONS]
  boatRegions: Uint8Array; // [NUM_REGIONS]
}

export function makeObsBuffers(): ObsBuffers {
  return {
    spatial: new Float32Array(SPATIAL_CHANNELS * SPATIAL_SIZE * SPATIAL_SIZE),
    players: new Float32Array(NUM_PLAYER_SLOTS * PLAYER_FEATURES),
    global: new Float32Array(GLOBAL_FEATURES),
    actionMask: new Uint8Array(9),
    targetMask: new Uint8Array(NUM_PLAYER_SLOTS),
    unitMask: new Uint8Array(10),
    spawnRegions: new Uint8Array(NUM_REGIONS),
    buildRegions: new Uint8Array(NUM_REGIONS),
    boatRegions: new Uint8Array(NUM_REGIONS),
  };
}

export class ObsExtractor {
  // Static terrain planes, rebuilt on reset.
  private landBins = new Float32Array(SPATIAL_SIZE * SPATIAL_SIZE);
  private mountainBins = new Float32Array(SPATIAL_SIZE * SPATIAL_SIZE);
  private binCounts = new Float32Array(SPATIAL_SIZE * SPATIAL_SIZE);

  private mapW = 0;
  private mapH = 0;

  /** Rebuild static terrain planes. Call once per game (maps are per-reset). */
  buildStatic(game: Game): void {
    this.landBins.fill(0);
    this.mountainBins.fill(0);
    this.binCounts.fill(0);
    this.mapW = game.width();
    this.mapH = game.height();
    const map = game.map();
    const n = this.mapW * this.mapH;
    for (let ref = 0; ref < n; ref++) {
      const bin = this.binOf(ref);
      this.binCounts[bin]++;
      if (map.isLand(ref)) {
        this.landBins[bin]++;
        const t = map.terrainType(ref);
        // TerrainType: Plains, Highland, Mountain (plus water types).
        if (t === 1 || t === 2) this.mountainBins[bin]++;
      }
    }
    for (let i = 0; i < this.binCounts.length; i++) {
      if (this.binCounts[i] > 0) {
        this.landBins[i] /= this.binCounts[i];
        this.mountainBins[i] /= this.binCounts[i];
      }
    }
  }

  private binOf(ref: TileRef): number {
    const x = ref % this.mapW;
    const y = (ref / this.mapW) | 0;
    const bx = Math.min(SPATIAL_SIZE - 1, (x * SPATIAL_SIZE / this.mapW) | 0);
    const by = Math.min(SPATIAL_SIZE - 1, (y * SPATIAL_SIZE / this.mapH) | 0);
    return by * SPATIAL_SIZE + bx;
  }

  private regionOf(ref: TileRef): number {
    const x = ref % this.mapW;
    const y = (ref / this.mapW) | 0;
    const rx = Math.min(REGION_GRID - 1, (x * REGION_GRID / this.mapW) | 0);
    const ry = Math.min(REGION_GRID - 1, (y * REGION_GRID / this.mapH) | 0);
    return ry * REGION_GRID + rx;
  }

  /**
   * Write the full observation for `me` into `out`.
   * Returns the ordered list of players placed into slots (slot 0 = me).
   */
  extract(game: Game, me: Player, out: ObsBuffers): (Player | null)[] {
    const map = game.map();
    const state = map.tileStateBuffer();
    const mySmallID = me.smallID();
    const allies = new Set(me.allies().map((a) => a.smallID()));

    const dyn = new Float32Array(6 * SPATIAL_SIZE * SPATIAL_SIZE);
    const counts = new Float32Array(SPATIAL_SIZE * SPATIAL_SIZE);
    const n = this.mapW * this.mapH;
    for (let ref = 0; ref < n; ref++) {
      const s = state[ref];
      const owner = s & 0x0fff;
      const bin = this.binOf(ref);
      counts[bin]++;
      if (owner !== 0) {
        if (owner === mySmallID) {
          dyn[0 * SPATIAL_SIZE * SPATIAL_SIZE + bin]++; // mine
          if (map.isBorder(ref)) {
            dyn[3 * SPATIAL_SIZE * SPATIAL_SIZE + bin]++; // my border
          }
          out.buildRegions[this.regionOf(ref)] = 1;
        } else if (allies.has(owner)) {
          dyn[2 * SPATIAL_SIZE * SPATIAL_SIZE + bin]++; // ally
        } else {
          dyn[1 * SPATIAL_SIZE * SPATIAL_SIZE + bin]++; // enemy
        }
      } else if (map.isLand(ref)) {
        out.spawnRegions[this.regionOf(ref)] = 1;
      }
      if (s & 0x2000) dyn[4 * SPATIAL_SIZE * SPATIAL_SIZE + bin]++; // fallout
      if (s & 0x4000) dyn[5 * SPATIAL_SIZE * SPATIAL_SIZE + bin]++; // defense
      if (map.isShoreline(ref) && owner !== mySmallID) {
        out.boatRegions[this.regionOf(ref)] = 1;
      }
    }

    const spatial = out.spatial;
    const planeSize = SPATIAL_SIZE * SPATIAL_SIZE;
    spatial.set(this.landBins, CH_LAND * planeSize);
    spatial.set(this.mountainBins, CH_MOUNTAIN * planeSize);
    for (let i = 0; i < planeSize; i++) {
      const c = counts[i] > 0 ? counts[i] : 1;
      spatial[CH_MINE * planeSize + i] = dyn[0 * planeSize + i] / c;
      spatial[CH_ENEMY * planeSize + i] = dyn[1 * planeSize + i] / c;
      spatial[CH_ALLY * planeSize + i] = dyn[2 * planeSize + i] / c;
      spatial[CH_MY_BORDER * planeSize + i] = dyn[3 * planeSize + i] / c;
      spatial[CH_FALLOUT * planeSize + i] = dyn[4 * planeSize + i] / c;
      spatial[CH_DEFENSE * planeSize + i] = dyn[5 * planeSize + i] / c;
    }
    spatial.fill(0, CH_MY_STRUCTURES * planeSize, (CH_MY_STRUCTURES + 1) * planeSize);
    spatial.fill(0, CH_ENEMY_STRUCTURES * planeSize, (CH_ENEMY_STRUCTURES + 1) * planeSize);

    // Rasterize units: structures into mine/enemy planes.
    for (const unit of game.units(...STRUCTURE_TYPES, ...NUKE_TYPES)) {
      if (!unit.isActive()) continue;
      const bin = this.binOf(unit.tile());
      const owner = unit.owner();
      const isMine = owner.isPlayer() && owner.smallID() === mySmallID;
      if (STRUCTURE_TYPES.includes(unit.type())) {
        const ch = isMine ? CH_MY_STRUCTURES : CH_ENEMY_STRUCTURES;
        spatial[ch * planeSize + bin] = Math.min(
          1,
          spatial[ch * planeSize + bin] + 0.34,
        );
      }
    }

    // Player slots: 0 = me, then others sorted by tiles owned.
    const others = game
      .players()
      .filter((p) => p.id() !== me.id())
      .sort((a, b) => b.numTilesOwned() - a.numTilesOwned())
      .slice(0, NUM_PLAYER_SLOTS - 1);
    const slots: (Player | null)[] = [me, ...others];
    while (slots.length < NUM_PLAYER_SLOTS) slots.push(null);

    const landTiles = Math.max(1, game.numLandTiles());
    const players = out.players;
    players.fill(0);
    for (let i = 0; i < NUM_PLAYER_SLOTS; i++) {
      const p = slots[i];
      if (p === null) continue;
      const base = i * PLAYER_FEATURES;
      const rel = i === 0 ? Relation.Friendly : me.relation(p);
      let incomingTroops = 0;
      for (const atk of p.outgoingAttacks()) {
        const t = atk.target();
        if (t.isPlayer() && t.id() === me.id()) incomingTroops += atk.troops();
      }
      players[base + 0] = 1; // exists
      players[base + 1] = i === 0 ? 1 : 0; // is_self
      players[base + 2] = p.numTilesOwned() / landTiles;
      players[base + 3] = Math.log1p(p.troops()) / 15;
      players[base + 4] = Math.log1p(Number(p.gold())) / 20;
      players[base + 5] = p.isAlive() ? 1 : 0;
      players[base + 6] = p.type() === PlayerType.Nation ? 1 : 0;
      players[base + 7] = p.type() === PlayerType.Bot ? 1 : 0;
      players[base + 8] = i > 0 && me.isAlliedWith(p) ? 1 : 0;
      players[base + 9] = i > 0 && me.hasEmbargoAgainst(p) ? 1 : 0;
      players[base + 10] = Math.log1p(incomingTroops) / 15;
      players[base + 11] = i > 0 && me.sharesBorderWith(p) ? 1 : 0;
      players[base + 12] = rel / 3;
      players[base + 13] = p.hasSpawned() ? 1 : 0;
    }

    const g = out.global;
    const cfg = game.config();
    let myIncoming = 0;
    for (const atk of me.incomingAttacks()) myIncoming += atk.troops();
    g[0] = game.ticks() / Math.max(1, cfg.gameConfig().maxTimerValue ?? 0) || 0;
    g[1] = game.inSpawnPhase() ? 1 : 0;
    g[2] = me.numTilesOwned() / landTiles;
    g[3] = Math.log1p(me.troops()) / 15;
    g[4] = Math.log1p(Number(me.gold())) / 20;
    g[5] = Math.log1p(myIncoming) / 15;
    g[6] = game.players().filter((p) => p.isAlive()).length / NUM_PLAYER_SLOTS;
    g[7] = Math.log1p(landTiles) / 15;
    g[8] = me.units(UnitType.Port).length > 0 ? 1 : 0;
    g[9] = game.getWinner() !== null ? 1 : 0;

    return slots;
  }
}
