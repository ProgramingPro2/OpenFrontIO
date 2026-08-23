/**
 * Translates a factorized policy action (5 ints) into concrete game intents.
 *
 * Masked-legal actions should yield real intents whenever the core can accept
 * them. When the chosen region cannot host a valid build/boat, a deterministic
 * validated fallback scans other candidate tiles — never inventing invalid
 * core intents. Truly impossible actions still degrade to an empty list.
 */
import {
  Game,
  Player,
  UnitType,
} from "../../src/core/game/Game";
import { TileRef } from "../../src/core/game/GameMap";
import { canBuildTransportShip } from "../../src/core/game/TransportShipUtils";
import { Intent } from "../../src/core/Schemas";
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
  NUM_REGIONS,
  REGION_GRID,
  TROOP_FRACTIONS,
} from "./spec";

// Order matters: index into this list is the policy's unit head output.
export const UNIT_HEAD_ORDER: readonly UnitType[] = [
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
];

export interface ActionVec {
  actionType: number;
  target: number; // player slot
  region: number; // coarse region index
  quantity: number; // index into TROOP_FRACTIONS
  unit: number; // index into UNIT_HEAD_ORDER
}

function isNukeUnit(unitType: UnitType): boolean {
  return (
    unitType === UnitType.AtomBomb ||
    unitType === UnitType.HydrogenBomb ||
    unitType === UnitType.MIRV
  );
}

export class ActionTranslator {
  // Last tile that made a unit / boat-region legal. -1 = unknown. Cleared
  // on env reset. A hit still runs the same canBuild / canBuildTransportShip
  // predicate as the scan; a miss falls through to the original sampler.
  private static readonly NO_TILE: TileRef = -1 as TileRef;
  private buildTileCache: TileRef[] = UNIT_HEAD_ORDER.map(
    () => ActionTranslator.NO_TILE,
  );
  private boatDestCache: TileRef[] = Array.from(
    { length: NUM_REGIONS },
    () => ActionTranslator.NO_TILE,
  );
  private spawnTileCache: TileRef[] = Array.from(
    { length: NUM_REGIONS },
    () => ActionTranslator.NO_TILE,
  );
  private boundsW = -1;
  private boundsH = -1;
  private bounds = new Int32Array(NUM_REGIONS * 4);

  /** Drop incremental legality tiles. Call on every env.reset(). */
  clearCaches(): void {
    this.buildTileCache.fill(ActionTranslator.NO_TILE);
    this.boatDestCache.fill(ActionTranslator.NO_TILE);
    this.spawnTileCache.fill(ActionTranslator.NO_TILE);
  }

  /**
   * @param slots player slots from the observation the action was based on
   * @param bordersWilderness when provided, reuse ObsExtractor's maintained
   *   wilderness-adjacency cache. Omit only for standalone/scan fallback.
   */
  translate(
    game: Game,
    me: Player,
    action: ActionVec,
    slots: (Player | null)[],
    bordersWilderness?: boolean,
  ): Intent[] {
    switch (action.actionType) {
      case ACTION_SPAWN:
        return this.spawn(game, me, action.region);
      case ACTION_ATTACK:
        return this.attack(game, me, action, slots, bordersWilderness);
      case ACTION_RETREAT_ALL:
        return this.retreatAll(me);
      case ACTION_BUILD:
        return this.build(game, me, action);
      case ACTION_BOAT:
        return this.boat(game, me, action);
      case ACTION_ALLY:
        return this.ally(me, action, slots);
      case ACTION_BREAK_ALLY:
        return this.breakAlly(me, action, slots);
      case ACTION_EMBARGO:
        return this.embargo(me, action, slots);
      case ACTION_NOOP:
      default:
        return [];
    }
  }

  private regionBounds(game: Game, region: number): [number, number, number, number] {
    const w = game.width();
    const h = game.height();
    if (w !== this.boundsW || h !== this.boundsH) {
      this.boundsW = w;
      this.boundsH = h;
      for (let r = 0; r < NUM_REGIONS; r++) {
        const rx = r % REGION_GRID;
        const ry = (r / REGION_GRID) | 0;
        const o = r * 4;
        this.bounds[o] = Math.floor((rx * w) / REGION_GRID);
        this.bounds[o + 1] = Math.floor((ry * h) / REGION_GRID);
        this.bounds[o + 2] = Math.min(w, Math.floor(((rx + 1) * w) / REGION_GRID));
        this.bounds[o + 3] = Math.min(h, Math.floor(((ry + 1) * h) / REGION_GRID));
      }
    }
    const o = region * 4;
    return [this.bounds[o], this.bounds[o + 1], this.bounds[o + 2], this.bounds[o + 3]];
  }

  private isValidSpawnTile(game: Game, tile: TileRef): boolean {
    return (
      game.isLand(tile) && !game.hasOwner(tile) && !game.isImpassable(tile)
    );
  }

  private spawn(game: Game, me: Player, region: number): Intent[] {
    if (me.hasSpawned() && !game.inSpawnPhase()) return [];
    const tile = this.findSpawnTile(game, region);
    if (tile !== null) return [{ type: "spawn", tile }];
    // Deterministic fallback: first valid unowned land anywhere.
    for (let r = 0; r < NUM_REGIONS; r++) {
      if (r === region) continue;
      const t = this.findSpawnTile(game, r);
      if (t !== null) return [{ type: "spawn", tile: t }];
    }
    return [];
  }

  /**
   * Same spawn-tile sampler used by translate(). Masks that include a
   * region must be able to produce a spawn intent from this helper.
   */
  findSpawnTile(game: Game, region: number): TileRef | null {
    const [x0, y0, x1, y1] = this.regionBounds(game, region);
    const cx = (x0 + x1) >> 1;
    const cy = (y0 + y1) >> 1;
    const center = game.ref(cx, cy);
    if (this.isValidSpawnTile(game, center)) return center;
    const dx = Math.max(1, x1 - x0);
    const dy = Math.max(1, y1 - y0);
    for (let i = 0; i < 64; i++) {
      const tile = game.ref(x0 + ((i * 37) % dx), y0 + ((i * 53) % dy));
      if (this.isValidSpawnTile(game, tile)) return tile;
    }
    return null;
  }

  private attack(
    game: Game,
    me: Player,
    action: ActionVec,
    slots: (Player | null)[],
    bordersWilderness?: boolean,
  ): Intent[] {
    const frac = TROOP_FRACTIONS[action.quantity] ?? 0.15;
    const troops = Math.floor(me.troops() * frac);
    if (troops < 1) return [];

    // Slot 0 is the agent itself; reinterpreting target==0 as "wilderness"
    // (TerraNullius) unlocks expansion into uninhabited land. Attack execution
    // auto-targets ALL unowned land adjacent to ALL border tiles, so the
    // region head is irrelevant for ATTACK — only require that we border
    // some wilderness.
    if (action.target === 0) {
      const wild =
        bordersWilderness ?? this.scanBordersWilderness(game, me);
      if (!wild) {
        return [];
      }
      return [
        {
          type: "attack",
          targetID: game.terraNullius().id(),
          troops,
        },
      ];
    }

    const target = slots[action.target];
    if (target === null || target.id() === me.id()) return [];
    if (!me.canAttackPlayer(target) || !me.sharesBorderWith(target)) return [];
    return [{ type: "attack", targetID: target.id(), troops }];
  }

  /**
   * Scan fallback: true when any unowned passable land borders any of my
   * tiles. Prefer the ObsExtractor wilderness cache at translate time.
   */
  scanBordersWilderness(game: Game, me: Player): boolean {
    for (const t of me.borderTiles()) {
      for (const n of game.map().neighbors(t)) {
        if (
          game.isLand(n) &&
          !game.isImpassable(n) &&
          !game.hasOwner(n)
        ) {
          return true;
        }
      }
    }
    return false;
  }

  private retreatAll(me: Player): Intent[] {
    const intents: Intent[] = [];
    for (const atk of me.outgoingAttacks()) {
      if (!atk.retreating()) {
        intents.push({ type: "cancel_attack", attackID: atk.id() });
      }
    }
    return intents;
  }

  private build(game: Game, me: Player, action: ActionVec): Intent[] {
    const unitType = UNIT_HEAD_ORDER[action.unit] ?? UnitType.City;
    if (game.config().isUnitDisabled(unitType)) return [];
    const inRegion = this.findBuildInRegion(game, me, unitType, action.region);
    if (inRegion !== null) {
      return [{ type: "build_unit", unit: unitType, tile: inRegion }];
    }
    // Deterministic fallback: first owned tile where canBuild succeeds.
    const fallback = this.findBuildAnywhere(game, me, unitType);
    if (fallback !== null) {
      return [{ type: "build_unit", unit: unitType, tile: fallback }];
    }
    return [];
  }

  /**
   * Same in-region sampler used by translate(). A BUILD region mask bit
   * means this helper returns a tile for at least one affordable unit.
   *
   * Structures snap to owned land. Warships return a water tile in the
   * same component as an owned Port. Nukes return a legal target tile
   * (never friendly land).
   */
  findBuildInRegion(
    game: Game,
    me: Player,
    unitType: UnitType,
    region: number,
  ): TileRef | null {
    if (unitType === UnitType.Warship) {
      return this.findWarshipInRegion(game, me, region);
    }
    if (isNukeUnit(unitType)) {
      return this.findNukeInRegion(game, me, unitType, region);
    }
    const [x0, y0, x1, y1] = this.regionBounds(game, region);
    let tries = 0;
    for (let y = y0; y < y1 && tries < 64; y += 2) {
      for (let x = x0; x < x1 && tries < 64; x += 2) {
        const tile = game.ref(x, y);
        if (!game.hasOwner(tile)) continue;
        const owner = game.owner(tile);
        if (!owner.isPlayer() || owner.id() !== me.id()) continue;
        tries++;
        const buildable = me.canBuild(unitType, tile);
        if (buildable !== false) return buildable;
      }
    }
    return null;
  }

  /**
   * Same anywhere-fallback used by translate(). A unit is mask-legal
   * only when this helper finds a tile (`canBuild` succeeds).
   */
  findBuildAnywhere(
    game: Game,
    me: Player,
    unitType: UnitType,
  ): TileRef | null {
    if (unitType === UnitType.Warship) {
      return this.findWarshipAnywhere(game, me);
    }
    if (isNukeUnit(unitType)) {
      return this.findNukeAnywhere(game, me, unitType);
    }
    // Early-exit on first success. Same predicate as translate fallback
    // (no 256-tile cap, which could disagree with the mask).
    void game;
    for (const tile of me.tiles()) {
      const buildable = me.canBuild(unitType, tile);
      if (buildable !== false) return buildable;
    }
    return null;
  }

  /** Coarse region index for a tile. Matches ObsExtractor.regionOf. */
  regionOfTile(game: Game, tile: TileRef): number {
    const w = game.width();
    const h = game.height();
    const x = game.x(tile);
    const y = game.y(tile);
    const rx = Math.min(REGION_GRID - 1, ((x * REGION_GRID) / w) | 0);
    const ry = Math.min(REGION_GRID - 1, ((y * REGION_GRID) / h) | 0);
    return ry * REGION_GRID + rx;
  }

  /**
   * First region where findBuildInRegion succeeds for this unit.
   * Used when recording warship / nuke goldens so the region head aims.
   */
  findBuildRegionForUnit(
    game: Game,
    me: Player,
    unitType: UnitType,
  ): number | null {
    for (let r = 0; r < NUM_REGIONS; r++) {
      if (this.findBuildInRegion(game, me, unitType, r) !== null) return r;
    }
    return null;
  }

  private findWarshipInRegion(
    game: Game,
    me: Player,
    region: number,
  ): TileRef | null {
    const [x0, y0, x1, y1] = this.regionBounds(game, region);
    let tries = 0;
    for (let y = y0; y < y1 && tries < 64; y += 2) {
      for (let x = x0; x < x1 && tries < 64; x += 2) {
        const tile = game.ref(x, y);
        if (!game.isWater(tile)) continue;
        tries++;
        if (me.canBuild(UnitType.Warship, tile) !== false) return tile;
      }
    }
    return null;
  }

  private findNukeInRegion(
    game: Game,
    me: Player,
    unitType: UnitType,
    region: number,
  ): TileRef | null {
    const [x0, y0, x1, y1] = this.regionBounds(game, region);
    let tries = 0;
    for (let y = y0; y < y1 && tries < 64; y += 2) {
      for (let x = x0; x < x1 && tries < 64; x += 2) {
        const tile = game.ref(x, y);
        if (game.isImpassable(tile)) continue;
        if (unitType === UnitType.MIRV && !game.hasOwner(tile)) continue;
        if (game.hasOwner(tile)) {
          const owner = game.owner(tile);
          if (owner.isPlayer() && owner.id() === me.id()) continue;
        }
        tries++;
        if (me.canBuild(unitType, tile) !== false) return tile;
      }
    }
    return null;
  }

  private findWarshipAnywhere(game: Game, me: Player): TileRef | null {
    for (const port of me.units(UnitType.Port)) {
      if (!port.isActive() || port.isUnderConstruction()) continue;
      const tile = this.waterNearPort(game, me, port.tile());
      if (tile !== null) return tile;
    }
    return null;
  }

  private waterNearPort(
    game: Game,
    me: Player,
    portTile: TileRef,
  ): TileRef | null {
    const map = game.map();
    for (const n of map.neighbors(portTile)) {
      if (game.isWater(n) && me.canBuild(UnitType.Warship, n) !== false) {
        return n;
      }
    }
    const seen = new Set<TileRef>([portTile]);
    const queue: TileRef[] = [];
    for (const n of map.neighbors(portTile)) {
      if (game.isWater(n) && !seen.has(n)) {
        seen.add(n);
        queue.push(n);
      }
    }
    let i = 0;
    while (i < queue.length && i < 64) {
      const tile = queue[i++];
      if (me.canBuild(UnitType.Warship, tile) !== false) return tile;
      for (const n of map.neighbors(tile)) {
        if (game.isWater(n) && !seen.has(n)) {
          seen.add(n);
          queue.push(n);
        }
      }
    }
    return null;
  }

  private findNukeAnywhere(
    game: Game,
    me: Player,
    unitType: UnitType,
  ): TileRef | null {
    for (const p of game.players()) {
      if (p.id() === me.id() || me.isOnSameTeam(p)) continue;
      for (const tile of p.tiles()) {
        if (me.canBuild(unitType, tile) !== false) return tile;
      }
    }
    if (unitType === UnitType.MIRV) return null;
    for (let r = 0; r < NUM_REGIONS; r++) {
      const tile = this.findNukeInRegion(game, me, unitType, r);
      if (tile !== null) return tile;
    }
    return null;
  }

  private boat(game: Game, me: Player, action: ActionVec): Intent[] {
    const frac = TROOP_FRACTIONS[action.quantity] ?? 0.15;
    const troops = Math.floor(me.troops() * frac);
    if (troops < 1) return [];

    const inRegion = this.findBoatInRegion(game, me, action.region);
    if (inRegion !== null) {
      return [{ type: "boat", dst: inRegion, troops }];
    }
    // Deterministic fallback across regions that look boatable.
    for (let r = 0; r < NUM_REGIONS; r++) {
      if (r === action.region) continue;
      const dst = this.findBoatInRegion(game, me, r);
      if (dst !== null) return [{ type: "boat", dst, troops }];
    }
    return [];
  }

  /**
   * Same in-region sampler used by translate(). A BOAT region mask bit
   * means `canBuildTransportShip` succeeds on a sampled shoreline tile.
   */
  findBoatInRegion(
    game: Game,
    me: Player,
    region: number,
  ): TileRef | null {
    const [x0, y0, x1, y1] = this.regionBounds(game, region);
    let tries = 0;
    for (let y = y0; y < y1 && tries < 64; y += 2) {
      for (let x = x0; x < x1 && tries < 64; x += 2) {
        const tile = game.ref(x, y);
        if (!game.isLand(tile) || !game.isShoreline(tile)) continue;
        const owner = game.owner(tile);
        if (owner.isPlayer() && owner.id() === me.id()) continue;
        tries++;
        // Only emit intents the core would actually accept.
        if (canBuildTransportShip(game, me, tile) !== false) {
          return tile;
        }
      }
    }
    return null;
  }

  private ally(
    me: Player,
    action: ActionVec,
    slots: (Player | null)[],
  ): Intent[] {
    const target = slots[action.target];
    if (target === null || target.id() === me.id()) return [];
    // A reciprocal request accepts their pending incoming request (handled
    // core-side in AllianceRequestExecution).
    const incomingFromThem = me
      .incomingAllianceRequests()
      .some((r) => r.requestor().id() === target.id());
    if (!incomingFromThem && !me.canSendAllianceRequest(target)) return [];
    return [{ type: "allianceRequest", recipient: target.id() }];
  }

  private breakAlly(
    me: Player,
    action: ActionVec,
    slots: (Player | null)[],
  ): Intent[] {
    const target = slots[action.target];
    if (target === null || !me.isAlliedWith(target)) return [];
    return [{ type: "breakAlliance", recipient: target.id() }];
  }

  private embargo(
    me: Player,
    action: ActionVec,
    slots: (Player | null)[],
  ): Intent[] {
    const target = slots[action.target];
    if (target === null || target.id() === me.id()) return [];
    if (me.isAlliedWith(target)) return [];
    const action_ = me.hasEmbargoAgainst(target) ? "stop" : "start";
    return [{ type: "embargo", targetID: target.id(), action: action_ }];
  }

  /**
   * True when this action vector produces at least one core intent.
   * NOOP is the empty success; every other type must emit an intent.
   */
  wouldEmitIntent(
    game: Game,
    me: Player,
    action: ActionVec,
    slots: (Player | null)[],
    bordersWilderness?: boolean,
  ): boolean {
    if (action.actionType === ACTION_NOOP) return true;
    return (
      this.translate(game, me, action, slots, bordersWilderness).length > 0
    );
  }

  /**
   * Overwrite spawnRegions with regions where findSpawnTile succeeds.
   * Returns whether any region is actually spawnable.
   */
  fillEffectiveSpawnRegions(game: Game, me: Player, out: Uint8Array): boolean {
    void me;
    let any = false;
    for (let r = 0; r < NUM_REGIONS; r++) {
      // Extractor zeros mean no unowned land; the 65-sample cannot succeed.
      if (out[r] === 0) {
        this.spawnTileCache[r] = ActionTranslator.NO_TILE;
        continue;
      }
      const cached = this.spawnTileCache[r];
      if (
        cached !== ActionTranslator.NO_TILE &&
        this.isValidSpawnTile(game, cached)
      ) {
        any = true;
        continue;
      }
      const tile = this.findSpawnTile(game, r);
      this.spawnTileCache[r] = tile ?? ActionTranslator.NO_TILE;
      if (tile !== null) {
        any = true;
      } else {
        out[r] = 0;
      }
    }
    return any;
  }

  /**
   * Keep only boat regions where findBoatInRegion succeeds. Coarse
   * enemy-shore bits are an over-approximation; zeros stay zero.
   */
  fillEffectiveBoatRegions(game: Game, me: Player, boatRegions: Uint8Array): boolean {
    let any = false;
    for (let r = 0; r < NUM_REGIONS; r++) {
      if (boatRegions[r] === 0) {
        this.boatDestCache[r] = ActionTranslator.NO_TILE;
        continue;
      }
      const cached = this.boatDestCache[r];
      if (
        cached !== ActionTranslator.NO_TILE &&
        canBuildTransportShip(game, me, cached) !== false
      ) {
        any = true;
        continue;
      }
      const dest = this.findBoatInRegion(game, me, r);
      this.boatDestCache[r] = dest ?? ActionTranslator.NO_TILE;
      if (dest !== null) {
        any = true;
      } else {
        boatRegions[r] = 0;
      }
    }
    return any;
  }

  /**
   * Unit bits are units `canBuild` can place somewhere. Structure region
   * bits stay the extractor's owned-tile over-approx. Warship / nuke bits
   * add water or legal-target regions. Translate falls back to
   * findBuildAnywhere, so any (legal unit, set region) pair emits an intent.
   */
  fillEffectiveBuildMasks(
    game: Game,
    me: Player,
    unitMask: Uint8Array,
    buildRegions: Uint8Array,
  ): boolean {
    unitMask.fill(0);
    const gold = me.gold();
    const pendingStructures: number[] = [];
    const pendingWarship: number[] = [];
    const pendingNukes: number[] = [];
    for (let u = 0; u < UNIT_HEAD_ORDER.length; u++) {
      const unitType = UNIT_HEAD_ORDER[u];
      if (game.config().isUnitDisabled(unitType)) continue;
      const cost = game.unitInfo(unitType).cost(game, me);
      if (cost > gold) continue;
      if (unitType === UnitType.Warship) pendingWarship.push(u);
      else if (isNukeUnit(unitType)) pendingNukes.push(u);
      else pendingStructures.push(u);
    }
    // Reuse last placeable tile when canBuild still succeeds. Same
    // predicate as findBuildAnywhere; only a miss rescans owned tiles.
    for (let i = pendingStructures.length - 1; i >= 0; i--) {
      const u = pendingStructures[i];
      const cached = this.buildTileCache[u];
      if (
        cached !== ActionTranslator.NO_TILE &&
        me.canBuild(UNIT_HEAD_ORDER[u], cached) !== false
      ) {
        unitMask[u] = 1;
        pendingStructures.splice(i, 1);
      } else {
        this.buildTileCache[u] = ActionTranslator.NO_TILE;
      }
    }
    // One owned-tile pass, same canBuild predicate as findBuildAnywhere.
    // Early-exit once every affordable structure has a placeable tile.
    for (const tile of me.tiles()) {
      for (let i = pendingStructures.length - 1; i >= 0; i--) {
        const u = pendingStructures[i];
        const buildable = me.canBuild(UNIT_HEAD_ORDER[u], tile);
        if (buildable !== false) {
          unitMask[u] = 1;
          this.buildTileCache[u] = buildable;
          pendingStructures.splice(i, 1);
        }
      }
      if (pendingStructures.length === 0) break;
    }
    const anyStructure = UNIT_HEAD_ORDER.some(
      (t, u) =>
        unitMask[u] === 1 && t !== UnitType.Warship && !isNukeUnit(t),
    );
    if (!anyStructure) {
      buildRegions.fill(0);
    }

    if (pendingWarship.length > 0) {
      const readyPort = me
        .units(UnitType.Port)
        .some((p) => p.isActive() && !p.isUnderConstruction());
      if (!readyPort) {
        for (const u of pendingWarship) {
          this.buildTileCache[u] = ActionTranslator.NO_TILE;
        }
        pendingWarship.length = 0;
      }
    }
    if (pendingWarship.length > 0) {
      const u = pendingWarship[0];
      const cached = this.buildTileCache[u];
      let tile: TileRef | null = null;
      if (
        cached !== ActionTranslator.NO_TILE &&
        me.canBuild(UnitType.Warship, cached) !== false
      ) {
        tile = cached;
      } else {
        this.buildTileCache[u] = ActionTranslator.NO_TILE;
        tile = this.findWarshipAnywhere(game, me);
      }
      if (tile !== null) {
        unitMask[u] = 1;
        this.buildTileCache[u] = tile;
        buildRegions[this.regionOfTile(game, tile)] = 1;
      } else {
        this.buildTileCache[u] = ActionTranslator.NO_TILE;
      }
    }

    if (pendingNukes.length > 0) {
      const readySilo = me
        .units(UnitType.MissileSilo)
        .some(
          (s) =>
            s.isActive() && !s.isInCooldown() && !s.isUnderConstruction(),
        );
      if (!readySilo) {
        for (const u of pendingNukes) {
          this.buildTileCache[u] = ActionTranslator.NO_TILE;
        }
        pendingNukes.length = 0;
      }
    }
    if (pendingNukes.length > 0) {
      for (let i = pendingNukes.length - 1; i >= 0; i--) {
        const u = pendingNukes[i];
        const cached = this.buildTileCache[u];
        if (
          cached !== ActionTranslator.NO_TILE &&
          me.canBuild(UNIT_HEAD_ORDER[u], cached) !== false
        ) {
          unitMask[u] = 1;
          buildRegions[this.regionOfTile(game, cached)] = 1;
          pendingNukes.splice(i, 1);
        } else {
          this.buildTileCache[u] = ActionTranslator.NO_TILE;
        }
      }
      if (pendingNukes.length > 0) {
        outer: for (const p of game.players()) {
          if (p.id() === me.id() || me.isOnSameTeam(p)) continue;
          for (const tile of p.tiles()) {
            for (let i = pendingNukes.length - 1; i >= 0; i--) {
              const u = pendingNukes[i];
              if (me.canBuild(UNIT_HEAD_ORDER[u], tile) !== false) {
                unitMask[u] = 1;
                this.buildTileCache[u] = tile;
                buildRegions[this.regionOfTile(game, tile)] = 1;
                pendingNukes.splice(i, 1);
              }
            }
            if (pendingNukes.length === 0) break outer;
          }
        }
      }
      for (let i = pendingNukes.length - 1; i >= 0; i--) {
        const u = pendingNukes[i];
        if (UNIT_HEAD_ORDER[u] === UnitType.MIRV) continue;
        const tile = this.findNukeAnywhere(game, me, UNIT_HEAD_ORDER[u]);
        if (tile !== null) {
          unitMask[u] = 1;
          this.buildTileCache[u] = tile;
          buildRegions[this.regionOfTile(game, tile)] = 1;
          pendingNukes.splice(i, 1);
        }
      }
      for (const u of pendingNukes) {
        this.buildTileCache[u] = ActionTranslator.NO_TILE;
      }
    }

    const anyUnit = UNIT_HEAD_ORDER.some((_, u) => unitMask[u] === 1);
    if (!anyUnit) {
      buildRegions.fill(0);
      return false;
    }
    let anyRegion = false;
    for (let r = 0; r < NUM_REGIONS; r++) {
      if (buildRegions[r] !== 0) {
        anyRegion = true;
        break;
      }
    }
    if (!anyRegion) {
      // Owned-tile counter missed the tile canBuild used. Give the
      // region head a support so it does not become an all-ones safe mask.
      buildRegions[0] = 1;
    }
    return true;
  }
}
