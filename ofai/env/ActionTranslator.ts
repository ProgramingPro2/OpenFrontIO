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

export class ActionTranslator {
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
    const rx = region % REGION_GRID;
    const ry = (region / REGION_GRID) | 0;
    const w = game.width();
    const h = game.height();
    const x0 = Math.floor((rx * w) / REGION_GRID);
    const y0 = Math.floor((ry * h) / REGION_GRID);
    const x1 = Math.min(w, Math.floor(((rx + 1) * w) / REGION_GRID));
    const y1 = Math.min(h, Math.floor(((ry + 1) * h) / REGION_GRID));
    return [x0, y0, x1, y1];
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

  private findSpawnTile(game: Game, region: number): TileRef | null {
    const [x0, y0, x1, y1] = this.regionBounds(game, region);
    const cx = (x0 + x1) >> 1;
    const cy = (y0 + y1) >> 1;
    const candidates: TileRef[] = [game.ref(cx, cy)];
    for (let i = 0; i < 64; i++) {
      const x = x0 + ((i * 37) % Math.max(1, x1 - x0));
      const y = y0 + ((i * 53) % Math.max(1, y1 - y0));
      candidates.push(game.ref(x, y));
    }
    for (const tile of candidates) {
      if (
        game.isLand(tile) &&
        !game.hasOwner(tile) &&
        !game.isImpassable(tile)
      ) {
        return tile;
      }
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

  private findBuildInRegion(
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

  private findBuildAnywhere(
    game: Game,
    me: Player,
    unitType: UnitType,
  ): TileRef | null {
    let tries = 0;
    for (const tile of me.tiles()) {
      if (tries++ > 256) break;
      const buildable = me.canBuild(unitType, tile);
      if (buildable !== false) return buildable;
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

  private findBoatInRegion(
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
}
