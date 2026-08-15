/**
 * Translates a factorized policy action (5 ints) into concrete game intents.
 * Invalid or impossible actions degrade to a no-op (empty intent list), which
 * the mask head should make rare during training.
 */
import {
  Game,
  Player,
  UnitType,
} from "../../src/core/game/Game";
import { TileRef } from "../../src/core/game/GameMap";
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
   */
  translate(
    game: Game,
    me: Player,
    action: ActionVec,
    slots: (Player | null)[],
  ): Intent[] {
    switch (action.actionType) {
      case ACTION_SPAWN:
        return this.spawn(game, me, action.region);
      case ACTION_ATTACK:
        return this.attack(game, me, action, slots);
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
    const [x0, y0, x1, y1] = this.regionBounds(game, region);
    // Prefer the region center, spiraling out is overkill: scan a bounded set
    // of candidates and pick the first valid land tile with no owner.
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
        return [{ type: "spawn", tile }];
      }
    }
    return [];
  }

  private attack(
    game: Game,
    me: Player,
    action: ActionVec,
    slots: (Player | null)[],
  ): Intent[] {
    const frac = TROOP_FRACTIONS[action.quantity] ?? 0.15;
    const troops = Math.floor(me.troops() * frac);
    if (troops < 1) return [];

    // Slot 0 is the agent itself; reinterpreting target==0 as "wilderness"
    // (TerraNullius) is what unlocks expansion into uninhabited land — the
    // core expansion mechanic. The region head selects where: we verify the
    // chosen region actually borders unowned land adjacent to us, else noop.
    if (action.target === 0) {
      if (!this.regionHasAdjacentWilderness(game, me, action.region)) {
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
    if (!me.canAttackPlayer(target)) return [];
    return [{ type: "attack", targetID: target.id(), troops }];
  }

  /**
   * True when the given coarse region contains unowned land that borders any
   * of my tiles — i.e. a wilderness invasion there would actually expand us.
   */
  private regionHasAdjacentWilderness(
    game: Game,
    me: Player,
    region: number,
  ): boolean {
    const [x0, y0, x1, y1] = this.regionBounds(game, region);
    const myID = me.smallID();
    for (let y = y0; y < y1; y++) {
      for (let x = x0; x < x1; x++) {
        const tile = game.ref(x, y);
        if (!game.isLand(tile) || game.isImpassable(tile)) continue;
        if (game.hasOwner(tile)) continue; // wilderness = unowned
        // Adjacent to one of my border tiles?
        for (const n of game.map().neighbors(tile)) {
          if (game.map().ownerID(n) === myID) return true;
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
    const [x0, y0, x1, y1] = this.regionBounds(game, action.region);
    // Scan my tiles in the region (subsampled) until canBuild accepts one.
    let tries = 0;
    for (let y = y0; y < y1 && tries < 48; y += 2) {
      for (let x = x0; x < x1 && tries < 48; x += 2) {
        const tile = game.ref(x, y);
        if (!game.hasOwner(tile)) continue;
        const owner = game.owner(tile);
        if (!owner.isPlayer() || owner.id() !== me.id()) continue;
        tries++;
        const buildable = me.canBuild(unitType, tile);
        if (buildable !== false) {
          return [{ type: "build_unit", unit: unitType, tile: buildable }];
        }
      }
    }
    return [];
  }

  private boat(game: Game, me: Player, action: ActionVec): Intent[] {
    const frac = TROOP_FRACTIONS[action.quantity] ?? 0.15;
    const troops = Math.floor(me.troops() * frac);
    if (troops < 1) return [];
    const [x0, y0, x1, y1] = this.regionBounds(game, action.region);
    let tries = 0;
    for (let y = y0; y < y1 && tries < 48; y += 2) {
      for (let x = x0; x < x1 && tries < 48; x += 2) {
        const tile = game.ref(x, y);
        if (!game.isLand(tile)) continue;
        if (!game.isShoreline(tile)) continue;
        const owner = game.owner(tile);
        if (owner.isPlayer() && owner.id() === me.id()) continue;
        tries++;
        return [{ type: "boat", dst: tile, troops }];
      }
    }
    return [];
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
    const action_ = me.hasEmbargoAgainst(target) ? "stop" : "start";
    return [{ type: "embargo", targetID: target.id(), action: action_ }];
  }
}
