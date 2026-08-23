/**
 * Test-only strong decision-boundary digest.
 *
 * The official core emits a player-unit hash every 10 ticks. Training uses
 * that as a cheap checksum. Differential replay and golden fixtures need a
 * stronger digest that also covers tile ownership, policy-visible
 * observations/masks, rewards, intent counts, and terminal causes.
 *
 * Not computed on the training path unless EnvConfig.enableDigest is set.
 */
import { createHash, type Hash } from "node:crypto";
import { Game, Player, TerraNullius } from "../../src/core/game/Game";
import { ObsBuffers } from "./ObsExtractor";
import { TerminalCause } from "./spec";

export interface StepDigestInput {
  reward: number;
  intentCount: number;
  actionAccepted: boolean;
  terminalCause: TerminalCause;
  win: boolean;
  stageSuccess: boolean;
  dead: boolean;
  tilesFrac: number;
  peakTilesFrac: number;
  kills: number;
}

export interface DecisionBoundary {
  tick: number;
  coreHash: number | null;
  tileDigest: string;
  obsDigest: string;
  stepDigest: string;
  digest: string;
  reward: number;
  intentCount: number;
  actionAccepted: boolean;
  terminalCause: TerminalCause;
  win: boolean;
  stageSuccess: boolean;
  tilesFrac: number;
}

const EMPTY_STEP: StepDigestInput = {
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

function sha256(): Hash {
  return createHash("sha256");
}

function hex(h: Hash): string {
  return h.digest("hex");
}

function writeU32(h: Hash, n: number): void {
  const buf = Buffer.allocUnsafe(4);
  buf.writeInt32LE(n | 0, 0);
  h.update(buf);
}

function writeF64(h: Hash, n: number): void {
  const buf = Buffer.allocUnsafe(8);
  buf.writeDoubleLE(n, 0);
  h.update(buf);
}

function writeStr(h: Hash, s: string): void {
  h.update(Buffer.from(s, "utf8"));
  h.update(Buffer.from([0]));
}

function isPlayer(x: Player | TerraNullius): x is Player {
  return typeof (x as Player).isPlayer === "function" && (x as Player).isPlayer();
}

function ownerSmallID(owner: Player | TerraNullius): number {
  return isPlayer(owner) ? owner.smallID() : 0;
}

/** Quantize floats so JSON/golden inspection stays human-readable later. */
export function quantizeReward(reward: number): number {
  return Math.round(reward * 1e6) / 1e6;
}

export function hashObsBuffers(obs: ObsBuffers): string {
  const h = sha256();
  h.update(Buffer.from(obs.spatial.buffer, obs.spatial.byteOffset, obs.spatial.byteLength));
  h.update(Buffer.from(obs.players.buffer, obs.players.byteOffset, obs.players.byteLength));
  h.update(Buffer.from(obs.global.buffer, obs.global.byteOffset, obs.global.byteLength));
  h.update(obs.actionMask);
  h.update(obs.targetMasks);
  h.update(obs.quantityMask);
  h.update(obs.unitMask);
  h.update(obs.spawnRegions);
  h.update(obs.buildRegions);
  h.update(obs.boatRegions);
  return hex(h);
}

export function hashTileState(game: Game): string {
  const h = sha256();
  const tiles = game.map().tileStateBuffer();
  h.update(Buffer.from(tiles.buffer, tiles.byteOffset, tiles.byteLength));
  writeU32(h, game.ticks());
  writeU32(h, game.inSpawnPhase() ? 1 : 0);
  writeU32(h, game.numLandTiles());

  const players = game.allPlayers().slice().sort((a, b) => a.smallID() - b.smallID());
  writeU32(h, players.length);
  for (const p of players) {
    writeU32(h, p.smallID());
    writeStr(h, p.id());
    writeU32(h, p.isAlive() ? 1 : 0);
    writeU32(h, p.hasSpawned() ? 1 : 0);
    writeU32(h, p.numTilesOwned());
    writeF64(h, p.troops());
    writeStr(h, p.gold().toString());
    const allies = p
      .alliances()
      .map((a) => a.other(p).smallID())
      .sort((a, b) => a - b);
    writeU32(h, allies.length);
    for (const id of allies) writeU32(h, id);
    const embargoes: number[] = [];
    for (const other of players) {
      if (other.id() === p.id()) continue;
      if (p.hasEmbargoAgainst(other)) embargoes.push(other.smallID());
    }
    embargoes.sort((a, b) => a - b);
    writeU32(h, embargoes.length);
    for (const id of embargoes) writeU32(h, id);
    const attacks = p.outgoingAttacks().slice();
    writeU32(h, attacks.length);
    for (const atk of attacks) {
      writeStr(h, atk.id());
      writeU32(h, ownerSmallID(atk.target()));
      writeF64(h, atk.troops());
      writeU32(h, atk.retreating() ? 1 : 0);
      writeU32(h, atk.isActive() ? 1 : 0);
    }
  }

  const units = game
    .units()
    .slice()
    .filter((u) => u.isActive())
    .sort((a, b) => a.id() - b.id());
  writeU32(h, units.length);
  for (const u of units) {
    writeU32(h, u.id());
    writeU32(h, u.type() as number);
    writeU32(h, ownerSmallID(u.owner()));
    writeU32(h, u.tile());
    writeU32(h, u.hash());
  }
  return hex(h);
}

export function hashStepMeta(step: StepDigestInput): string {
  const h = sha256();
  writeF64(h, quantizeReward(step.reward));
  writeU32(h, step.intentCount);
  writeU32(h, step.actionAccepted ? 1 : 0);
  writeStr(h, step.terminalCause);
  writeU32(h, step.win ? 1 : 0);
  writeU32(h, step.stageSuccess ? 1 : 0);
  writeU32(h, step.dead ? 1 : 0);
  writeF64(h, quantizeReward(step.tilesFrac));
  writeF64(h, quantizeReward(step.peakTilesFrac));
  writeU32(h, step.kills);
  return hex(h);
}

export function computeDecisionBoundary(
  game: Game,
  obs: ObsBuffers,
  coreHash: number | null,
  step: StepDigestInput = EMPTY_STEP,
): DecisionBoundary {
  const tileDigest = hashTileState(game);
  const obsDigest = hashObsBuffers(obs);
  const stepDigest = hashStepMeta(step);
  const h = sha256();
  writeU32(h, game.ticks());
  writeU32(h, coreHash ?? 0);
  writeU32(h, coreHash === null ? 0 : 1);
  writeStr(h, tileDigest);
  writeStr(h, obsDigest);
  writeStr(h, stepDigest);
  return {
    tick: game.ticks(),
    coreHash,
    tileDigest,
    obsDigest,
    stepDigest,
    digest: hex(h),
    reward: quantizeReward(step.reward),
    intentCount: step.intentCount,
    actionAccepted: step.actionAccepted,
    terminalCause: step.terminalCause,
    win: step.win,
    stageSuccess: step.stageSuccess,
    tilesFrac: quantizeReward(step.tilesFrac),
  };
}

export function boundaryRecord(b: DecisionBoundary): Record<string, unknown> {
  return {
    tick: b.tick,
    coreHash: b.coreHash,
    digest: b.digest,
    tileDigest: b.tileDigest,
    obsDigest: b.obsDigest,
    stepDigest: b.stepDigest,
    reward: b.reward,
    intentCount: b.intentCount,
    actionAccepted: b.actionAccepted,
    terminalCause: b.terminalCause,
    win: b.win,
    stageSuccess: b.stageSuccess,
    tilesFrac: b.tilesFrac,
  };
}
