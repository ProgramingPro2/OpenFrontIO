/**
 * Fixed observation batch layout for sequential and parallel backends.
 *
 * Float32 fields are packed first (4-byte aligned), then Uint8 fields.
 * Per-env slices are views into one buffer so extract/mask write in place
 * and stacking does not copy. Parallel workers share the same layout via
 * SharedArrayBuffer.
 */
import { BatchObs } from "./batchObs";
import { ObsBuffers } from "./ObsExtractor";
import {
  GLOBAL_FEATURES,
  NUM_ACTION_TYPES,
  NUM_PLAYER_SLOTS,
  NUM_QUANTITIES,
  NUM_REGIONS,
  NUM_UNIT_TYPES,
  PLAYER_FEATURES,
  SPATIAL_CHANNELS,
  SPATIAL_SIZE,
  TARGET_MASKS_SIZE,
} from "./spec";

export const SPATIAL_ELEMS = SPATIAL_CHANNELS * SPATIAL_SIZE * SPATIAL_SIZE;
export const PLAYER_ELEMS = NUM_PLAYER_SLOTS * PLAYER_FEATURES;
export const ACTION_STRIDE = 5;

export interface AttachedBatch {
  batch: BatchObs;
  envViews: ObsBuffers[];
}

export function f32ElemsPerEnv(): number {
  return SPATIAL_ELEMS + PLAYER_ELEMS + GLOBAL_FEATURES;
}

export function u8ElemsPerEnv(): number {
  return (
    NUM_ACTION_TYPES +
    TARGET_MASKS_SIZE +
    NUM_QUANTITIES +
    NUM_UNIT_TYPES +
    NUM_REGIONS * 3
  );
}

export function batchByteLength(k: number): number {
  return k * (f32ElemsPerEnv() * 4 + u8ElemsPerEnv());
}

export function actionsByteLength(k: number): number {
  return k * ACTION_STRIDE * 4;
}

export function attachBatch(buffer: ArrayBufferLike, k: number): AttachedBatch {
  const f32Count = k * f32ElemsPerEnv();
  const f32 = new Float32Array(buffer, 0, f32Count);
  let fOff = 0;
  const spatial = f32.subarray(fOff, fOff + k * SPATIAL_ELEMS);
  fOff += k * SPATIAL_ELEMS;
  const players = f32.subarray(fOff, fOff + k * PLAYER_ELEMS);
  fOff += k * PLAYER_ELEMS;
  const global = f32.subarray(fOff, fOff + k * GLOBAL_FEATURES);

  const u8 = new Uint8Array(buffer, f32.byteLength, k * u8ElemsPerEnv());
  let uOff = 0;
  const actionMask = u8.subarray(uOff, uOff + k * NUM_ACTION_TYPES);
  uOff += k * NUM_ACTION_TYPES;
  const targetMasks = u8.subarray(uOff, uOff + k * TARGET_MASKS_SIZE);
  uOff += k * TARGET_MASKS_SIZE;
  const quantityMask = u8.subarray(uOff, uOff + k * NUM_QUANTITIES);
  uOff += k * NUM_QUANTITIES;
  const unitMask = u8.subarray(uOff, uOff + k * NUM_UNIT_TYPES);
  uOff += k * NUM_UNIT_TYPES;
  const spawnRegions = u8.subarray(uOff, uOff + k * NUM_REGIONS);
  uOff += k * NUM_REGIONS;
  const buildRegions = u8.subarray(uOff, uOff + k * NUM_REGIONS);
  uOff += k * NUM_REGIONS;
  const boatRegions = u8.subarray(uOff, uOff + k * NUM_REGIONS);

  const batch: BatchObs = {
    spatial,
    players,
    global,
    actionMask,
    targetMasks,
    quantityMask,
    unitMask,
    spawnRegions,
    buildRegions,
    boatRegions,
  };

  const envViews: ObsBuffers[] = [];
  for (let i = 0; i < k; i++) {
    envViews.push({
      spatial: spatial.subarray(i * SPATIAL_ELEMS, (i + 1) * SPATIAL_ELEMS),
      players: players.subarray(i * PLAYER_ELEMS, (i + 1) * PLAYER_ELEMS),
      global: global.subarray(i * GLOBAL_FEATURES, (i + 1) * GLOBAL_FEATURES),
      actionMask: actionMask.subarray(
        i * NUM_ACTION_TYPES,
        (i + 1) * NUM_ACTION_TYPES,
      ),
      targetMasks: targetMasks.subarray(
        i * TARGET_MASKS_SIZE,
        (i + 1) * TARGET_MASKS_SIZE,
      ),
      quantityMask: quantityMask.subarray(
        i * NUM_QUANTITIES,
        (i + 1) * NUM_QUANTITIES,
      ),
      unitMask: unitMask.subarray(i * NUM_UNIT_TYPES, (i + 1) * NUM_UNIT_TYPES),
      spawnRegions: spawnRegions.subarray(i * NUM_REGIONS, (i + 1) * NUM_REGIONS),
      buildRegions: buildRegions.subarray(i * NUM_REGIONS, (i + 1) * NUM_REGIONS),
      boatRegions: boatRegions.subarray(i * NUM_REGIONS, (i + 1) * NUM_REGIONS),
    });
  }
  return { batch, envViews };
}

export function allocOwnedBatch(k: number): AttachedBatch {
  return attachBatch(new ArrayBuffer(batchByteLength(k)), k);
}

export function allocSharedBatch(k: number): {
  sab: SharedArrayBuffer;
  attached: AttachedBatch;
} {
  const sab = new SharedArrayBuffer(batchByteLength(k));
  return { sab, attached: attachBatch(sab, k) };
}

export function allocSharedActions(k: number): {
  sab: SharedArrayBuffer;
  view: Int32Array;
} {
  const sab = new SharedArrayBuffer(actionsByteLength(k));
  return { sab, view: new Int32Array(sab) };
}

/** Copy length-prefixed TCP action blob (5 i32 / env) into a shared i32 view. */
export function copyActionsFromBlob(
  blob: Buffer,
  dest: Int32Array,
  k: number,
): void {
  for (let i = 0; i < k; i++) {
    const src = i * 20;
    const dst = i * ACTION_STRIDE;
    dest[dst] = blob.readInt32LE(src);
    dest[dst + 1] = blob.readInt32LE(src + 4);
    dest[dst + 2] = blob.readInt32LE(src + 8);
    dest[dst + 3] = blob.readInt32LE(src + 12);
    dest[dst + 4] = blob.readInt32LE(src + 16);
  }
}
