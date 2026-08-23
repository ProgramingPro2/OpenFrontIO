/**
 * Reusable K-env observation batch used by EnvServer. Kept in a side-effect
 * free module so tests can import stackObs without starting a TCP listener.
 *
 * Training backends prefer attachBatch / allocOwnedBatch views so each env
 * writes directly into the batch. stackObs remains the copy fallback and
 * the oracle used by unit tests.
 */
import { DType } from "./framing";
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

export interface BatchObs {
  spatial: Float32Array;
  players: Float32Array;
  global: Float32Array;
  actionMask: Uint8Array;
  targetMasks: Uint8Array;
  quantityMask: Uint8Array;
  unitMask: Uint8Array;
  spawnRegions: Uint8Array;
  buildRegions: Uint8Array;
  boatRegions: Uint8Array;
}

export function allocBatchObs(k: number): BatchObs {
  const planeN = SPATIAL_CHANNELS * SPATIAL_SIZE * SPATIAL_SIZE;
  return {
    spatial: new Float32Array(k * planeN),
    players: new Float32Array(k * NUM_PLAYER_SLOTS * PLAYER_FEATURES),
    global: new Float32Array(k * GLOBAL_FEATURES),
    actionMask: new Uint8Array(k * NUM_ACTION_TYPES),
    targetMasks: new Uint8Array(k * TARGET_MASKS_SIZE),
    quantityMask: new Uint8Array(k * NUM_QUANTITIES),
    unitMask: new Uint8Array(k * NUM_UNIT_TYPES),
    spawnRegions: new Uint8Array(k * NUM_REGIONS),
    buildRegions: new Uint8Array(k * NUM_REGIONS),
    boatRegions: new Uint8Array(k * NUM_REGIONS),
  };
}

/** Stack per-env obs into one batch. Reuses `reuse` when K matches. */
export function stackObs(obsList: ObsBuffers[], reuse?: BatchObs): BatchObs {
  const k = obsList.length;
  const planeN = SPATIAL_CHANNELS * SPATIAL_SIZE * SPATIAL_SIZE;
  const batch =
    reuse !== undefined && reuse.spatial.length === k * planeN
      ? reuse
      : allocBatchObs(k);
  for (let i = 0; i < k; i++) {
    const o = obsList[i];
    batch.spatial.set(o.spatial, i * planeN);
    batch.players.set(o.players, i * NUM_PLAYER_SLOTS * PLAYER_FEATURES);
    batch.global.set(o.global, i * GLOBAL_FEATURES);
    batch.actionMask.set(o.actionMask, i * NUM_ACTION_TYPES);
    batch.targetMasks.set(o.targetMasks, i * TARGET_MASKS_SIZE);
    batch.quantityMask.set(o.quantityMask, i * NUM_QUANTITIES);
    batch.unitMask.set(o.unitMask, i * NUM_UNIT_TYPES);
    batch.spawnRegions.set(o.spawnRegions, i * NUM_REGIONS);
    batch.buildRegions.set(o.buildRegions, i * NUM_REGIONS);
    batch.boatRegions.set(o.boatRegions, i * NUM_REGIONS);
  }
  return batch;
}

/** Wire tensor table. Shape is flat [elems] to match the existing TCP schema. */
export function obsTensors(batch: BatchObs) {
  return {
    spatial: { dtype: "f32" as DType, data: batch.spatial },
    players: { dtype: "f32" as DType, data: batch.players },
    global: { dtype: "f32" as DType, data: batch.global },
    action_mask: { dtype: "u8" as DType, data: batch.actionMask },
    target_masks: { dtype: "u8" as DType, data: batch.targetMasks },
    quantity_mask: { dtype: "u8" as DType, data: batch.quantityMask },
    unit_mask: { dtype: "u8" as DType, data: batch.unitMask },
    spawn_regions: { dtype: "u8" as DType, data: batch.spawnRegions },
    build_regions: { dtype: "u8" as DType, data: batch.buildRegions },
    boat_regions: { dtype: "u8" as DType, data: batch.boatRegions },
  };
}
