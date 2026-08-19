import { EnvConfig } from "./spec";

export type NextConfig = Partial<EnvConfig> | string | null | undefined;

/**
 * Python-owned next episode identity, or the watch/ad-hoc `{seed}-r{n}`
 * fallback when nextConfigs is omitted or the slot is empty.
 */
export function resolveAutoReset(
  nextConfigs: NextConfig[] | undefined,
  index: number,
  currentSeed: string,
  resetCount: number,
): { spec: string | Partial<EnvConfig>; fallback: boolean } {
  const nxt = nextConfigs?.[index];
  if (typeof nxt === "string" && nxt.length > 0) {
    return { spec: nxt, fallback: false };
  }
  if (nxt && typeof nxt === "object") {
    return { spec: nxt, fallback: false };
  }
  return { spec: `${currentSeed}-r${resetCount}`, fallback: true };
}
