/**
 * Replay-fidelity gates: strong decision-boundary digests, committed goldens,
 * and randomized official-core differential traces.
 *
 * Run: npx vitest run ofai/tests/fidelity.test.ts
 * Regenerate goldens: UPDATE_OFAI_GOLDEN=1 npx vitest run ofai/tests/fidelity.test.ts
 */
import { existsSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { NodeGameMapLoader } from "../../tests/perf/fullgame/NodeGameMapLoader";
import { ACTION_NOOP, ACTION_SPAWN } from "../env/spec";
import { doomsdayClockRequiredTiles } from "../../src/core/game/DoomsdayClock";
import { DoomsdayClockConfigSchema } from "../../src/core/Schemas";
import { TerrainCache } from "../env/TerrainCache";
import {
  assertBoundariesMatch,
  FEATURE_SCENARIOS,
  fixturePath,
  loadGolden,
  mulberry32,
  randomLegalAction,
  recordScenario,
  replayActions,
  SCENARIOS,
  testConfig,
  writeGolden,
} from "./fidelity";
import { AgentEnv } from "../env/AgentEnv";
import { boundaryRecord } from "../env/stateDigest";

const terrain = new TerrainCache(
  new NodeGameMapLoader(path.join(__dirname, "../../resources/maps")),
);

const UPDATE = process.env.UPDATE_OFAI_GOLDEN === "1";

describe("strong state digest", () => {
  it("is stable across two identical resets and differs after a spawn", async () => {
    const cfg = testConfig({
      map: "Halkidiki",
      mapSize: "Compact",
      nations: "disabled",
      bots: 0,
      seed: "digest-stable",
      maxTicks: 400,
    });
    const a = await AgentEnv.create(cfg, terrain);
    const b = await AgentEnv.create(cfg, terrain);
    expect(a.strongStateDigest().digest).toBe(b.strongStateDigest().digest);
    expect(a.currentHash()).toBe(b.currentHash());

    const spawnA = a.peekObs();
    let region = 0;
    for (let i = 0; i < spawnA.spawnRegions.length; i++) {
      if (spawnA.spawnRegions[i] === 1) {
        region = i;
        break;
      }
    }
    a.step({
      actionType: ACTION_SPAWN,
      target: 0,
      region,
      quantity: 2,
      unit: 0,
    });
    expect(a.strongStateDigest().digest).not.toBe(b.strongStateDigest().digest);
    expect(a.strongStateDigest().obsDigest).not.toBe(
      b.strongStateDigest().obsDigest,
    );
    expect(a.strongStateDigest().tileDigest).not.toBe(
      b.strongStateDigest().tileDigest,
    );
  }, 120000);

  it("includes the official 10-tick hash alongside the strong digest", async () => {
    const env = await AgentEnv.create(
      testConfig({
        map: "Halkidiki",
        mapSize: "Compact",
        nations: "disabled",
        bots: 0,
        seed: "digest-core-hash",
        maxTicks: 200,
      }),
      terrain,
    );
    const before = env.strongStateDigest();
    expect(before.coreHash).not.toBeNull();
    env.step({
      actionType: ACTION_NOOP,
      target: 0,
      region: 0,
      quantity: 0,
      unit: 0,
    });
    const after = env.strongStateDigest();
    expect(after.coreHash).not.toBeNull();
    expect(after.digest).not.toBe(before.digest);
  }, 120000);
});

describe("golden decision-boundary fixtures", () => {
  it.each(SCENARIOS.map((s) => [s.name, s] as const))(
    "replays %s against the committed golden",
    async (name, scenario) => {
      if (UPDATE) {
        writeGolden(await recordScenario(terrain, scenario));
      }
      if (!existsSync(fixturePath(name))) {
        throw new Error(
          `missing golden ${name}; generate with UPDATE_OFAI_GOLDEN=1`,
        );
      }
      const golden = loadGolden(name);
      expect(golden.name).toBe(name);
      expect(golden.actions.length).toBe(golden.boundaries.length);
      const replayed = await replayActions(
        terrain,
        golden.config,
        golden.actions,
      );
      expect(boundaryRecord(replayed.init).digest).toBe(golden.init.digest);
      expect(boundaryRecord(replayed.init).tileDigest).toBe(
        golden.init.tileDigest,
      );
      expect(boundaryRecord(replayed.init).obsDigest).toBe(
        golden.init.obsDigest,
      );
      assertBoundariesMatch(golden.boundaries, replayed.boundaries);
    },
    180000,
  );

  it("covers spawn, combat, boats, structures, diplomacy, nations, and terminals", () => {
    const names = SCENARIOS.map((s) => s.name);
    expect(names).toEqual([
      "spawn",
      "wilderness-combat",
      "player-combat",
      "boats",
      "structures",
      "diplomacy",
      "nations",
      "terminal-no-spawn",
      "terminal-timeout",
    ]);
    for (const name of names) {
      expect(existsSync(fixturePath(name))).toBe(true);
    }
    const noSpawn = loadGolden("terminal-no-spawn");
    expect(
      noSpawn.boundaries.some((b) => b.terminalCause === "no_spawn"),
    ).toBe(true);
    const timeout = loadGolden("terminal-timeout");
    expect(
      timeout.boundaries.some((b) => b.terminalCause === "timeout"),
    ).toBe(true);
    const spawn = loadGolden("spawn");
    expect(spawn.boundaries[0]?.intentCount).toBeGreaterThan(0);
    expect(spawn.boundaries.some((b) => (b.reward as number) > 0)).toBe(true);
    const actionOf = (name: string) =>
      (loadGolden(name).actions as number[][]).map((a) => a[0]);
    expect(actionOf("boats")).toContain(5);
    expect(actionOf("structures")).toContain(4);
    expect(actionOf("diplomacy").some((a) => a === 6 || a === 8)).toBe(true);
    expect(actionOf("wilderness-combat")).toContain(2);
    expect(loadGolden("nations").config.nations).toBe(3);
  });
});

describe("feature golden decision-boundary fixtures", () => {
  it.each(FEATURE_SCENARIOS.map((s) => [s.name, s] as const))(
    "replays %s against the committed golden",
    async (name, scenario) => {
      if (UPDATE) {
        writeGolden(await recordScenario(terrain, scenario));
      }
      if (!existsSync(fixturePath(name))) {
        throw new Error(
          `missing golden ${name}; generate with UPDATE_OFAI_GOLDEN=1`,
        );
      }
      const golden = loadGolden(name);
      expect(golden.name).toBe(name);
      expect(golden.actions.length).toBe(golden.boundaries.length);
      if (scenario.requireEvents !== undefined) {
        for (const event of scenario.requireEvents) {
          expect(golden.events ?? []).toContain(event);
        }
      }
      const replayed = await replayActions(
        terrain,
        golden.config,
        golden.actions,
      );
      expect(boundaryRecord(replayed.init).digest).toBe(golden.init.digest);
      expect(boundaryRecord(replayed.init).tileDigest).toBe(
        golden.init.tileDigest,
      );
      expect(boundaryRecord(replayed.init).obsDigest).toBe(
        golden.init.obsDigest,
      );
      assertBoundariesMatch(golden.boundaries, replayed.boundaries);
    },
    240000,
  );

  it("covers port/factory, rails/trains, trade, warships, nukes, and win", () => {
    const names = FEATURE_SCENARIOS.map((s) => s.name);
    expect(names).toEqual([
      "port-factory",
      "rails-trains",
      "trade-ships",
      "warships",
      "warships-combat",
      "nukes",
      "nukes-atom",
      "nukes-mirv",
      "sam-intercept",
      "win",
    ]);
    for (const name of names) {
      expect(existsSync(fixturePath(name))).toBe(true);
    }
    const eventsOf = (name: string) => loadGolden(name).events ?? [];
    expect(eventsOf("port-factory")).toEqual(
      expect.arrayContaining(["port_complete", "factory_complete"]),
    );
    expect(eventsOf("rails-trains")).toEqual(
      expect.arrayContaining([
        "city_complete",
        "factory_complete",
        "rail_station",
        "train",
      ]),
    );
    expect(eventsOf("trade-ships")).toEqual(
      expect.arrayContaining(["port_complete", "trade_ship"]),
    );
    expect(eventsOf("warships")).toEqual(
      expect.arrayContaining(["warship", "warship_moved"]),
    );
    expect(eventsOf("warships-combat")).toEqual(
      expect.arrayContaining(["warship", "warship_moved", "warship_combat"]),
    );
    expect(eventsOf("nukes")).toEqual(
      expect.arrayContaining(["hbomb_launch", "hbomb_impact"]),
    );
    expect(eventsOf("nukes-atom")).toEqual(
      expect.arrayContaining(["atom_launch", "atom_impact"]),
    );
    expect(eventsOf("nukes-mirv")).toEqual(
      expect.arrayContaining(["mirv_launch", "mirv_impact"]),
    );
    expect(eventsOf("sam-intercept")).toEqual(
      expect.arrayContaining(["atom_launch", "nuke_intercepted"]),
    );
    expect(eventsOf("win")).toContain("win");
    const win = loadGolden("win");
    expect(win.boundaries.some((b) => b.terminalCause === "win")).toBe(true);
    expect(win.boundaries.some((b) => b.win === true)).toBe(true);
  });

  it("replays nukes-mirv byte-identically twice in one process", async () => {
    const golden = loadGolden("nukes-mirv");
    const first = await replayActions(terrain, golden.config, golden.actions);
    const second = await replayActions(terrain, golden.config, golden.actions);
    expect(boundaryRecord(first.init).digest).toBe(golden.init.digest);
    expect(boundaryRecord(second.init).digest).toBe(golden.init.digest);
    assertBoundariesMatch(golden.boundaries, first.boundaries);
    assertBoundariesMatch(golden.boundaries, second.boundaries);
  }, 240000);
});

describe("randomized official-core differential traces", () => {
  const traces: Array<{ seed: string; map: string; bots: number; nations: EnvConfig["nations"]; steps: number }> = [
    { seed: "diff-a", map: "Halkidiki", bots: 1, nations: "disabled", steps: 20 },
    { seed: "diff-b", map: "FourIslands", bots: 2, nations: "disabled", steps: 16 },
    { seed: "diff-c", map: "FourIslands", bots: 2, nations: 2, steps: 12 },
  ];

  it.each(traces.map((t) => [t.seed, t] as const))(
    "keeps two official actors byte-equal on %s",
    async (_name, spec) => {
      const cfg = testConfig({
        map: spec.map,
        mapSize: spec.map === "Halkidiki" ? "Compact" : "Normal",
        nations: spec.nations,
        bots: spec.bots,
        seed: spec.seed,
        maxTicks: 2500,
      });
      const left = await AgentEnv.create(cfg, terrain);
      const right = await AgentEnv.create(cfg, terrain);
      expect(left.strongStateDigest().digest).toBe(right.strongStateDigest().digest);
      const rng = mulberry32(simpleSeed(spec.seed));
      let obs = left.peekObs();
      for (let i = 0; i < spec.steps; i++) {
        const action = randomLegalAction(obs, rng);
        const a = left.step(action);
        const b = right.step(action);
        const da = left.strongStateDigest();
        const db = right.strongStateDigest();
        expect(da.digest).toBe(db.digest);
        expect(da.tileDigest).toBe(db.tileDigest);
        expect(da.obsDigest).toBe(db.obsDigest);
        expect(da.stepDigest).toBe(db.stepDigest);
        expect(da.coreHash).toBe(db.coreHash);
        expect(a.info.terminalCause).toBe(b.info.terminalCause);
        expect(a.info.intentCount).toBe(b.info.intentCount);
        expect(a.reward).toBe(b.reward);
        expect(a.done).toBe(b.done);
        if (a.done) break;
        obs = a.obs;
      }
    },
    180000,
  );
});

describe("enableDigest wire field", () => {
  it("omits digest from step info unless enableDigest is set", async () => {
    const off = await AgentEnv.create(
      testConfig({
        map: "Halkidiki",
        mapSize: "Compact",
        nations: "disabled",
        bots: 0,
        seed: "digest-off",
        maxTicks: 200,
        enableDigest: false,
      }),
      terrain,
    );
    const rOff = off.step({
      actionType: ACTION_NOOP,
      target: 0,
      region: 0,
      quantity: 0,
      unit: 0,
    });
    expect(rOff.info.digest).toBeUndefined();

    const on = await AgentEnv.create(
      testConfig({
        map: "Halkidiki",
        mapSize: "Compact",
        nations: "disabled",
        bots: 0,
        seed: "digest-off",
        maxTicks: 200,
        enableDigest: true,
      }),
      terrain,
    );
    const rOn = on.step({
      actionType: ACTION_NOOP,
      target: 0,
      region: 0,
      quantity: 0,
      unit: 0,
    });
    expect(rOn.info.digest).toMatch(/^[0-9a-f]{64}$/);
    expect(rOn.info.tileDigest).toMatch(/^[0-9a-f]{64}$/);
    expect(rOn.info.obsDigest).toMatch(/^[0-9a-f]{64}$/);
  }, 120000);
});

describe("doomsday official constraint", () => {
  it("stays disabled by default and grace is not wire-configurable", async () => {
    const keys = Object.keys(DoomsdayClockConfigSchema.shape).sort();
    expect(keys).toEqual(["enabled", "speed"]);
    for (const speed of ["slow", "normal", "fast", "veryfast"] as const) {
      expect(doomsdayClockRequiredTiles(speed, 1000, 599)).toBe(0);
      expect(doomsdayClockRequiredTiles(speed, 1000, 600)).toBe(0);
    }
    expect(doomsdayClockRequiredTiles("veryfast", 1000, 700)).toBeGreaterThan(0);

    const off = await AgentEnv.create(
      testConfig({
        map: "Halkidiki",
        mapSize: "Compact",
        nations: "disabled",
        bots: 0,
        seed: "doomsday-default-off",
        maxTicks: 200,
      }),
      terrain,
    );
    let r = off.step({
      actionType: ACTION_NOOP,
      target: 0,
      region: 0,
      quantity: 0,
      unit: 0,
    });
    expect(off.peekSlots().some((p) => p !== null && p.inDoomsdayClock())).toBe(
      false,
    );

    const enabled = await AgentEnv.create(
      testConfig({
        map: "Halkidiki",
        mapSize: "Compact",
        nations: 2,
        difficulty: "Easy",
        bots: 0,
        seed: "doomsday-grace-600",
        maxTicks: 400,
        doomsdayClock: { enabled: true, speed: "veryfast" },
      }),
      terrain,
    );
    let spawnRegion = 0;
    const spawnObs = enabled.peekObs();
    for (let i = 0; i < spawnObs.spawnRegions.length; i++) {
      if (spawnObs.spawnRegions[i] === 1) {
        spawnRegion = i;
        break;
      }
    }
    r = enabled.step({
      actionType: ACTION_SPAWN,
      target: 0,
      region: spawnRegion,
      quantity: 0,
      unit: 0,
    });
    for (let i = 0; i < 8 && !r.done; i++) {
      r = enabled.step({
        actionType: ACTION_NOOP,
        target: 0,
        region: 0,
        quantity: 0,
        unit: 0,
      });
    }
    expect(
      enabled.peekSlots().some((p) => p !== null && p.inDoomsdayClock()),
    ).toBe(false);
    expect(enabled.gameTicks()).toBeLessThan(6000);
  }, 180000);
});

function simpleSeed(s: string): number {
  let h = 0;
  for (let i = 0; i < s.length; i++) {
    h = (h * 31 + s.charCodeAt(i)) | 0;
  }
  return h >>> 0;
}
