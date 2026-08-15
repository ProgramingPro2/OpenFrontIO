/**
 * Training-safe terrain loader.
 *
 * The upstream loadTerrainMap() caches TerrainMapData (including the mutable
 * GameMapImpl) in a module-level Map, which poisons every subsequent game on
 * the same map - fine for one-game-per-page-load, fatal for training. This
 * loader caches only the immutable raw bytes/manifest and builds a fresh
 * GameMapImpl per game via genTerrainFromBin().
 */
import {
  GameMapSize,
  GameMapType,
  TeamGameSpawnAreas,
} from "../../src/core/game/Game";
import { GameMap } from "../../src/core/game/GameMap";
import { GameMapLoader } from "../../src/core/game/GameMapLoader";
import {
  AdditionalNation,
  genTerrainFromBin,
  MapManifest,
  Nation as ManifestNation,
} from "../../src/core/game/TerrainMapLoader";

export interface FreshTerrain {
  nations: ManifestNation[];
  additionalNations: AdditionalNation[];
  gameMap: GameMap;
  miniGameMap: GameMap;
  teamGameSpawnAreas?: TeamGameSpawnAreas;
}

interface CachedBins {
  manifest: MapManifest;
  mapBin: Uint8Array;
  map4xBin: Uint8Array;
  map16xBin: Uint8Array;
}

export class TerrainCache {
  private bins = new Map<string, CachedBins>();

  constructor(private loader: GameMapLoader) {}

  private async loadBins(
    map: GameMapType,
    mapSize: GameMapSize,
  ): Promise<CachedBins> {
    const key = `${map}:${mapSize}`;
    const hit = this.bins.get(key);
    if (hit !== undefined) return hit;
    const files = this.loader.getMapData(map);
    const [manifest, mapBin, map4xBin, map16xBin] = await Promise.all([
      files.manifest(),
      files.mapBin(),
      files.map4xBin(),
      files.map16xBin(),
    ]);
    const entry: CachedBins = { manifest, mapBin, map4xBin, map16xBin };
    this.bins.set(key, entry);
    return entry;
  }

  /** Fresh GameMapImpl instances per call; safe across any number of games. */
  async freshTerrain(
    map: GameMapType,
    mapSize: GameMapSize,
  ): Promise<FreshTerrain> {
    const { manifest, mapBin, map4xBin, map16xBin } = await this.loadBins(
      map,
      mapSize,
    );
    const compact = mapSize === GameMapSize.Compact;

    const gameMap = await genTerrainFromBin(
      compact ? manifest.map4x : manifest.map,
      compact ? map4xBin : mapBin,
    );
    const miniGameMap = await genTerrainFromBin(
      compact ? manifest.map16x : manifest.map4x,
      compact ? map16xBin : map4xBin,
    );

    // Deep-copy nation defs so compact scaling never mutates the cached
    // manifest (upstream loadTerrainMap mutates it in place).
    const scale = (n: { coordinates?: [number, number] }) =>
      compact && n.coordinates !== undefined
        ? ([Math.floor(n.coordinates[0] / 2), Math.floor(n.coordinates[1] / 2)] as [number, number])
        : n.coordinates;
    const nations: ManifestNation[] = manifest.nations.map((n) => ({
      ...n,
      coordinates: scale(n),
    }));
    const additionalNations: AdditionalNation[] = (
      manifest.additionalNations ?? []
    ).map((n) => ({ ...n, coordinates: scale(n) }));

    let teamGameSpawnAreas = manifest.teamGameSpawnAreas;
    if (compact && teamGameSpawnAreas !== undefined) {
      const scaled: TeamGameSpawnAreas = {};
      for (const [k, areas] of Object.entries(teamGameSpawnAreas)) {
        scaled[k] = areas.map((a) => ({
          x: Math.floor(a.x / 2),
          y: Math.floor(a.y / 2),
          width: Math.max(1, Math.floor(a.width / 2)),
          height: Math.max(1, Math.floor(a.height / 2)),
        }));
      }
      teamGameSpawnAreas = scaled;
    }

    return { nations, additionalNations, gameMap, miniGameMap, teamGameSpawnAreas };
  }
}
