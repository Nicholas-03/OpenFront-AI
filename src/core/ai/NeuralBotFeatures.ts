import {
  Game,
  Player,
  Relation,
  Structures,
  Unit,
  UnitType,
} from "../game/Game";
import { TileRef } from "../game/GameMap";
import {
  borderStats,
  byStrength,
  centerTile,
  findBuildTile,
  findUpgradeUnit,
  isNeuralBotActionAvailable,
  maxPlayerDistance,
  nearestOpponent,
  opponentsOf,
  playerDistance,
  rankPlayer,
  structureCount,
} from "./NeuralBotTactics";

export const NEURAL_BOT_SCALAR_FEATURE_NAMES = [
  "owned_land_ratio",
  "troops_log",
  "gold_log",
  "outgoing_attacks_ratio",
  "neutral_border_ratio",
  "enemy_border_ratio",
  "strongest_tiles_vs_self",
  "strongest_troops_vs_self",
  "weakest_tiles_vs_self",
  "weakest_troops_vs_self",
  "nearest_opponent_closeness",
  "bordering_opponents_ratio",
  "rank_score",
  "tick_progress",
  "alive_opponents_ratio",
  "incoming_attacks_ratio",
  "incoming_troops_vs_self",
  "troop_capacity_ratio",
  "strongest_max_troops_vs_self",
  "strongest_troop_capacity_ratio",
  "weakest_max_troops_vs_self",
  "weakest_troop_capacity_ratio",
  "nearest_opponent_troops_vs_self",
  "nearest_opponent_max_troops_vs_self",
  "nearest_opponent_troop_capacity_ratio",
  "border_enemy_troops_vs_self",
  "border_enemy_max_troops_vs_self",
  "border_enemy_troop_density_vs_self",
  "total_enemy_troops_vs_self",
  "total_enemy_max_troops_vs_self",
  "total_enemy_troop_capacity_ratio",
  "strongest_enemy_gold_vs_self",
  "weakest_enemy_gold_vs_self",
  "city_count",
  "factory_count",
  "port_count",
  "defense_post_count",
  "sam_launcher_count",
  "missile_silo_count",
  "warship_count",
  "transport_ship_count",
  "enemy_city_pressure",
  "enemy_factory_pressure",
  "enemy_port_pressure",
  "enemy_warship_pressure",
  "warship_health_average",
  "damaged_warship_ratio",
  "under_construction_ratio",
  "upgradable_structure_ratio",
  "city_affordability",
  "factory_affordability",
  "port_affordability",
  "warship_affordability",
  "defense_post_affordability",
  "sam_launcher_affordability",
  "missile_silo_affordability",
  "can_build_city",
  "can_build_factory",
  "can_build_port",
  "can_build_warship",
  "can_build_defense_post",
  "can_build_sam_launcher",
  "can_build_missile_silo",
  "can_upgrade_structure",
  "neighboring_players_ratio",
  "enemy_neighbor_ratio",
  "allied_neighbor_ratio",
  "strongest_neighbor_tiles_vs_self",
  "strongest_neighbor_troops_vs_self",
  "weakest_neighbor_tiles_vs_self",
  "weakest_neighbor_troops_vs_self",
  "nearest_neighbor_closeness",
  "nearest_neighbor_tiles_vs_self",
  "nearest_neighbor_troops_vs_self",
  "nearest_neighbor_gold_vs_self",
  "nearest_neighbor_relation_score",
  "nearest_neighbor_city_pressure",
  "nearest_neighbor_factory_pressure",
  "nearest_neighbor_defense_pressure",
  "alliance_count_ratio",
  "incoming_alliance_request_ratio",
  "outgoing_alliance_request_ratio",
  "best_alliance_candidate_closeness",
  "can_accept_alliance",
  "can_request_nearest_alliance",
  "can_request_strongest_alliance",
] as const;

export const NEURAL_BOT_FEATURE_NAMES = NEURAL_BOT_SCALAR_FEATURE_NAMES;
export const NEURAL_BOT_SCALAR_INPUT_SIZE =
  NEURAL_BOT_SCALAR_FEATURE_NAMES.length;

export const NEURAL_BOT_GLOBAL_MAP_SIZE = 64;
export const NEURAL_BOT_LOCAL_MAP_SIZE = 32;
export const NEURAL_BOT_SPATIAL_POOL_SIZE = 16;
export const NEURAL_BOT_LOCAL_VIEW_RADIUS_TILES = 48;
export const NEURAL_BOT_LONG_GAME_TICK_HORIZON = 6_000;
export const NEURAL_BOT_LOCAL_VIEW_NAMES = [
  "enemy_front",
  "neutral_front",
  "weakest_enemy_front",
  "strongest_threat_front",
] as const;
export const NEURAL_BOT_LOCAL_VIEW_COUNT = NEURAL_BOT_LOCAL_VIEW_NAMES.length;

export const NEURAL_BOT_MAP_CHANNEL_NAMES = [
  "terrain_land",
  "terrain_water",
  "neutral_land",
  "own_land",
  "enemy_land",
  "friendly_land",
  "own_border",
  "enemy_border",
  "own_structure",
  "enemy_structure",
  "own_warship",
  "enemy_warship",
  "fallout",
  "own_troop_density",
  "enemy_troop_density",
] as const;

export const NEURAL_BOT_MAP_CHANNEL_COUNT = NEURAL_BOT_MAP_CHANNEL_NAMES.length;
export const NEURAL_BOT_GLOBAL_MAP_VALUES =
  NEURAL_BOT_GLOBAL_MAP_SIZE *
  NEURAL_BOT_GLOBAL_MAP_SIZE *
  NEURAL_BOT_MAP_CHANNEL_COUNT;
export const NEURAL_BOT_LOCAL_MAP_VALUES =
  NEURAL_BOT_LOCAL_MAP_SIZE *
  NEURAL_BOT_LOCAL_MAP_SIZE *
  NEURAL_BOT_MAP_CHANNEL_COUNT;
export const NEURAL_BOT_POOLED_SPATIAL_VALUES =
  NEURAL_BOT_SPATIAL_POOL_SIZE *
  NEURAL_BOT_SPATIAL_POOL_SIZE *
  NEURAL_BOT_MAP_CHANNEL_COUNT;
export const NEURAL_BOT_INPUT_SIZE =
  NEURAL_BOT_SCALAR_INPUT_SIZE +
  NEURAL_BOT_POOLED_SPATIAL_VALUES * (1 + NEURAL_BOT_LOCAL_VIEW_COUNT);

export interface NeuralBotFeatureContext {
  game: Game;
  player: Player;
  opponentBaseline: number;
  tickHorizon: number;
}

export interface NeuralBotSpatialState {
  scalarFeatures: number[];
  globalMap: number[];
  localMap: number[];
  localCenter: TileRef | null;
  localMaps: number[][];
  localCenters: Array<TileRef | null>;
  input: number[];
}

interface SpatialLookup {
  enemyStructureTiles: Set<TileRef>;
  enemyWarshipTiles: Set<TileRef>;
  ownStructureTiles: Set<TileRef>;
  ownWarshipTiles: Set<TileRef>;
}

function clamp01(value: number): number {
  if (!Number.isFinite(value)) {
    return 0;
  }
  return Math.max(0, Math.min(1, value));
}

function clampCoord(value: number, maxExclusive: number): number {
  return Math.max(0, Math.min(maxExclusive - 1, value));
}

function normRatio(value: number): number {
  return value / (1 + Math.abs(value));
}

function logNorm(value: number, scale: number): number {
  return clamp01(Math.log1p(Math.max(0, value)) / Math.log1p(scale));
}

function countNorm(value: number, scale: number): number {
  return logNorm(value, scale);
}

function maxTroopsFor(game: Game, player: Player): number {
  return Math.max(1, game.config().maxTroops(player));
}

function troopCapacityRatio(game: Game, player: Player | null): number {
  return player === null
    ? 0
    : clamp01(player.troops() / maxTroopsFor(game, player));
}

function troopDensity(player: Player | null): number {
  return player === null
    ? 0
    : player.troops() / Math.max(1, player.numTilesOwned());
}

function affordability(game: Game, player: Player, type: UnitType): number {
  if (game.config().isUnitDisabled(type)) {
    return 0;
  }

  const cost = game.unitInfo(type).cost(game, player);
  if (cost <= 0n) {
    return 1;
  }

  return clamp01(Number(player.gold()) / Number(cost));
}

function maxOpponentUnitCount(opponents: Player[], type: UnitType): number {
  return Math.max(0, ...opponents.map((opponent) => opponent.unitCount(type)));
}

function neighborPlayers(player: Player): Player[] {
  return player
    .nearby()
    .filter(
      (neighbor): neighbor is Player =>
        neighbor.isPlayer() && neighbor.isAlive() && neighbor !== player,
    );
}

function nearestPlayerFrom(
  game: Game,
  player: Player,
  candidates: Player[],
): Player | null {
  const playerCenter = centerTile(game, player);
  if (playerCenter === null) {
    return null;
  }

  let nearest: Player | null = null;
  let nearestDistance = Number.POSITIVE_INFINITY;
  for (const candidate of candidates) {
    const candidateCenter = centerTile(game, candidate);
    if (candidateCenter === null) {
      continue;
    }
    const distance = game.manhattanDist(playerCenter, candidateCenter);
    if (distance < nearestDistance) {
      nearest = candidate;
      nearestDistance = distance;
    }
  }
  return nearest;
}

function relationScore(player: Player, other: Player | null): number {
  if (other === null) {
    return 0;
  }
  if (player.isFriendly(other)) {
    return 1;
  }

  switch (player.relation(other)) {
    case Relation.Hostile:
      return 0;
    case Relation.Distrustful:
      return 0.25;
    case Relation.Neutral:
      return 0.5;
    case Relation.Friendly:
      return 0.75;
  }
}

function bestAllianceCandidateCloseness(game: Game, player: Player): number {
  const nearestCandidate = nearestPlayerFrom(
    game,
    player,
    game
      .players()
      .filter((candidate) => player.canSendAllianceRequest(candidate)),
  );
  return nearestCandidate
    ? clamp01(
        1 -
          playerDistance(game, player, nearestCandidate) /
            maxPlayerDistance(game),
      )
    : 0;
}

function averageWarshipHealth(game: Game, warships: Unit[]): number {
  if (warships.length === 0) {
    return 0;
  }

  const maxHealth = game.unitInfo(UnitType.Warship).maxHealth ?? 1000;
  const total = warships.reduce(
    (acc, warship) => acc + clamp01(warship.health() / maxHealth),
    0,
  );
  return total / warships.length;
}

function damagedWarshipRatio(game: Game, warships: Unit[]): number {
  if (warships.length === 0) {
    return 0;
  }

  const maxHealth = game.unitInfo(UnitType.Warship).maxHealth ?? 1000;
  return (
    warships.filter((warship) => warship.health() < maxHealth * 0.55).length /
    warships.length
  );
}

export function buildNeuralBotFeatures({
  game,
  player,
  opponentBaseline,
  tickHorizon,
}: NeuralBotFeatureContext): number[] {
  const opponents = opponentsOf(game, player);
  const opponentsByStrength = byStrength(opponents);
  const strongest = opponentsByStrength[0] ?? null;
  const weakest = opponentsByStrength[opponentsByStrength.length - 1] ?? null;
  const nearest = nearestOpponent(game, player);
  const borders = borderStats(game, player);
  const selfTiles = Math.max(1, player.numTilesOwned());
  const selfTroops = Math.max(1, player.troops());
  const selfMaxTroops = maxTroopsFor(game, player);
  const selfGold = Math.max(1, Number(player.gold()));
  const rank = rankPlayer(game, player);
  const structures = player.units(...Structures.types);
  const ownStructureCount = Math.max(1, structureCount(player));
  const ownWarships = player.units(UnitType.Warship);
  const borderingOpponents = opponents.filter((opponent) =>
    player.sharesBorderWith(opponent),
  );
  const borderEnemyTroops = borderingOpponents.reduce(
    (total, opponent) => total + opponent.troops(),
    0,
  );
  const borderEnemyMaxTroops = borderingOpponents.reduce(
    (total, opponent) => total + maxTroopsFor(game, opponent),
    0,
  );
  const borderEnemyTiles = borderingOpponents.reduce(
    (total, opponent) => total + opponent.numTilesOwned(),
    0,
  );
  const totalEnemyTroops = opponents.reduce(
    (total, opponent) => total + opponent.troops(),
    0,
  );
  const totalEnemyMaxTroops = opponents.reduce(
    (total, opponent) => total + maxTroopsFor(game, opponent),
    0,
  );
  const neighbors = neighborPlayers(player);
  const neighborsByStrength = byStrength(neighbors);
  const enemyNeighbors = neighbors.filter(
    (neighbor) => !player.isFriendly(neighbor),
  );
  const alliedNeighbors = neighbors.filter((neighbor) =>
    player.isFriendly(neighbor),
  );
  const strongestNeighbor = neighborsByStrength[0] ?? null;
  const weakestNeighbor =
    neighborsByStrength[neighborsByStrength.length - 1] ?? null;
  const nearestNeighbor = nearestPlayerFrom(game, player, neighbors);
  const incomingTroops = player
    .incomingAttacks()
    .reduce((acc, attack) => acc + attack.troops(), 0);
  const upgradableStructure = findUpgradeUnit(player);
  const canBuildCity = findBuildTile(game, player, UnitType.City) !== null;
  const canBuildFactory =
    findBuildTile(game, player, UnitType.Factory) !== null;
  const canBuildPort = findBuildTile(game, player, UnitType.Port) !== null;
  const canBuildWarship =
    findBuildTile(game, player, UnitType.Warship) !== null;
  const canBuildDefensePost =
    findBuildTile(game, player, UnitType.DefensePost) !== null;
  const canBuildSamLauncher =
    findBuildTile(game, player, UnitType.SAMLauncher) !== null;
  const canBuildMissileSilo =
    findBuildTile(game, player, UnitType.MissileSilo) !== null;

  const features = [
    player.numTilesOwned() / game.numLandTiles(),
    logNorm(player.troops(), 250_000),
    logNorm(Number(player.gold()), 1_000_000),
    Math.min(1, player.outgoingAttacks().length / 6),
    borders.neutral,
    borders.enemy,
    strongest ? normRatio(strongest.numTilesOwned() / selfTiles) : 0,
    strongest ? normRatio(strongest.troops() / selfTroops) : 0,
    weakest ? normRatio(weakest.numTilesOwned() / selfTiles) : 0,
    weakest ? normRatio(weakest.troops() / selfTroops) : 0,
    nearest
      ? clamp01(
          1 - playerDistance(game, player, nearest) / maxPlayerDistance(game),
        )
      : 0,
    opponents.filter((opponent) => player.sharesBorderWith(opponent)).length /
      Math.max(1, opponents.length),
    clamp01(1 - (rank - 1) / Math.max(1, game.allPlayers().length - 1)),
    clamp01(game.ticks() / Math.max(1, tickHorizon)),
    opponents.length / Math.max(1, opponentBaseline),
    Math.min(1, player.incomingAttacks().length / 6),
    logNorm(incomingTroops / selfTroops, 8),
    troopCapacityRatio(game, player),
    strongest ? normRatio(maxTroopsFor(game, strongest) / selfMaxTroops) : 0,
    troopCapacityRatio(game, strongest),
    weakest ? normRatio(maxTroopsFor(game, weakest) / selfMaxTroops) : 0,
    troopCapacityRatio(game, weakest),
    nearest ? normRatio(nearest.troops() / selfTroops) : 0,
    nearest ? normRatio(maxTroopsFor(game, nearest) / selfMaxTroops) : 0,
    troopCapacityRatio(game, nearest),
    normRatio(borderEnemyTroops / selfTroops),
    normRatio(borderEnemyMaxTroops / selfMaxTroops),
    normRatio(
      borderEnemyTroops / Math.max(1, borderEnemyTiles) / troopDensity(player),
    ),
    normRatio(totalEnemyTroops / selfTroops),
    normRatio(totalEnemyMaxTroops / selfMaxTroops),
    clamp01(totalEnemyTroops / Math.max(1, totalEnemyMaxTroops)),
    strongest ? normRatio(Number(strongest.gold()) / selfGold) : 0,
    weakest ? normRatio(Number(weakest.gold()) / selfGold) : 0,
    countNorm(player.unitCount(UnitType.City), 12),
    countNorm(player.unitCount(UnitType.Factory), 8),
    countNorm(player.unitCount(UnitType.Port), 8),
    countNorm(player.unitCount(UnitType.DefensePost), 16),
    countNorm(player.unitCount(UnitType.SAMLauncher), 5),
    countNorm(player.unitCount(UnitType.MissileSilo), 5),
    countNorm(player.unitCount(UnitType.Warship), 10),
    countNorm(player.unitCount(UnitType.TransportShip), 10),
    normRatio(
      maxOpponentUnitCount(opponents, UnitType.City) / ownStructureCount,
    ),
    normRatio(
      maxOpponentUnitCount(opponents, UnitType.Factory) / ownStructureCount,
    ),
    normRatio(
      maxOpponentUnitCount(opponents, UnitType.Port) / ownStructureCount,
    ),
    normRatio(
      maxOpponentUnitCount(opponents, UnitType.Warship) /
        Math.max(1, ownWarships.length),
    ),
    averageWarshipHealth(game, ownWarships),
    damagedWarshipRatio(game, ownWarships),
    structures.filter((unit) => unit.isUnderConstruction()).length /
      ownStructureCount,
    structures.filter((unit) => player.canUpgradeUnit(unit)).length /
      ownStructureCount,
    affordability(game, player, UnitType.City),
    affordability(game, player, UnitType.Factory),
    affordability(game, player, UnitType.Port),
    affordability(game, player, UnitType.Warship),
    affordability(game, player, UnitType.DefensePost),
    affordability(game, player, UnitType.SAMLauncher),
    affordability(game, player, UnitType.MissileSilo),
    canBuildCity ? 1 : 0,
    canBuildFactory ? 1 : 0,
    canBuildPort ? 1 : 0,
    canBuildWarship ? 1 : 0,
    canBuildDefensePost ? 1 : 0,
    canBuildSamLauncher ? 1 : 0,
    canBuildMissileSilo ? 1 : 0,
    upgradableStructure ? 1 : 0,
    neighbors.length / Math.max(1, opponentBaseline),
    enemyNeighbors.length / Math.max(1, neighbors.length),
    alliedNeighbors.length / Math.max(1, neighbors.length),
    strongestNeighbor
      ? normRatio(strongestNeighbor.numTilesOwned() / selfTiles)
      : 0,
    strongestNeighbor ? normRatio(strongestNeighbor.troops() / selfTroops) : 0,
    weakestNeighbor
      ? normRatio(weakestNeighbor.numTilesOwned() / selfTiles)
      : 0,
    weakestNeighbor ? normRatio(weakestNeighbor.troops() / selfTroops) : 0,
    nearestNeighbor
      ? clamp01(
          1 -
            playerDistance(game, player, nearestNeighbor) /
              maxPlayerDistance(game),
        )
      : 0,
    nearestNeighbor
      ? normRatio(nearestNeighbor.numTilesOwned() / selfTiles)
      : 0,
    nearestNeighbor ? normRatio(nearestNeighbor.troops() / selfTroops) : 0,
    nearestNeighbor ? normRatio(Number(nearestNeighbor.gold()) / selfGold) : 0,
    relationScore(player, nearestNeighbor),
    nearestNeighbor
      ? normRatio(nearestNeighbor.unitCount(UnitType.City) / ownStructureCount)
      : 0,
    nearestNeighbor
      ? normRatio(
          nearestNeighbor.unitCount(UnitType.Factory) / ownStructureCount,
        )
      : 0,
    nearestNeighbor
      ? normRatio(
          nearestNeighbor.unitCount(UnitType.DefensePost) / ownStructureCount,
        )
      : 0,
    player.alliances().length / Math.max(1, opponentBaseline),
    player.incomingAllianceRequests().length / Math.max(1, opponentBaseline),
    player.outgoingAllianceRequests().length / Math.max(1, opponentBaseline),
    bestAllianceCandidateCloseness(game, player),
    isNeuralBotActionAvailable(game, player, "accept_alliance") ? 1 : 0,
    isNeuralBotActionAvailable(game, player, "request_alliance_nearest")
      ? 1
      : 0,
    isNeuralBotActionAvailable(game, player, "request_alliance_strongest")
      ? 1
      : 0,
  ];

  if (features.length !== NEURAL_BOT_SCALAR_INPUT_SIZE) {
    throw new Error(
      `Expected ${NEURAL_BOT_SCALAR_INPUT_SIZE} neural bot scalar features, built ${features.length}`,
    );
  }

  return features;
}

export function buildNeuralBotInput(
  context: NeuralBotFeatureContext,
): number[] {
  return buildNeuralBotSpatialState(context).input;
}

export function buildNeuralBotSpatialState(
  context: NeuralBotFeatureContext,
): NeuralBotSpatialState {
  const scalarFeatures = buildNeuralBotFeatures(context);
  const lookup = buildSpatialLookup(context.game, context.player);
  const localCenters = findLocalViewCenters(context.game, context.player);
  const globalMap = buildGlobalMapView(context.game, context.player, lookup);
  const localMaps = localCenters.map((localCenter) =>
    buildLocalMapView(context.game, context.player, lookup, localCenter),
  );
  const localCenter = localCenters[0] ?? null;
  const localMap =
    localMaps[0] ?? new Array(NEURAL_BOT_LOCAL_MAP_VALUES).fill(0);
  const input = [
    ...scalarFeatures,
    ...poolSpatialView(globalMap, NEURAL_BOT_GLOBAL_MAP_SIZE),
    ...localMaps.flatMap((view) =>
      poolSpatialView(view, NEURAL_BOT_LOCAL_MAP_SIZE),
    ),
  ];

  if (input.length !== NEURAL_BOT_INPUT_SIZE) {
    throw new Error(
      `Expected ${NEURAL_BOT_INPUT_SIZE} neural bot inputs, built ${input.length}`,
    );
  }

  return {
    scalarFeatures,
    globalMap,
    localMap,
    localCenter,
    localMaps,
    localCenters,
    input,
  };
}

function buildSpatialLookup(game: Game, player: Player): SpatialLookup {
  const lookup: SpatialLookup = {
    enemyStructureTiles: new Set(),
    enemyWarshipTiles: new Set(),
    ownStructureTiles: new Set(),
    ownWarshipTiles: new Set(),
  };

  for (const unit of game.units()) {
    if (!unit.isActive()) {
      continue;
    }
    const owner = unit.owner();
    const isOwn = owner === player;
    const isEnemy = owner !== player && !owner.isFriendly(player);
    if (!isOwn && !isEnemy) {
      continue;
    }

    if (Structures.has(unit.type())) {
      (isOwn ? lookup.ownStructureTiles : lookup.enemyStructureTiles).add(
        unit.tile(),
      );
    }
    if (unit.type() === UnitType.Warship) {
      (isOwn ? lookup.ownWarshipTiles : lookup.enemyWarshipTiles).add(
        unit.tile(),
      );
    }
  }

  return lookup;
}

function buildGlobalMapView(
  game: Game,
  player: Player,
  lookup: SpatialLookup,
): number[] {
  const view = new Array(NEURAL_BOT_GLOBAL_MAP_VALUES).fill(0);
  for (let y = 0; y < NEURAL_BOT_GLOBAL_MAP_SIZE; y++) {
    const sourceY = clampCoord(
      Math.floor(((y + 0.5) * game.height()) / NEURAL_BOT_GLOBAL_MAP_SIZE),
      game.height(),
    );
    for (let x = 0; x < NEURAL_BOT_GLOBAL_MAP_SIZE; x++) {
      const sourceX = clampCoord(
        Math.floor(((x + 0.5) * game.width()) / NEURAL_BOT_GLOBAL_MAP_SIZE),
        game.width(),
      );
      writeTileChannels(
        view,
        y * NEURAL_BOT_GLOBAL_MAP_SIZE + x,
        game,
        player,
        game.ref(sourceX, sourceY),
        lookup,
      );
    }
  }
  return view;
}

function buildLocalMapView(
  game: Game,
  player: Player,
  lookup: SpatialLookup,
  center: TileRef | null,
): number[] {
  const view = new Array(NEURAL_BOT_LOCAL_MAP_VALUES).fill(0);
  if (center === null) {
    return view;
  }

  const centerX = game.x(center);
  const centerY = game.y(center);
  for (let y = 0; y < NEURAL_BOT_LOCAL_MAP_SIZE; y++) {
    const sourceY = Math.round(
      centerY +
        ((y + 0.5) / NEURAL_BOT_LOCAL_MAP_SIZE - 0.5) *
          NEURAL_BOT_LOCAL_VIEW_RADIUS_TILES *
          2,
    );
    for (let x = 0; x < NEURAL_BOT_LOCAL_MAP_SIZE; x++) {
      const sourceX = Math.round(
        centerX +
          ((x + 0.5) / NEURAL_BOT_LOCAL_MAP_SIZE - 0.5) *
            NEURAL_BOT_LOCAL_VIEW_RADIUS_TILES *
            2,
      );
      if (!game.isValidCoord(sourceX, sourceY)) {
        continue;
      }
      writeTileChannels(
        view,
        y * NEURAL_BOT_LOCAL_MAP_SIZE + x,
        game,
        player,
        game.ref(sourceX, sourceY),
        lookup,
      );
    }
  }
  return view;
}

function writeTileChannels(
  view: number[],
  cellIndex: number,
  game: Game,
  player: Player,
  tile: TileRef,
  lookup: SpatialLookup,
): void {
  const base = cellIndex * NEURAL_BOT_MAP_CHANNEL_COUNT;
  const land = game.isLand(tile);
  const owner = game.owner(tile);
  const isOwn = owner === player;
  const isFriendly =
    owner.isPlayer() && owner !== player && player.isFriendly(owner);
  const isEnemy =
    owner.isPlayer() && owner !== player && !player.isFriendly(owner);
  const isBorder = game.isBorder(tile);

  view[base] = land ? 1 : 0;
  view[base + 1] = land ? 0 : 1;
  view[base + 2] = land && !owner.isPlayer() ? 1 : 0;
  view[base + 3] = isOwn ? 1 : 0;
  view[base + 4] = isEnemy ? 1 : 0;
  view[base + 5] = isFriendly ? 1 : 0;
  view[base + 6] = isOwn && isBorder ? 1 : 0;
  view[base + 7] = isEnemy && isBorder ? 1 : 0;
  view[base + 8] = lookup.ownStructureTiles.has(tile) ? 1 : 0;
  view[base + 9] = lookup.enemyStructureTiles.has(tile) ? 1 : 0;
  view[base + 10] = lookup.ownWarshipTiles.has(tile) ? 1 : 0;
  view[base + 11] = lookup.enemyWarshipTiles.has(tile) ? 1 : 0;
  view[base + 12] = game.hasFallout(tile) ? 1 : 0;
  view[base + 13] =
    isOwn && owner.isPlayer() ? logNorm(troopDensity(owner), 5_000) : 0;
  view[base + 14] =
    isEnemy && owner.isPlayer() ? logNorm(troopDensity(owner), 5_000) : 0;
}

function poolSpatialView(view: number[], sourceSize: number): number[] {
  const pooled = new Array(NEURAL_BOT_POOLED_SPATIAL_VALUES).fill(0);
  const cellScale = sourceSize / NEURAL_BOT_SPATIAL_POOL_SIZE;

  for (let channel = 0; channel < NEURAL_BOT_MAP_CHANNEL_COUNT; channel++) {
    for (let pooledY = 0; pooledY < NEURAL_BOT_SPATIAL_POOL_SIZE; pooledY++) {
      const startY = Math.floor(pooledY * cellScale);
      const endY = Math.floor((pooledY + 1) * cellScale);
      for (let pooledX = 0; pooledX < NEURAL_BOT_SPATIAL_POOL_SIZE; pooledX++) {
        const startX = Math.floor(pooledX * cellScale);
        const endX = Math.floor((pooledX + 1) * cellScale);
        let sum = 0;
        let count = 0;

        for (let sourceY = startY; sourceY < endY; sourceY++) {
          for (let sourceX = startX; sourceX < endX; sourceX++) {
            sum +=
              view[
                (sourceY * sourceSize + sourceX) *
                  NEURAL_BOT_MAP_CHANNEL_COUNT +
                  channel
              ];
            count++;
          }
        }

        pooled[
          (channel * NEURAL_BOT_SPATIAL_POOL_SIZE + pooledY) *
            NEURAL_BOT_SPATIAL_POOL_SIZE +
            pooledX
        ] = count === 0 ? 0 : sum / count;
      }
    }
  }

  return pooled;
}

function enemyPlayers(game: Game, player: Player): Player[] {
  return game
    .players()
    .filter(
      (candidate) =>
        candidate !== player &&
        candidate.isAlive() &&
        !player.isFriendly(candidate),
    );
}

function findLocalViewCenters(
  game: Game,
  player: Player,
): Array<TileRef | null> {
  const enemyBorderTiles: TileRef[] = [];
  const neutralBorderTiles: TileRef[] = [];
  const enemyBorderTilesByPlayer = new Map<Player, TileRef[]>();

  for (const tile of player.borderTiles()) {
    const touchingEnemies = new Set<Player>();
    let touchesNeutral = false;
    game.forEachNeighbor(tile, (neighbor) => {
      if (!game.isLand(neighbor)) {
        return;
      }
      const owner = game.owner(neighbor);
      if (owner === player) {
        return;
      }
      if (owner.isPlayer() && !player.isFriendly(owner)) {
        touchingEnemies.add(owner);
      } else if (!owner.isPlayer()) {
        touchesNeutral = true;
      }
    });

    if (touchingEnemies.size > 0) {
      enemyBorderTiles.push(tile);
      for (const enemy of touchingEnemies) {
        const tiles = enemyBorderTilesByPlayer.get(enemy) ?? [];
        tiles.push(tile);
        enemyBorderTilesByPlayer.set(enemy, tiles);
      }
    } else if (touchesNeutral) {
      neutralBorderTiles.push(tile);
    }
  }

  const enemies = byStrength(enemyPlayers(game, player));
  const borderingEnemies = enemies.filter((enemy) =>
    player.sharesBorderWith(enemy),
  );
  const strongestThreat = borderingEnemies[0] ?? enemies[0] ?? null;
  const weakestEnemy =
    borderingEnemies[borderingEnemies.length - 1] ??
    enemies[enemies.length - 1] ??
    null;
  const enemyFrontCenter = tileNearestCentroid(game, enemyBorderTiles);
  const neutralFrontCenter = tileNearestCentroid(game, neutralBorderTiles);
  const fallbackCenter =
    enemyFrontCenter ?? neutralFrontCenter ?? centerTile(game, player);

  return [
    enemyFrontCenter ??
      centerForPlayerFront(game, player, strongestThreat) ??
      fallbackCenter,
    neutralFrontCenter ?? enemyFrontCenter ?? fallbackCenter,
    centerForPlayerFront(
      game,
      player,
      weakestEnemy,
      enemyBorderTilesByPlayer,
    ) ??
      enemyFrontCenter ??
      fallbackCenter,
    centerForPlayerFront(
      game,
      player,
      strongestThreat,
      enemyBorderTilesByPlayer,
    ) ??
      enemyFrontCenter ??
      fallbackCenter,
  ];
}

function centerForPlayerFront(
  game: Game,
  player: Player,
  target: Player | null,
  borderTilesByPlayer = new Map<Player, TileRef[]>(),
): TileRef | null {
  if (target === null) {
    return null;
  }

  return (
    tileNearestCentroid(game, borderTilesByPlayer.get(target) ?? []) ??
    centerTile(game, target) ??
    centerTile(game, player)
  );
}

function tileNearestCentroid(game: Game, tiles: TileRef[]): TileRef | null {
  if (tiles.length === 0) {
    return null;
  }

  let totalX = 0;
  let totalY = 0;
  for (const tile of tiles) {
    totalX += game.x(tile);
    totalY += game.y(tile);
  }
  const centerX = totalX / tiles.length;
  const centerY = totalY / tiles.length;

  let bestTile = tiles[0];
  let bestDistance = Number.POSITIVE_INFINITY;
  for (const tile of tiles) {
    const dx = game.x(tile) - centerX;
    const dy = game.y(tile) - centerY;
    const distance = dx * dx + dy * dy;
    if (distance < bestDistance) {
      bestDistance = distance;
      bestTile = tile;
    }
  }
  return bestTile;
}
