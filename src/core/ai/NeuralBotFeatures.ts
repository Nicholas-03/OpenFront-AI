import { Game, Player, Structures, Unit, UnitType } from "../game/Game";
import {
  borderStats,
  byStrength,
  findBuildTile,
  findUpgradeUnit,
  maxPlayerDistance,
  nearestOpponent,
  opponentsOf,
  playerDistance,
  rankPlayer,
  structureCount,
} from "./NeuralBotTactics";

export const NEURAL_BOT_FEATURE_NAMES = [
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
] as const;

export const NEURAL_BOT_INPUT_SIZE = NEURAL_BOT_FEATURE_NAMES.length;

export interface NeuralBotFeatureContext {
  game: Game;
  player: Player;
  opponentBaseline: number;
  tickHorizon: number;
}

function clamp01(value: number): number {
  if (!Number.isFinite(value)) {
    return 0;
  }
  return Math.max(0, Math.min(1, value));
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
  const rank = rankPlayer(game, player);
  const structures = player.units(...Structures.types);
  const ownStructureCount = Math.max(1, structureCount(player));
  const ownWarships = player.units(UnitType.Warship);
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
  ];

  if (features.length !== NEURAL_BOT_INPUT_SIZE) {
    throw new Error(
      `Expected ${NEURAL_BOT_INPUT_SIZE} neural bot features, built ${features.length}`,
    );
  }

  return features;
}
