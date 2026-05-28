import { AllianceRequestExecution } from "../execution/alliance/AllianceRequestExecution";
import { AttackExecution } from "../execution/AttackExecution";
import { ConstructionExecution } from "../execution/ConstructionExecution";
import { UpgradeStructureExecution } from "../execution/UpgradeStructureExecution";
import {
  Game,
  Player,
  Structures,
  TerraNullius,
  Unit,
  UnitType,
} from "../game/Game";
import { TileRef } from "../game/GameMap";
import type { NeuralBotAction } from "./NeuralBotPolicy";

export const MAX_NEURAL_PARALLEL_ATTACKS = 6;

export const NEURAL_BUILD_ACTIONS: Partial<Record<NeuralBotAction, UnitType>> =
  {
    build_city: UnitType.City,
    build_factory: UnitType.Factory,
    build_port: UnitType.Port,
    build_warship: UnitType.Warship,
    build_defense_post: UnitType.DefensePost,
    build_sam_launcher: UnitType.SAMLauncher,
    build_missile_silo: UnitType.MissileSilo,
  };

const STRUCTURE_SEARCH_SAMPLE_LIMIT = 700;
const WATER_SEARCH_RADIUS = 12;
const BUILD_SEARCH_TYPES = [
  UnitType.City,
  UnitType.Factory,
  UnitType.Port,
  UnitType.Warship,
  UnitType.DefensePost,
  UnitType.SAMLauncher,
  UnitType.MissileSilo,
] as const;

const UPGRADE_PRIORITY = [
  UnitType.City,
  UnitType.Factory,
  UnitType.Port,
  UnitType.SAMLauncher,
  UnitType.MissileSilo,
] as const;

export type NeuralBuildUnitType = (typeof BUILD_SEARCH_TYPES)[number];

export function centerTile(game: Game, player: Player): TileRef | null {
  const spawnTile = player.spawnTile();
  if (spawnTile !== undefined) {
    return spawnTile;
  }

  const firstTile = player.tiles().values().next().value;
  return typeof firstTile === "number" ? firstTile : null;
}

export function opponentsOf(game: Game, player: Player): Player[] {
  return game
    .players()
    .filter((candidate) => candidate !== player && candidate.isAlive());
}

export function byStrength(players: Player[]): Player[] {
  return players.slice().sort((left, right) => {
    const rightStrength = right.isAlive()
      ? right.numTilesOwned() + right.troops() / 500
      : -1;
    const leftStrength = left.isAlive()
      ? left.numTilesOwned() + left.troops() / 500
      : -1;
    return rightStrength - leftStrength;
  });
}

export function rankPlayer(game: Game, player: Player): number {
  const ordered = byStrength(game.allPlayers());
  const index = ordered.findIndex((candidate) => candidate === player);
  return index === -1 ? ordered.length : index + 1;
}

export function nearestOpponent(game: Game, player: Player): Player | null {
  const playerCenter = centerTile(game, player);
  if (playerCenter === null) {
    return null;
  }

  let nearest: Player | null = null;
  let nearestDistance = Number.POSITIVE_INFINITY;
  for (const opponent of opponentsOf(game, player)) {
    const opponentCenter = centerTile(game, opponent);
    if (opponentCenter === null) {
      continue;
    }
    const distance = game.manhattanDist(playerCenter, opponentCenter);
    if (distance < nearestDistance) {
      nearest = opponent;
      nearestDistance = distance;
    }
  }
  return nearest;
}

export function playerDistance(
  game: Game,
  left: Player,
  right: Player,
): number {
  const leftCenter = centerTile(game, left);
  const rightCenter = centerTile(game, right);
  if (leftCenter === null || rightCenter === null) {
    return maxPlayerDistance(game);
  }
  return game.manhattanDist(leftCenter, rightCenter);
}

export function maxPlayerDistance(game: Game): number {
  return game.width() + game.height();
}

export function borderStats(
  game: Game,
  player: Player,
): { enemy: number; neutral: number } {
  let enemy = 0;
  let neutral = 0;
  let total = 0;

  for (const tile of player.borderTiles()) {
    game.forEachNeighbor(tile, (neighbor) => {
      if (!game.isLand(neighbor)) {
        return;
      }
      const owner = game.owner(neighbor);
      if (owner === player) {
        return;
      }
      total++;
      if (owner.isPlayer()) {
        enemy++;
      } else {
        neutral++;
      }
    });
  }

  return {
    enemy: total === 0 ? 0 : enemy / total,
    neutral: total === 0 ? 0 : neutral / total,
  };
}

export function tryNeuralBotAction(
  game: Game,
  player: Player,
  action: NeuralBotAction,
  maxParallelAttacks = MAX_NEURAL_PARALLEL_ATTACKS,
): boolean {
  if (!isNeuralBotActionAvailable(game, player, action, maxParallelAttacks)) {
    return false;
  }

  switch (action) {
    case "expand":
      game.addExecution(new AttackExecution(null, player, null));
      return true;
    case "attack_nearest":
      return attackPlayer(
        game,
        player,
        nearestOpponent(game, player),
        maxParallelAttacks,
      );
    case "attack_weakest": {
      const opponentsByStrength = byStrength(opponentsOf(game, player));
      return attackPlayer(
        game,
        player,
        opponentsByStrength[opponentsByStrength.length - 1],
        maxParallelAttacks,
      );
    }
    case "attack_strongest":
      return attackPlayer(
        game,
        player,
        byStrength(opponentsOf(game, player))[0],
        maxParallelAttacks,
      );
    case "upgrade_structure":
      return upgradeStructure(game, player);
    case "accept_alliance":
      return requestAlliance(game, player, bestIncomingAllianceRequest(player));
    case "request_alliance_nearest":
      return requestAlliance(
        game,
        player,
        nearestAllianceCandidate(game, player),
      );
    case "request_alliance_strongest":
      return requestAlliance(
        game,
        player,
        strongestAllianceCandidate(game, player),
      );
    case "build_city":
    case "build_factory":
    case "build_port":
    case "build_warship":
    case "build_defense_post":
    case "build_sam_launcher":
    case "build_missile_silo":
      return buildUnit(game, player, NEURAL_BUILD_ACTIONS[action]!);
    case "hold":
      return true;
  }
}

export function isNeuralBotActionAvailable(
  game: Game,
  player: Player,
  action: NeuralBotAction,
  maxParallelAttacks = MAX_NEURAL_PARALLEL_ATTACKS,
): boolean {
  if (!player.isAlive()) {
    return false;
  }

  switch (action) {
    case "expand":
      return (
        player.troops() >= 500 &&
        player.outgoingAttacks().length < maxParallelAttacks &&
        borderStats(game, player).neutral > 0
      );
    case "attack_nearest":
      return canAttackPlayer(
        game,
        player,
        nearestOpponent(game, player),
        maxParallelAttacks,
      );
    case "attack_weakest": {
      const opponentsByStrength = byStrength(opponentsOf(game, player));
      return canAttackPlayer(
        game,
        player,
        opponentsByStrength[opponentsByStrength.length - 1],
        maxParallelAttacks,
      );
    }
    case "attack_strongest":
      return canAttackPlayer(
        game,
        player,
        byStrength(opponentsOf(game, player))[0],
        maxParallelAttacks,
      );
    case "upgrade_structure":
      return findUpgradeUnit(player) !== null;
    case "accept_alliance":
      return bestIncomingAllianceRequest(player) !== null;
    case "request_alliance_nearest":
      return nearestAllianceCandidate(game, player) !== null;
    case "request_alliance_strongest":
      return strongestAllianceCandidate(game, player) !== null;
    case "build_city":
    case "build_factory":
    case "build_port":
    case "build_warship":
    case "build_defense_post":
    case "build_sam_launcher":
    case "build_missile_silo":
      return (
        findBuildTile(game, player, NEURAL_BUILD_ACTIONS[action]!) !== null
      );
    case "hold":
      return true;
  }
}

function allianceCandidates(game: Game, player: Player): Player[] {
  return game
    .players()
    .filter((candidate) => candidate !== player)
    .filter((candidate) => player.canSendAllianceRequest(candidate));
}

function bestIncomingAllianceRequest(player: Player): Player | null {
  const request = player
    .incomingAllianceRequests()
    .map((allianceRequest) => allianceRequest.requestor())
    .find((requestor) => player.canSendAllianceRequest(requestor));
  return request ?? null;
}

function nearestAllianceCandidate(game: Game, player: Player): Player | null {
  const playerCenter = centerTile(game, player);
  if (playerCenter === null) {
    return null;
  }

  let nearest: Player | null = null;
  let nearestDistance = Number.POSITIVE_INFINITY;
  for (const candidate of allianceCandidates(game, player)) {
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

function strongestAllianceCandidate(game: Game, player: Player): Player | null {
  return byStrength(allianceCandidates(game, player))[0] ?? null;
}

function requestAlliance(
  game: Game,
  player: Player,
  target: Player | null,
): boolean {
  if (target === null || !player.canSendAllianceRequest(target)) {
    return false;
  }

  game.addExecution(new AllianceRequestExecution(player, target.id()));
  return true;
}

function canAttackPlayer(
  game: Game,
  player: Player,
  target: Player | TerraNullius | null | undefined,
  maxParallelAttacks: number,
): target is Player {
  return !(
    player.troops() < 500 ||
    !target ||
    !target.isPlayer() ||
    !player.canAttackPlayer(target) ||
    !player.sharesBorderWith(target) ||
    player.outgoingAttacks().length >= maxParallelAttacks
  );
}

function attackPlayer(
  game: Game,
  player: Player,
  target: Player | TerraNullius | null | undefined,
  maxParallelAttacks: number,
): boolean {
  if (!canAttackPlayer(game, player, target, maxParallelAttacks)) {
    return false;
  }

  game.addExecution(new AttackExecution(null, player, target.id()));
  return true;
}

function buildUnit(game: Game, player: Player, type: UnitType): boolean {
  const tile = findBuildTile(game, player, type);
  if (tile === null) {
    return false;
  }

  game.addExecution(new ConstructionExecution(player, type, tile));
  return true;
}

function upgradeStructure(game: Game, player: Player): boolean {
  const unit = findUpgradeUnit(player);
  if (unit === null) {
    return false;
  }

  game.addExecution(new UpgradeStructureExecution(player, unit.id()));
  return true;
}

export function findBuildTile(
  game: Game,
  player: Player,
  type: UnitType,
): TileRef | null {
  if (!BUILD_SEARCH_TYPES.includes(type as NeuralBuildUnitType)) {
    return null;
  }

  for (const tile of buildCandidates(game, player, type)) {
    if (player.canBuild(type, tile) !== false) {
      return tile;
    }
  }

  return null;
}

export function findUpgradeUnit(player: Player): Unit | null {
  const units = player
    .units(...UPGRADE_PRIORITY)
    .filter((unit) => player.canUpgradeUnit(unit))
    .sort(
      (left, right) =>
        left.level() - right.level() ||
        UPGRADE_PRIORITY.indexOf(
          left.type() as (typeof UPGRADE_PRIORITY)[number],
        ) -
          UPGRADE_PRIORITY.indexOf(
            right.type() as (typeof UPGRADE_PRIORITY)[number],
          ),
    );

  return units[0] ?? null;
}

function buildCandidates(
  game: Game,
  player: Player,
  type: UnitType,
): TileRef[] {
  if (type === UnitType.Warship) {
    return waterBuildCandidates(game, player);
  }

  const candidates: TileRef[] = [];
  const seen = new Set<TileRef>();
  const push = (tile: TileRef | null | undefined) => {
    if (tile === undefined || tile === null || seen.has(tile)) {
      return;
    }
    seen.add(tile);
    candidates.push(tile);
  };

  if (type === UnitType.DefensePost || type === UnitType.SAMLauncher) {
    for (const tile of player.borderTiles()) {
      push(tile);
    }
  }

  if (type === UnitType.Port) {
    for (const tile of player.borderTiles()) {
      if (game.isShore(tile)) {
        push(tile);
      }
    }
  }

  push(centerTile(game, player));
  for (const tile of sampledOwnedTiles(player)) {
    if (type !== UnitType.Port || game.isShore(tile)) {
      push(tile);
    }
  }
  for (const tile of player.borderTiles()) {
    push(tile);
  }

  return candidates;
}

function waterBuildCandidates(game: Game, player: Player): TileRef[] {
  const candidates: TileRef[] = [];
  const seen = new Set<TileRef>();
  const push = (tile: TileRef | null | undefined) => {
    if (
      tile === undefined ||
      tile === null ||
      seen.has(tile) ||
      !game.isWater(tile)
    ) {
      return;
    }
    seen.add(tile);
    candidates.push(tile);
  };

  for (const port of player.units(UnitType.Port)) {
    if (!port.isActive() || port.isUnderConstruction()) {
      continue;
    }
    for (const tile of game.bfs(port.tile(), (gm, tile) => {
      return game.manhattanDist(port.tile(), tile) <= WATER_SEARCH_RADIUS;
    })) {
      push(tile);
    }
  }

  for (const tile of player.borderTiles()) {
    game.forEachNeighbor(tile, (neighbor) => push(neighbor));
  }

  return candidates;
}

function sampledOwnedTiles(player: Player): TileRef[] {
  const owned = Array.from(player.tiles());
  if (owned.length <= STRUCTURE_SEARCH_SAMPLE_LIMIT) {
    return owned;
  }

  const step = Math.ceil(owned.length / STRUCTURE_SEARCH_SAMPLE_LIMIT);
  const sampled: TileRef[] = [];
  for (let index = 0; index < owned.length; index += step) {
    sampled.push(owned[index]);
  }
  return sampled;
}

export function strategicUnitScore(player: Player): number {
  return (
    player.unitCount(UnitType.City) * 2 +
    player.unitCount(UnitType.Factory) * 2 +
    player.unitCount(UnitType.Port) * 2 +
    player.unitCount(UnitType.DefensePost) +
    player.unitCount(UnitType.SAMLauncher) * 3 +
    player.unitCount(UnitType.MissileSilo) * 4 +
    player.unitCount(UnitType.Warship) * 3
  );
}

export function structureCount(player: Player): number {
  return player.units(...Structures.types).length;
}
