import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import {
  buildNeuralBotInput,
  NEURAL_BOT_INPUT_SIZE,
} from "../core/ai/NeuralBotFeatures";
import {
  NEURAL_BOT_ACTIONS,
  NeuralBotAction,
} from "../core/ai/NeuralBotPolicy";
import {
  borderStats,
  byStrength,
  isNeuralBotActionAvailable,
  nearestOpponent,
  opponentsOf,
  rankPlayer,
  tryNeuralBotAction,
} from "../core/ai/NeuralBotTactics";
import { Config } from "../core/configuration/Config";
import { NationExecution } from "../core/execution/NationExecution";
import { SpawnExecution } from "../core/execution/SpawnExecution";
import { getSpawnTiles } from "../core/execution/Util";
import {
  Difficulty,
  Game,
  GameMapSize,
  GameMapType,
  GameMode,
  GameType,
  Gold,
  Nation,
  Player,
  PlayerInfo,
  PlayerType,
  TerraNullius,
  UnitType,
} from "../core/game/Game";
import { createGame } from "../core/game/GameImpl";
import { TileRef } from "../core/game/GameMap";
import { genTerrainFromBin, MapManifest } from "../core/game/TerrainMapLoader";
import { UserSettings } from "../core/game/UserSettings";
import { GameConfig } from "../core/Schemas";

export const TRAINING_ACTIONS = NEURAL_BOT_ACTIONS;
export type TrainingAction = NeuralBotAction;
export const TRAINING_OPPONENT_AI = [
  "heuristic",
  "tribe",
  "nation",
  "mixed",
] as const;
export type TrainingOpponentAI = (typeof TRAINING_OPPONENT_AI)[number];

type ResolvedOpponentAI = Exclude<TrainingOpponentAI, "mixed">;
const MIXED_OPPONENT_AI_ROTATION = [
  "tribe",
  "nation",
  "heuristic",
] as const satisfies readonly ResolvedOpponentAI[];
const ATTACK_ACTIONS = new Set<TrainingAction>([
  "attack_nearest",
  "attack_weakest",
  "attack_strongest",
]);
const ALLIANCE_REQUEST_ACTIONS = new Set<TrainingAction>([
  "request_alliance_nearest",
  "request_alliance_strongest",
]);
const ATTACK_OR_EXPAND_ACTIONS = [
  "expand",
  "attack_nearest",
  "attack_weakest",
  "attack_strongest",
] as const satisfies readonly TrainingAction[];

interface OpponentRewardStats {
  alive: number;
  tiles: number;
  troops: number;
}

export interface RewardBreakdown {
  alive: number;
  alliance: number;
  attackAction: number;
  defensiveBuildPenalty: number;
  diplomacyDelayPenalty: number;
  elimination: number;
  enemyTile: number;
  enemyTroop: number;
  gold: number;
  invalidActionPenalty: number;
  missedAttackPenalty: number;
  navalBuild: number;
  passiveBorderPenalty: number;
  rank: number;
  tile: number;
  total: number;
  troop: number;
  unit: number;
  win: number;
}

export interface TrainingEnvOptions {
  decisionInterval: number;
  maxTicks: number;
  mapName: string;
  opponentAI: TrainingOpponentAI;
  opponentRampEpisode?: number;
  opponentStart?: number;
  opponents: number;
  seed: number;
}

export interface StepResult {
  actionSucceeded: boolean;
  done: boolean;
  reward: number;
  rewardBreakdown: RewardBreakdown;
}

export interface EpisodeSummary {
  alive: boolean;
  alliances: number;
  aliveOpponents: number;
  ownedTiles: number;
  rank: number;
  reward: number;
  ticks: number;
  won: boolean;
}

class TrainingConfig extends Config {
  disableNavMesh(): boolean {
    return true;
  }

  minDistanceBetweenPlayers(): number {
    return 12;
  }

  spawnImmunityDuration(): number {
    return 0;
  }

  nationSpawnImmunityDuration(): number {
    return 0;
  }

  attackAmount(attacker: Player, defender: Player | TerraNullius): number {
    const ratio = defender.isPlayer() ? 0.18 : 0.12;
    return Math.max(250, Math.floor(attacker.troops() * ratio));
  }

  attackTilesPerTick(
    attackTroops: number,
    attacker: Player,
    defender: Player | TerraNullius,
    numAdjacentTilesWithEnemy: number,
  ): number {
    return Math.max(
      1,
      super.attackTilesPerTick(
        attackTroops,
        attacker,
        defender,
        numAdjacentTilesWithEnemy,
      ),
    );
  }

  goldAdditionRate(player: Player): Gold {
    return BigInt(player.type() === PlayerType.Human ? 1000 : 500);
  }
}

class SeededRandom {
  private state: number;

  constructor(seed: number) {
    this.state = seed >>> 0;
  }

  next(): number {
    this.state = (1664525 * this.state + 1013904223) >>> 0;
    return this.state / 0x100000000;
  }

  int(min: number, maxExclusive: number): number {
    return Math.floor(this.next() * (maxExclusive - min)) + min;
  }
}

function createTrainingConfig(): GameConfig {
  return {
    gameMap: GameMapType.Asia,
    gameMapSize: GameMapSize.Normal,
    gameMode: GameMode.FFA,
    gameType: GameType.Public,
    difficulty: Difficulty.Medium,
    nations: "disabled",
    donateGold: false,
    donateTroops: false,
    bots: 0,
    infiniteGold: false,
    infiniteTroops: false,
    instantBuild: false,
    randomSpawn: false,
    disableNavMesh: true,
    spawnImmunityDuration: 0,
  };
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function trainingUnitScore(player: Player): number {
  return (
    player.unitCount(UnitType.City) * 3 +
    player.unitCount(UnitType.Factory) * 3 +
    player.unitCount(UnitType.Port) * 1.5 +
    player.unitCount(UnitType.Warship) * 2 +
    player.unitCount(UnitType.MissileSilo) * 1.5 +
    player.unitCount(UnitType.SAMLauncher) * 0.35 +
    player.unitCount(UnitType.DefensePost) * 0.15
  );
}

function suppressExpectedTrainingWarnings(callback: () => void): void {
  const originalWarn = console.warn;
  console.warn = (...args: unknown[]) => {
    if (
      typeof args[0] === "string" &&
      (args[0].startsWith("cannot build") ||
        args[0].startsWith("cannot build construction"))
    ) {
      return;
    }
    originalWarn(...args);
  };
  try {
    callback();
  } finally {
    console.warn = originalWarn;
  }
}

export class OpenFrontTrainingEnv {
  readonly inputSize = NEURAL_BOT_INPUT_SIZE;

  private game: Game;
  private learner: Player;
  private rng: SeededRandom;
  private mapBin: Uint8Array;
  private miniMapBin: Uint8Array;
  private manifest: MapManifest;
  private heuristicOpponents: Player[] = [];
  private currentOpponentCount: number;
  private totalReward = 0;
  private positiveUnitReward = 0;

  constructor(private readonly options: TrainingEnvOptions) {
    this.rng = new SeededRandom(options.seed);
    this.currentOpponentCount = options.opponents;
  }

  async init(): Promise<void> {
    const dirname = path.dirname(fileURLToPath(import.meta.url));
    const repoRoot = path.resolve(dirname, "../..");
    const mapDir = path.join(
      repoRoot,
      "tests",
      "testdata",
      "maps",
      this.options.mapName,
    );

    this.mapBin = new Uint8Array(fs.readFileSync(path.join(mapDir, "map.bin")));
    this.miniMapBin = new Uint8Array(
      fs.readFileSync(path.join(mapDir, "map4x.bin")),
    );
    this.manifest = JSON.parse(
      fs.readFileSync(path.join(mapDir, "manifest.json"), "utf8"),
    ) as MapManifest;
  }

  mapDescription(): string {
    return `${this.manifest.name} ${this.manifest.map.width}x${this.manifest.map.height} land=${this.manifest.map.num_land_tiles}`;
  }

  async reset(episode: number): Promise<number[]> {
    this.totalReward = 0;
    this.positiveUnitReward = 0;
    this.currentOpponentCount = this.opponentsForEpisode(episode);
    this.rng = new SeededRandom(this.options.seed + episode * 9973);

    const gameMap = await genTerrainFromBin(
      this.manifest.map,
      new Uint8Array(this.mapBin),
    );
    const miniGameMap = await genTerrainFromBin(
      this.manifest.map4x,
      new Uint8Array(this.miniMapBin),
    );
    const config = new TrainingConfig(
      createTrainingConfig(),
      new UserSettings(),
      false,
    );
    const gameID = `train-${episode}`;
    const opponentModes = this.opponentModes(this.currentOpponentCount);
    const learnerInfo = new PlayerInfo(
      "Learner",
      PlayerType.Human,
      null,
      "learner",
    );
    const heuristicInfos: PlayerInfo[] = [];
    const tribeInfos: PlayerInfo[] = [];
    const nationInfos: PlayerInfo[] = [];

    opponentModes.forEach((opponentAI, index) => {
      const opponentNumber = index + 1;
      switch (opponentAI) {
        case "heuristic":
          heuristicInfos.push(
            new PlayerInfo(
              `Heuristic ${opponentNumber}`,
              PlayerType.Human,
              null,
              `heuristic-${opponentNumber}`,
            ),
          );
          return;
        case "tribe":
          tribeInfos.push(
            new PlayerInfo(
              `Tribe ${opponentNumber}`,
              PlayerType.Bot,
              null,
              `tribe-${opponentNumber}`,
            ),
          );
          return;
        case "nation":
          nationInfos.push(
            new PlayerInfo(
              `Nation ${opponentNumber}`,
              PlayerType.Nation,
              null,
              `nation-${opponentNumber}`,
            ),
          );
          return;
      }
    });

    const players = [learnerInfo, ...heuristicInfos, ...tribeInfos];
    const nations = nationInfos.map(
      (playerInfo) => new Nation(undefined, playerInfo),
    );
    const spawnInfos = [
      learnerInfo,
      ...heuristicInfos,
      ...tribeInfos,
      ...nationInfos,
    ];

    const originalLog = console.log;
    console.log = (...args: unknown[]) => {
      if (
        typeof args[0] === "string" &&
        args[0].startsWith("[GameImpl] Constructor total:")
      ) {
        return;
      }
      originalLog(...args);
    };
    try {
      this.game = createGame(players, nations, gameMap, miniGameMap, config);
    } finally {
      console.log = originalLog;
    }

    const centers = this.findSpawnCenters(spawnInfos.length);
    this.game.addExecution(
      ...spawnInfos.map(
        (player, index) => new SpawnExecution(gameID, player, centers[index]),
      ),
    );
    this.game.executeNextTick();
    this.game.endSpawnPhase();
    this.game.addExecution(
      ...nations.map((nation) => new NationExecution(gameID, nation)),
    );
    this.game.executeNextTick();

    this.learner = this.game.player("learner");
    this.heuristicOpponents = heuristicInfos.map((playerInfo) =>
      this.game.player(playerInfo.id),
    );
    return this.state();
  }

  step(actionIndex: number): StepResult {
    const action = TRAINING_ACTIONS[actionIndex];
    const beforeTiles = this.learner.numTilesOwned();
    const beforeTroops = this.learner.troops();
    const beforeGold = Number(this.learner.gold());
    const beforeRank = this.rank(this.learner);
    const beforeTrainingUnits = trainingUnitScore(this.learner);
    const beforeAlliances = this.learner.alliances().length;
    const beforeOpponents = this.opponentRewardStats();
    const beforeBorders = this.borderStats(this.learner);
    const attackAvailable = [...ATTACK_ACTIONS].some((candidate) =>
      isNeuralBotActionAvailable(this.game, this.learner, candidate),
    );
    const attackOrExpandAvailable = ATTACK_OR_EXPAND_ACTIONS.some((candidate) =>
      isNeuralBotActionAvailable(this.game, this.learner, candidate),
    );

    const actionSucceeded = this.applyAction(this.learner, action);
    for (const opponent of this.heuristicOpponents) {
      this.applyAction(opponent, this.heuristicAction(opponent));
    }

    suppressExpectedTrainingWarnings(() => {
      for (let i = 0; i < this.options.decisionInterval; i++) {
        this.game.executeNextTick();
      }
    });

    const afterTiles = this.learner.numTilesOwned();
    const afterTroops = this.learner.troops();
    const afterGold = Number(this.learner.gold());
    const afterRank = this.rank(this.learner);
    const afterTrainingUnits = trainingUnitScore(this.learner);
    const afterAlliances = this.learner.alliances().length;
    const afterOpponents = this.opponentRewardStats();
    const tileReward = (afterTiles - beforeTiles) / 25;
    const enemyTileReward =
      Math.max(0, beforeOpponents.tiles - afterOpponents.tiles) / 18;
    const troopReward = clamp(
      (afterTroops - beforeTroops) / 120_000,
      -0.15,
      0.15,
    );
    const enemyTroopReward =
      Math.max(0, beforeOpponents.troops - afterOpponents.troops) / 80_000;
    const goldReward = clamp((afterGold - beforeGold) / 300_000, -0.08, 0.12);
    const rankReward = (beforeRank - afterRank) * 0.5;
    const unitReward = this.cappedPositiveUnitReward(
      (afterTrainingUnits - beforeTrainingUnits) * 0.12,
    );
    const eliminationReward =
      Math.max(0, beforeOpponents.alive - afterOpponents.alive) * 8;
    const winReward = this.learnerWon() ? 24 : 0;
    const attackActionReward =
      actionSucceeded && ATTACK_ACTIONS.has(action) ? 0.55 : 0;
    const allianceReward = Math.max(0, afterAlliances - beforeAlliances) * 1.5;
    const passiveBorderPenalty =
      action === "hold" && attackOrExpandAvailable
        ? beforeBorders.enemy > 0
          ? -0.75
          : -0.3
        : action === "hold" && beforeBorders.enemy > 0
          ? -0.25
          : 0;
    const missedAttackPenalty =
      attackAvailable && !ATTACK_ACTIONS.has(action) ? -0.18 : 0;
    const diplomacyDelayPenalty =
      attackAvailable && actionSucceeded && ALLIANCE_REQUEST_ACTIONS.has(action)
        ? -0.3
        : 0;
    const navalBuildReward =
      actionSucceeded && action === "build_port"
        ? 0.7
        : actionSucceeded && action === "build_warship"
          ? 1.1
          : 0;
    const defensiveBuildPenalty =
      actionSucceeded &&
      (action === "build_defense_post" || action === "build_sam_launcher")
        ? -0.12
        : 0;
    const aliveReward = this.learner.isAlive() ? 0.008 : -6;
    const invalidActionPenalty = actionSucceeded ? 0 : -0.12;
    const rewardBreakdown: RewardBreakdown = {
      alive: aliveReward,
      alliance: allianceReward,
      attackAction: attackActionReward,
      defensiveBuildPenalty,
      diplomacyDelayPenalty,
      elimination: eliminationReward,
      enemyTile: enemyTileReward,
      enemyTroop: enemyTroopReward,
      gold: goldReward,
      invalidActionPenalty,
      missedAttackPenalty,
      navalBuild: navalBuildReward,
      passiveBorderPenalty,
      rank: rankReward,
      tile: tileReward,
      total: 0,
      troop: troopReward,
      unit: unitReward,
      win: winReward,
    };
    const reward = Object.entries(rewardBreakdown)
      .filter(([key]) => key !== "total")
      .reduce((acc, [, value]) => acc + value, 0);
    rewardBreakdown.total = reward;

    this.totalReward += reward;

    return {
      actionSucceeded,
      reward,
      rewardBreakdown,
      done: this.done(),
    };
  }

  state(): number[] {
    return buildNeuralBotInput({
      game: this.game,
      player: this.learner,
      opponentBaseline: this.currentOpponentCount,
      tickHorizon: this.options.maxTicks,
    });
  }

  validActionMask(): boolean[] {
    return TRAINING_ACTIONS.map((action) =>
      isNeuralBotActionAvailable(this.game, this.learner, action),
    );
  }

  summary(): EpisodeSummary {
    const opponentStats = this.opponentRewardStats();
    return {
      alive: this.learner.isAlive(),
      alliances: this.learner.alliances().length,
      aliveOpponents: opponentStats.alive,
      ownedTiles: this.learner.numTilesOwned(),
      rank: this.rank(this.learner),
      reward: this.totalReward,
      ticks: this.game.ticks(),
      won: this.learnerWon(),
    };
  }

  private applyAction(player: Player, action: TrainingAction): boolean {
    return tryNeuralBotAction(this.game, player, action);
  }

  private cappedPositiveUnitReward(delta: number): number {
    if (delta <= 0) {
      return delta;
    }

    const cap = 4;
    const remaining = Math.max(0, cap - this.positiveUnitReward);
    const reward = Math.min(delta, remaining);
    this.positiveUnitReward += reward;
    return reward;
  }

  private opponentRewardStats(): OpponentRewardStats {
    return this.game
      .allPlayers()
      .filter((player) => player !== this.learner)
      .reduce(
        (stats, player) => ({
          alive: stats.alive + (player.isAlive() ? 1 : 0),
          tiles: stats.tiles + player.numTilesOwned(),
          troops: stats.troops + (player.isAlive() ? player.troops() : 0),
        }),
        { alive: 0, tiles: 0, troops: 0 },
      );
  }

  private learnerWon(): boolean {
    const winner = this.game.getWinner();
    return (
      winner === this.learner ||
      (winner === null &&
        this.learner.isAlive() &&
        this.game.players().length === 1)
    );
  }

  private heuristicAction(player: Player): TrainingAction {
    const borderStats = this.borderStats(player);
    if (borderStats.neutral > 0 && player.outgoingAttacks().length < 2) {
      return "expand";
    }
    if (player.troops() < 8_000) {
      return "hold";
    }
    return this.rng.next() < 0.65 ? "attack_weakest" : "attack_strongest";
  }

  private opponentsForEpisode(episode: number): number {
    const rampEpisode = this.options.opponentRampEpisode ?? 0;
    const start = this.options.opponentStart ?? this.options.opponents;
    if (episode < rampEpisode) {
      return Math.max(1, Math.min(this.options.opponents, start));
    }
    return this.options.opponents;
  }

  private opponentModes(opponentCount: number): ResolvedOpponentAI[] {
    return Array.from({ length: opponentCount }, (_, index) => {
      switch (this.options.opponentAI) {
        case "heuristic":
        case "tribe":
        case "nation":
          return this.options.opponentAI;
        case "mixed":
          return MIXED_OPPONENT_AI_ROTATION[
            index % MIXED_OPPONENT_AI_ROTATION.length
          ];
      }
    });
  }

  private findSpawnCenters(count: number): TileRef[] {
    const centers: TileRef[] = [];
    const smallestSide = Math.min(this.game.width(), this.game.height());
    const margin = Math.max(1, Math.min(6, Math.floor(smallestSide / 5)));
    const samples = 1200;
    const hasWater = this.mapHasWater();

    for (let index = 0; index < count; index++) {
      let bestTile: TileRef | null = null;
      let bestScore = Number.NEGATIVE_INFINITY;
      for (let attempt = 0; attempt < samples; attempt++) {
        const x = this.rng.int(
          margin,
          Math.max(margin + 1, this.game.width() - margin),
        );
        const y = this.rng.int(
          margin,
          Math.max(margin + 1, this.game.height() - margin),
        );
        const tile = this.game.ref(x, y);
        if (!this.game.isLand(tile) || !getSpawnTiles(this.game, tile, true)) {
          continue;
        }
        const distance =
          centers.length === 0
            ? 0
            : Math.min(
                ...centers.map((center) =>
                  this.game.manhattanDist(center, tile),
                ),
              );
        const shoreScore = hasWater ? this.shoreSpawnScore(tile) : 0;
        const score = distance + shoreScore;
        if (score > bestScore) {
          bestScore = score;
          bestTile = tile;
        }
      }

      if (bestTile === null) {
        throw new Error("Unable to find enough valid spawn centers");
      }
      centers.push(bestTile);
    }

    return centers;
  }

  private mapHasWater(): boolean {
    for (let y = 0; y < this.game.height(); y++) {
      for (let x = 0; x < this.game.width(); x++) {
        if (this.game.isWater(this.game.ref(x, y))) {
          return true;
        }
      }
    }
    return false;
  }

  private shoreSpawnScore(tile: TileRef): number {
    if (this.game.isShore(tile)) {
      return 24;
    }

    let bestDistance = Number.POSITIVE_INFINITY;
    for (let dy = -10; dy <= 10; dy++) {
      for (let dx = -10; dx <= 10; dx++) {
        const x = this.game.x(tile) + dx;
        const y = this.game.y(tile) + dy;
        if (!this.game.isValidCoord(x, y)) {
          continue;
        }
        const candidate = this.game.ref(x, y);
        if (this.game.isLand(candidate) && this.game.isShore(candidate)) {
          bestDistance = Math.min(bestDistance, Math.abs(dx) + Math.abs(dy));
        }
      }
    }

    return Number.isFinite(bestDistance) ? Math.max(0, 20 - bestDistance) : 0;
  }

  private opponents(): Player[] {
    return this.opponentsOf(this.learner);
  }

  private opponentsOf(player: Player): Player[] {
    return opponentsOf(this.game, player);
  }

  private byStrength(players: Player[]): Player[] {
    return byStrength(players);
  }

  private nearestOpponent(player: Player): Player | null {
    return nearestOpponent(this.game, player);
  }

  private borderStats(player: Player): { enemy: number; neutral: number } {
    return borderStats(this.game, player);
  }

  private rank(player: Player): number {
    return rankPlayer(this.game, player);
  }

  private done(): boolean {
    const matchFinished =
      this.game.getWinner() !== null ||
      !this.learner.isAlive() ||
      this.game.players().length <= 1;
    if (matchFinished) {
      return true;
    }

    if (this.options.opponentAI !== "heuristic") {
      return false;
    }

    return this.game.ticks() >= this.options.maxTicks;
  }
}
