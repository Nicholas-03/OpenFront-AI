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
  nearestOpponent,
  opponentsOf,
  rankPlayer,
  strategicUnitScore,
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

export interface TrainingEnvOptions {
  decisionInterval: number;
  maxTicks: number;
  mapName: string;
  opponentAI: TrainingOpponentAI;
  opponents: number;
  seed: number;
}

export interface StepResult {
  done: boolean;
  reward: number;
}

export interface EpisodeSummary {
  alive: boolean;
  ownedTiles: number;
  rank: number;
  reward: number;
  ticks: number;
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
  private totalReward = 0;

  constructor(private readonly options: TrainingEnvOptions) {
    this.rng = new SeededRandom(options.seed);
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

  async reset(episode: number): Promise<number[]> {
    this.totalReward = 0;
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
    const opponentModes = this.opponentModes();
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
    const beforeTiles = this.learner.numTilesOwned();
    const beforeTroops = this.learner.troops();
    const beforeRank = this.rank(this.learner);
    const beforeStrategicUnits = strategicUnitScore(this.learner);

    const actionSucceeded = this.applyAction(
      this.learner,
      TRAINING_ACTIONS[actionIndex],
    );
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
    const afterRank = this.rank(this.learner);
    const afterStrategicUnits = strategicUnitScore(this.learner);
    const tileReward = (afterTiles - beforeTiles) / 35;
    const troopReward = (afterTroops - beforeTroops) / 50_000;
    const rankReward = (beforeRank - afterRank) * 0.5;
    const unitReward = (afterStrategicUnits - beforeStrategicUnits) * 0.25;
    const aliveReward = this.learner.isAlive() ? 0.03 : -4;
    const invalidActionPenalty = actionSucceeded ? 0 : -0.08;
    const reward =
      tileReward +
      troopReward +
      rankReward +
      unitReward +
      aliveReward +
      invalidActionPenalty;

    this.totalReward += reward;

    return {
      reward,
      done: this.done(),
    };
  }

  state(): number[] {
    return buildNeuralBotInput({
      game: this.game,
      player: this.learner,
      opponentBaseline: this.options.opponents,
      tickHorizon: this.options.maxTicks,
    });
  }

  summary(): EpisodeSummary {
    return {
      alive: this.learner.isAlive(),
      ownedTiles: this.learner.numTilesOwned(),
      rank: this.rank(this.learner),
      reward: this.totalReward,
      ticks: this.game.ticks(),
    };
  }

  private applyAction(player: Player, action: TrainingAction): boolean {
    return tryNeuralBotAction(this.game, player, action);
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

  private opponentModes(): ResolvedOpponentAI[] {
    return Array.from({ length: this.options.opponents }, (_, index) => {
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
    const margin = 6;
    const samples = 600;

    for (let index = 0; index < count; index++) {
      let bestTile: TileRef | null = null;
      let bestDistance = -1;
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
            ? Number.MAX_SAFE_INTEGER
            : Math.min(
                ...centers.map((center) =>
                  this.game.manhattanDist(center, tile),
                ),
              );
        if (distance > bestDistance) {
          bestDistance = distance;
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
    return (
      this.game.ticks() >= this.options.maxTicks ||
      !this.learner.isAlive() ||
      this.game.players().length <= 1
    );
  }
}
