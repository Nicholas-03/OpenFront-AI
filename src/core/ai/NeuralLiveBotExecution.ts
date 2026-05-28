import { SpawnExecution } from "../execution/SpawnExecution";
import { getSpawnTiles } from "../execution/Util";
import { Execution, Game, Player, PlayerID } from "../game/Game";
import { TileRef } from "../game/GameMap";
import { PseudoRandom } from "../PseudoRandom";
import { simpleHash } from "../Util";
import {
  buildNeuralBotInput,
  NEURAL_BOT_LONG_GAME_TICK_HORIZON,
} from "./NeuralBotFeatures";
import {
  forwardNeuralPolicy,
  LoadedNeuralPolicy,
  NEURAL_BOT_ACTIONS,
  NeuralBotAction,
} from "./NeuralBotPolicy";
import {
  isNeuralBotActionAvailable,
  MAX_NEURAL_PARALLEL_ATTACKS,
  tryNeuralBotAction,
} from "./NeuralBotTactics";

const DECISION_INTERVAL_TICKS = 20;

export class NeuralLiveBotExecution implements Execution {
  private mg: Game;
  private active = true;
  private lastDecisionTick = Number.NEGATIVE_INFINITY;
  private spawnQueued = false;
  private random: PseudoRandom;

  constructor(
    private readonly gameID: string,
    private readonly playerID: PlayerID,
    private readonly policy: LoadedNeuralPolicy | null,
  ) {
    this.random = new PseudoRandom(simpleHash(`${gameID}:${playerID}`));
  }

  activeDuringSpawnPhase(): boolean {
    return true;
  }

  init(mg: Game): void {
    this.mg = mg;
  }

  tick(ticks: number): void {
    const player = this.mg.player(this.playerID);
    if (!player.isAlive() && player.hasSpawned()) {
      this.active = false;
      return;
    }

    if (!player.hasSpawned()) {
      this.trySpawn(player);
      return;
    }

    if (this.mg.inSpawnPhase()) {
      return;
    }

    if (ticks - this.lastDecisionTick < DECISION_INTERVAL_TICKS) {
      return;
    }

    this.lastDecisionTick = ticks;
    this.decide(player);
  }

  isActive(): boolean {
    return this.active;
  }

  private trySpawn(player: Player): void {
    if (this.spawnQueued || this.mg.inSpawnPhase()) {
      return;
    }

    const tile = this.findSpawnTile();
    if (tile === null) {
      return;
    }

    this.spawnQueued = true;
    this.mg.addExecution(new SpawnExecution(this.gameID, player.info(), tile));
  }

  private findSpawnTile(): TileRef | null {
    const spawnedPlayers = this.mg
      .players()
      .filter((player) => player.spawnTile() !== undefined);
    let bestTile: TileRef | null = null;
    let bestScore = Number.NEGATIVE_INFINITY;
    const margin = 6;

    for (let attempt = 0; attempt < 700; attempt++) {
      const x = this.random.nextInt(
        margin,
        Math.max(margin + 1, this.mg.width() - margin),
      );
      const y = this.random.nextInt(
        margin,
        Math.max(margin + 1, this.mg.height() - margin),
      );
      const tile = this.mg.ref(x, y);
      if (!this.mg.isLand(tile) || !getSpawnTiles(this.mg, tile, true)) {
        continue;
      }

      const distanceScore =
        spawnedPlayers.length === 0
          ? 1
          : Math.min(
              ...spawnedPlayers.map((player) =>
                this.mg.manhattanDist(tile, player.spawnTile()!),
              ),
            );
      const edgePenalty = this.mg.isOnEdgeOfMap(tile) ? 100 : 0;
      const score = distanceScore - edgePenalty;

      if (score > bestScore) {
        bestScore = score;
        bestTile = tile;
      }
    }

    return bestTile;
  }

  private decide(player: Player): void {
    const rankedActions = this.policy
      ? this.rankedPolicyActions(player)
      : ([
          "expand",
          "build_city",
          "build_defense_post",
          "build_port",
          "attack_weakest",
          "attack_nearest",
          "upgrade_structure",
          "hold",
        ] as const);

    const fallbacks: readonly NeuralBotAction[] = [
      "expand",
      "build_city",
      "build_factory",
      "build_defense_post",
      "build_port",
      "build_warship",
      "upgrade_structure",
      "attack_weakest",
      "attack_nearest",
      "attack_strongest",
      "hold",
    ];

    const tried = new Set<NeuralBotAction>();
    for (const action of [...rankedActions, ...fallbacks]) {
      if (tried.has(action)) {
        continue;
      }
      tried.add(action);
      if (this.tryAction(player, action)) {
        return;
      }
    }
  }

  private rankedPolicyActions(player: Player): NeuralBotAction[] {
    if (!this.policy) {
      return ["expand", "build_city", "attack_weakest", "hold"];
    }

    const pass = forwardNeuralPolicy(this.policy, this.state(player));
    return pass.probs
      .map((probability, index) => ({
        probability,
        action: this.policy!.actions[index],
      }))
      .sort((left, right) => right.probability - left.probability)
      .map(({ action }) => action)
      .filter((action): action is NeuralBotAction =>
        NEURAL_BOT_ACTIONS.includes(action as NeuralBotAction),
      )
      .filter((action) =>
        isNeuralBotActionAvailable(
          this.mg,
          player,
          action,
          MAX_NEURAL_PARALLEL_ATTACKS,
        ),
      );
  }

  private tryAction(player: Player, action: NeuralBotAction): boolean {
    return tryNeuralBotAction(
      this.mg,
      player,
      action,
      MAX_NEURAL_PARALLEL_ATTACKS,
    );
  }

  private state(player: Player): number[] {
    return buildNeuralBotInput({
      game: this.mg,
      player,
      opponentBaseline: Math.max(1, this.mg.allPlayers().length - 1),
      tickHorizon: this.liveTickHorizon(),
    });
  }

  private liveTickHorizon(): number {
    const maxTimerMinutes = this.mg.config().gameConfig().maxTimerValue;
    return maxTimerMinutes === null || maxTimerMinutes === undefined
      ? NEURAL_BOT_LONG_GAME_TICK_HORIZON
      : Math.max(NEURAL_BOT_LONG_GAME_TICK_HORIZON, maxTimerMinutes * 60 * 10);
  }
}
