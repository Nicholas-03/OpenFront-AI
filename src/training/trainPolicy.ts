import fs from "node:fs";
import path from "node:path";

import {
  NEURAL_BOT_GLOBAL_MAP_SIZE,
  NEURAL_BOT_LOCAL_MAP_SIZE,
  NEURAL_BOT_LONG_GAME_TICK_HORIZON,
  NEURAL_BOT_MAP_CHANNEL_COUNT,
  NEURAL_BOT_SCALAR_INPUT_SIZE,
  NEURAL_BOT_SPATIAL_POOL_SIZE,
} from "../core/ai/NeuralBotFeatures";
import { NEURAL_BOT_HIDDEN_LAYERS } from "../core/ai/NeuralBotPolicy";
import {
  NeuralPolicy,
  PolicyGradientSample,
  SerializedNeuralPolicy,
} from "./NeuralPolicy";
import {
  OpenFrontTrainingEnv,
  RewardBreakdown,
  TRAINING_ACTIONS,
  TRAINING_OPPONENT_AI,
  TrainingAction,
  TrainingOpponentAI,
} from "./OpenFrontTrainingEnv";
import { ParallelRolloutPool } from "./ParallelRolloutPool";
import { EpisodeRollout, runPolicyEpisode, Transition } from "./Rollout";
import { SeededRandom } from "./TrainingRandom";

const DEFAULT_LEARNING_RATE = 0.001;

type CurriculumMode = "auto" | "off";
type RewardBreakdownKey = keyof RewardBreakdown;

interface CurriculumPhase {
  actions: readonly TrainingAction[];
  name: string;
}

const FULL_ACTION_PHASE: CurriculumPhase = {
  name: "full",
  actions: TRAINING_ACTIONS,
};

const CURRICULUM_PHASES: Array<CurriculumPhase & { throughEpisode: number }> = [
  {
    name: "expand_attack",
    throughEpisode: 2_000,
    actions: [
      "expand",
      "attack_nearest",
      "attack_weakest",
      "attack_strongest",
      "accept_alliance",
      "hold",
    ],
  },
  {
    name: "economy",
    throughEpisode: 5_000,
    actions: [
      "expand",
      "attack_nearest",
      "attack_weakest",
      "attack_strongest",
      "build_city",
      "build_factory",
      "upgrade_structure",
      "accept_alliance",
      "request_alliance_nearest",
      "request_alliance_strongest",
      "hold",
    ],
  },
  {
    name: "navy_economy",
    throughEpisode: 8_000,
    actions: [
      "expand",
      "attack_nearest",
      "attack_weakest",
      "attack_strongest",
      "build_city",
      "build_factory",
      "build_port",
      "build_warship",
      "upgrade_structure",
      "accept_alliance",
      "request_alliance_nearest",
      "request_alliance_strongest",
      "hold",
    ],
  },
];

interface SavedTrainingSnapshot {
  actions?: unknown;
  episode?: unknown;
  summary?: unknown;
  policy?: SerializedNeuralPolicy;
}

interface DiagnosticsWindow {
  actionCounts: number[];
  actionSucceeded: number;
  alive: number;
  alliances: number;
  aliveOpponents: number;
  bestReward: number;
  episodes: number;
  ranks: number;
  reward: number;
  rewardBreakdown: Record<RewardBreakdownKey, number>;
  ticks: number;
  tiles: number;
  transitions: number;
  wins: number;
  worstReward: number;
}

function numberArg(name: string, fallback: number): number {
  const index = process.argv.indexOf(name);
  if (index === -1) {
    return fallback;
  }
  const parsed = Number(process.argv[index + 1]);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function positiveIntegerArg(name: string, fallback: number): number {
  return Math.max(1, Math.floor(numberArg(name, fallback)));
}

function optionalNumberArg(name: string): number | undefined {
  const index = process.argv.indexOf(name);
  if (index === -1) {
    return undefined;
  }
  const parsed = Number(process.argv[index + 1]);
  return Number.isFinite(parsed) ? parsed : undefined;
}

function stringArg(name: string, fallback: string): string {
  const index = process.argv.indexOf(name);
  return index === -1 ? fallback : (process.argv[index + 1] ?? fallback);
}

function hasFlag(name: string): boolean {
  return process.argv.includes(name);
}

function opponentAIArg(): TrainingOpponentAI {
  const value = stringArg("--opponentAI", "heuristic");
  if (TRAINING_OPPONENT_AI.includes(value as TrainingOpponentAI)) {
    return value as TrainingOpponentAI;
  }
  throw new Error(
    `Invalid --opponentAI "${value}". Expected one of: ${TRAINING_OPPONENT_AI.join(", ")}`,
  );
}

function curriculumModeArg(): CurriculumMode {
  const value = stringArg("--curriculum", "auto");
  if (value === "auto" || value === "off") {
    return value;
  }
  throw new Error('Invalid --curriculum value. Expected "auto" or "off".');
}

function curriculumPhaseForEpisode(
  episode: number,
  mode: CurriculumMode,
): CurriculumPhase {
  if (mode === "off") {
    return FULL_ACTION_PHASE;
  }

  return (
    CURRICULUM_PHASES.find((phase) => episode <= phase.throughEpisode) ??
    FULL_ACTION_PHASE
  );
}

function actionIndexes(actions: readonly TrainingAction[]): number[] {
  return actions.map((action) => TRAINING_ACTIONS.indexOf(action));
}

function newDiagnosticsWindow(): DiagnosticsWindow {
  return {
    actionCounts: new Array(TRAINING_ACTIONS.length).fill(0),
    actionSucceeded: 0,
    alive: 0,
    alliances: 0,
    aliveOpponents: 0,
    bestReward: Number.NEGATIVE_INFINITY,
    episodes: 0,
    ranks: 0,
    reward: 0,
    rewardBreakdown: zeroRewardBreakdown(),
    ticks: 0,
    tiles: 0,
    transitions: 0,
    wins: 0,
    worstReward: Number.POSITIVE_INFINITY,
  };
}

function zeroRewardBreakdown(): Record<RewardBreakdownKey, number> {
  return {
    alive: 0,
    alliance: 0,
    attackAction: 0,
    defensiveBuildPenalty: 0,
    elimination: 0,
    enemyTile: 0,
    enemyTroop: 0,
    gold: 0,
    invalidActionPenalty: 0,
    passiveBorderPenalty: 0,
    rank: 0,
    tile: 0,
    total: 0,
    troop: 0,
    unit: 0,
    win: 0,
  };
}

function namedActionCounts(counts: readonly number[]): Record<string, number> {
  return Object.fromEntries(
    counts.map((count, index) => [TRAINING_ACTIONS[index], count]),
  );
}

function normalizedActionCounts(
  counts: readonly number[],
  total: number,
): Record<string, number> {
  return Object.fromEntries(
    counts.map((count, index) => [
      TRAINING_ACTIONS[index],
      total === 0 ? 0 : count / total,
    ]),
  );
}

function averageRewardBreakdown(
  breakdown: Record<RewardBreakdownKey, number>,
  episodes: number,
): Record<RewardBreakdownKey, number> {
  return Object.fromEntries(
    Object.entries(breakdown).map(([key, value]) => [
      key,
      episodes === 0 ? 0 : value / episodes,
    ]),
  ) as Record<RewardBreakdownKey, number>;
}

function diagnosticHints(window: DiagnosticsWindow): string[] {
  const hints: string[] = [];
  const actionFractions = normalizedActionCounts(
    window.actionCounts,
    window.transitions,
  );
  const rewardAverage = averageRewardBreakdown(
    window.rewardBreakdown,
    window.episodes,
  );
  const attackFraction =
    (actionFractions.attack_nearest ?? 0) +
    (actionFractions.attack_weakest ?? 0) +
    (actionFractions.attack_strongest ?? 0);

  if (attackFraction < 0.12) {
    hints.push("low_attack_rate");
  }
  if ((actionFractions.hold ?? 0) > 0.35) {
    hints.push("too_many_holds");
  }
  if ((actionFractions.build_defense_post ?? 0) > 0.12) {
    hints.push("defense_post_bias");
  }
  if (window.episodes > 0 && window.tiles / window.episodes < 350) {
    hints.push("poor_expansion");
  }
  if (window.episodes > 0 && window.wins / window.episodes < 0.1) {
    hints.push("low_win_rate");
  }
  if (rewardAverage.alive > Math.max(1, rewardAverage.enemyTile)) {
    hints.push("reward_may_still_favor_survival");
  }
  if (
    window.transitions > 0 &&
    window.actionSucceeded / window.transitions < 0.85
  ) {
    hints.push("many_actions_failed");
  }
  return hints;
}

function appendJsonLine(filePath: string, record: unknown): void {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.appendFileSync(filePath, `${JSON.stringify(record)}\n`);
}

function discountedSamples(
  transitions: Transition[],
  gamma: number,
): PolicyGradientSample[] {
  let running = 0;
  const returns = new Array(transitions.length);
  for (let index = transitions.length - 1; index >= 0; index--) {
    running = transitions[index].reward + gamma * running;
    returns[index] = running;
  }

  const mean =
    returns.reduce((acc, value) => acc + value, 0) /
    Math.max(1, returns.length);
  const variance =
    returns.reduce((acc, value) => acc + (value - mean) ** 2, 0) /
    Math.max(1, returns.length);
  const std = Math.sqrt(variance) || 1;

  return transitions.map((transition, index) => ({
    input: transition.input,
    hidden: transition.hidden,
    activations: transition.activations,
    probs: transition.probs,
    action: transition.action,
    advantage: (returns[index] - mean) / std,
  }));
}

function actionCounts(transitions: Transition[]): string {
  const counts = new Array(TRAINING_ACTIONS.length).fill(0);
  for (const transition of transitions) {
    counts[transition.action]++;
  }
  return counts
    .map((count, index) => `${TRAINING_ACTIONS[index]}=${count}`)
    .join(" ");
}

function formatSeconds(seconds: number): string {
  if (seconds < 60) {
    return `${seconds.toFixed(1)}s`;
  }
  const minutes = Math.floor(seconds / 60);
  const remainder = Math.floor(seconds % 60);
  return `${minutes}m ${remainder}s`;
}

function trainingActionsMatch(actions: unknown): boolean {
  return (
    Array.isArray(actions) &&
    actions.length === TRAINING_ACTIONS.length &&
    actions.every((action, index) => action === TRAINING_ACTIONS[index])
  );
}

function loadRestartPolicy(
  restartPath: string,
  inputSize: number,
  learningRate: number,
  random: SeededRandom,
): {
  policy: NeuralPolicy;
  episode: number;
  summary: unknown;
} | null {
  if (!fs.existsSync(restartPath)) {
    console.warn(
      `--restart was set, but no saved policy exists at ${restartPath}. Starting fresh.`,
    );
    return null;
  }

  try {
    const snapshot = JSON.parse(
      fs.readFileSync(restartPath, "utf8"),
    ) as SavedTrainingSnapshot;

    if (!trainingActionsMatch(snapshot.actions)) {
      console.warn(
        `Saved policy at ${restartPath} uses different actions. Starting fresh.`,
      );
      return null;
    }

    if (snapshot.policy === undefined) {
      console.warn(
        `Saved policy at ${restartPath} does not include neural weights. Starting fresh.`,
      );
      return null;
    }

    const policy = NeuralPolicy.fromJSON(
      snapshot.policy,
      {
        inputSize,
        hiddenLayers: NEURAL_BOT_HIDDEN_LAYERS,
        outputSize: TRAINING_ACTIONS.length,
        learningRate,
      },
      () => random.next(),
    );

    if (policy === null) {
      console.warn(
        `Saved policy at ${restartPath} does not match the current network shape. Starting fresh.`,
      );
      return null;
    }

    return {
      policy,
      episode:
        typeof snapshot.episode === "number" &&
        Number.isFinite(snapshot.episode)
          ? Math.max(0, Math.floor(snapshot.episode))
          : 0,
      summary: snapshot.summary ?? {
        interrupted: false,
        message: "Restarted from a saved policy without a summary.",
      },
    };
  } catch (error) {
    console.warn(
      `Could not load restart policy from ${restartPath}: ${error instanceof Error ? error.message : String(error)}. Starting fresh.`,
    );
    return null;
  }
}

function savePolicySnapshot(
  outputDir: string,
  publicDir: string | null,
  episode: number,
  summary: unknown,
  policy: NeuralPolicy,
): string {
  if (!policy.isStable()) {
    console.warn("policy is unstable; refusing to save this checkpoint");
    return path.join(outputDir, "openfront-policy.json");
  }

  const snapshot = JSON.stringify(
    {
      actions: TRAINING_ACTIONS,
      episode,
      summary,
      policy: policy.toJSON(),
    },
    null,
    2,
  );
  const outputPath = path.join(outputDir, "openfront-policy.json");

  fs.mkdirSync(outputDir, { recursive: true });
  fs.writeFileSync(outputPath, snapshot);
  if (publicDir !== null) {
    const publicPath = path.join(publicDir, "openfront-policy.json");
    fs.mkdirSync(publicDir, { recursive: true });
    fs.writeFileSync(publicPath, snapshot);
  }
  return outputPath;
}

async function main(): Promise<void> {
  const episodesArg = optionalNumberArg("--episodes");
  const trainForever = episodesArg === undefined || episodesArg <= 0;
  const episodes = trainForever ? Number.POSITIVE_INFINITY : episodesArg;
  const maxTicks = numberArg("--maxTicks", NEURAL_BOT_LONG_GAME_TICK_HORIZON);
  const decisionInterval = numberArg("--decisionInterval", 20);
  const opponents = numberArg("--opponents", 3);
  const seed = numberArg("--seed", 42);
  const mapName = stringArg("--map", "training_plains");
  const opponentAI = opponentAIArg();
  const curriculumMode = curriculumModeArg();
  const outputDir = stringArg("--outDir", "training-output");
  const publicDir = hasFlag("--noPublicCopy")
    ? null
    : stringArg("--publicDir", path.join("resources", "training"));
  const logEvery = positiveIntegerArg("--logEvery", 1000);
  const checkpointEvery = positiveIntegerArg("--checkpointEvery", 1000);
  const diagnosticsEvery = positiveIntegerArg("--diagnosticsEvery", 1000);
  const workerCount = positiveIntegerArg("--workers", 1);
  const learningRate = numberArg("--learningRate", DEFAULT_LEARNING_RATE);
  const restart = hasFlag("--restart");
  const restartFrom = stringArg(
    "--restartFrom",
    path.join(outputDir, "openfront-policy.json"),
  );
  const diagnosticsFile = hasFlag("--noDiagnostics")
    ? null
    : stringArg(
        "--diagnosticsFile",
        path.join(outputDir, "training-diagnostics.jsonl"),
      );
  const runId = new Date().toISOString().replace(/[:.]/g, "-");

  const random = new SeededRandom(seed);
  const envOptions = {
    decisionInterval,
    maxTicks,
    mapName,
    opponentAI,
    opponents,
    seed,
  };
  const env = new OpenFrontTrainingEnv(envOptions);
  await env.init();

  let policy = new NeuralPolicy(
    env.inputSize,
    [...NEURAL_BOT_HIDDEN_LAYERS],
    TRAINING_ACTIONS.length,
    learningRate,
    () => random.next(),
  );

  fs.mkdirSync(outputDir, { recursive: true });
  let lastSavedPath = path.join(outputDir, "openfront-policy.json");
  let latestEpisode = 0;
  let latestSummary: unknown = {
    interrupted: false,
    message: "No completed episodes yet.",
  };

  if (restart) {
    const restarted = loadRestartPolicy(
      restartFrom,
      env.inputSize,
      learningRate,
      random,
    );
    if (restarted !== null) {
      policy = restarted.policy;
      latestEpisode = restarted.episode;
      latestSummary = restarted.summary;
      console.log(
        `restarted neural policy from ${restartFrom} at episode=${latestEpisode}`,
      );
    }
  }

  let stopRequested = false;

  const saveLatest = (reason: "checkpoint" | "complete" | "interrupted") => {
    lastSavedPath = savePolicySnapshot(
      outputDir,
      publicDir,
      latestEpisode,
      {
        ...(typeof latestSummary === "object" && latestSummary !== null
          ? latestSummary
          : { summary: latestSummary }),
        saveReason: reason,
      },
      policy,
    );
  };

  process.once("SIGINT", () => {
    stopRequested = true;
    console.log("\nCtrl+C received. Saving the latest neural policy...");
    saveLatest("interrupted");
    console.log(`saved trained policy snapshot to ${lastSavedPath}`);
    if (publicDir !== null) {
      console.log(
        `copied live-game policy to ${path.join(publicDir, "openfront-policy.json")}`,
      );
    }
    console.log("Stopping after the current episode cleanup.");
  });

  const episodeEnd =
    opponentAI === "heuristic" ? "maxTicks_or_match_finish" : "match_finish";
  console.log(
    `training openfront policy: episodes=${trainForever ? "until Ctrl+C" : episodes} map=${mapName} opponents=${opponents} opponentAI=${opponentAI} maxTicks=${maxTicks} episodeEnd=${episodeEnd} curriculum=${curriculumMode} logEvery=${logEvery} checkpointEvery=${checkpointEvery} diagnosticsEvery=${diagnosticsEvery} diagnostics=${diagnosticsFile ?? "off"} workers=${workerCount} learningRate=${learningRate} restart=${restart ? restartFrom : "off"} publicCopy=${publicDir ?? "off"}`,
  );
  console.log(`simulation map: ${env.mapDescription()}`);
  console.log(
    `network: ${[env.inputSize, ...NEURAL_BOT_HIDDEN_LAYERS, TRAINING_ACTIONS.length].join(" -> ")} (${policy.parameterCount()} parameters)`,
  );
  console.log(
    `spatial input: global=${NEURAL_BOT_GLOBAL_MAP_SIZE}x${NEURAL_BOT_GLOBAL_MAP_SIZE} local=${NEURAL_BOT_LOCAL_MAP_SIZE}x${NEURAL_BOT_LOCAL_MAP_SIZE} channels=${NEURAL_BOT_MAP_CHANNEL_COUNT} pooled=${NEURAL_BOT_SPATIAL_POOL_SIZE}x${NEURAL_BOT_SPATIAL_POOL_SIZE} scalars=${NEURAL_BOT_SCALAR_INPUT_SIZE}`,
  );
  console.log(`actions: ${TRAINING_ACTIONS.join(", ")}`);

  if (diagnosticsFile !== null) {
    appendJsonLine(diagnosticsFile, {
      type: "run_start",
      runId,
      startedAt: new Date().toISOString(),
      config: {
        checkpointEvery,
        curriculum: curriculumMode,
        decisionInterval,
        diagnosticsEvery,
        episodeEnd,
        episodes: trainForever ? "until Ctrl+C" : episodes,
        learningRate,
        logEvery,
        map: mapName,
        mapDescription: env.mapDescription(),
        maxTicks,
        opponentAI,
        opponents,
        seed,
        workers: workerCount,
      },
      network: {
        actions: TRAINING_ACTIONS,
        hiddenLayers: NEURAL_BOT_HIDDEN_LAYERS,
        inputSize: env.inputSize,
        outputSize: TRAINING_ACTIONS.length,
        parameterCount: policy.parameterCount(),
        scalarInputs: NEURAL_BOT_SCALAR_INPUT_SIZE,
      },
    });
  }

  const startedAtMs = Date.now();
  let lastLogAtMs = startedAtMs;
  let lastLoggedEpisode = latestEpisode;
  let windowEpisodes = 0;
  let windowReward = 0;
  let windowAlive = 0;
  let windowTiles = 0;
  let windowBestReward = Number.NEGATIVE_INFINITY;
  let latestEpisodeSummary: EpisodeRollout["summary"] | null = null;
  let lastCheckpointEpisode = latestEpisode;
  let diagnosticsWindow = newDiagnosticsWindow();
  let lastDiagnosticsEpisode = latestEpisode;
  let latestTrainStats: ReturnType<NeuralPolicy["train"]> | null = null;

  const epsilonForEpisode = (episode: number): number =>
    trainForever ? 0.1 : Math.max(0.04, 0.25 * (1 - episode / episodes));

  const recordRollout = (rollout: EpisodeRollout): void => {
    const { summary } = rollout;
    latestEpisode = rollout.episode;
    latestSummary = summary;
    latestEpisodeSummary = summary;
    windowEpisodes++;
    windowReward += summary.reward;
    windowAlive += summary.alive ? 1 : 0;
    windowTiles += summary.ownedTiles;
    windowBestReward = Math.max(windowBestReward, summary.reward);
  };

  const recordDiagnostics = (rollout: EpisodeRollout): void => {
    const { summary } = rollout;
    diagnosticsWindow.episodes++;
    diagnosticsWindow.reward += summary.reward;
    diagnosticsWindow.alive += summary.alive ? 1 : 0;
    diagnosticsWindow.wins += summary.won ? 1 : 0;
    diagnosticsWindow.tiles += summary.ownedTiles;
    diagnosticsWindow.ticks += summary.ticks;
    diagnosticsWindow.ranks += summary.rank;
    diagnosticsWindow.alliances += summary.alliances;
    diagnosticsWindow.aliveOpponents += summary.aliveOpponents;
    diagnosticsWindow.bestReward = Math.max(
      diagnosticsWindow.bestReward,
      summary.reward,
    );
    diagnosticsWindow.worstReward = Math.min(
      diagnosticsWindow.worstReward,
      summary.reward,
    );

    for (const transition of rollout.transitions) {
      diagnosticsWindow.transitions++;
      diagnosticsWindow.actionCounts[transition.action]++;
      diagnosticsWindow.actionSucceeded += transition.actionSucceeded ? 1 : 0;
      for (const [key, value] of Object.entries(
        transition.rewardBreakdown,
      ) as Array<[RewardBreakdownKey, number]>) {
        diagnosticsWindow.rewardBreakdown[key] += value;
      }
    }
  };

  const logProgress = (
    trainStats: ReturnType<NeuralPolicy["train"]>,
    transitions: Transition[],
  ): void => {
    if (latestEpisodeSummary === null) {
      return;
    }

    const shouldLog =
      latestEpisode === 1 ||
      latestEpisode - lastLoggedEpisode >= logEvery ||
      (!trainForever && latestEpisode >= episodes);
    if (shouldLog) {
      const nowMs = Date.now();
      const elapsedSeconds = (nowMs - startedAtMs) / 1000;
      const intervalSeconds = Math.max(0.001, (nowMs - lastLogAtMs) / 1000);
      const intervalEpisodes = latestEpisode - lastLoggedEpisode;
      console.log(
        [
          `episode=${trainForever ? String(latestEpisode) : `${latestEpisode}/${episodes}`}`,
          `reward=${latestEpisodeSummary.reward.toFixed(2)}`,
          `tiles=${latestEpisodeSummary.ownedTiles}`,
          `rank=${latestEpisodeSummary.rank}`,
          `alive=${latestEpisodeSummary.alive ? "yes" : "no"}`,
          `loss=${trainStats.loss.toFixed(4)}`,
          `grad=${trainStats.gradNorm.toFixed(2)}`,
          `update=${trainStats.skipped ? "skipped" : trainStats.clipped ? "clipped" : "ok"}`,
          `maxParam=${trainStats.maxAbsParameter.toFixed(2)}`,
          `eps=${(intervalEpisodes / intervalSeconds).toFixed(2)}/s`,
          `elapsed=${formatSeconds(elapsedSeconds)}`,
          `avgReward=${(windowReward / windowEpisodes).toFixed(2)}`,
          `aliveRate=${((windowAlive / windowEpisodes) * 100).toFixed(1)}%`,
          `avgTiles=${(windowTiles / windowEpisodes).toFixed(0)}`,
          `bestReward=${windowBestReward.toFixed(2)}`,
          `curriculum=${curriculumPhaseForEpisode(latestEpisode, curriculumMode).name}`,
          actionCounts(transitions),
        ].join(" "),
      );
      lastLogAtMs = nowMs;
      lastLoggedEpisode = latestEpisode;
      windowEpisodes = 0;
      windowReward = 0;
      windowAlive = 0;
      windowTiles = 0;
      windowBestReward = Number.NEGATIVE_INFINITY;
    }
  };

  const maybeCheckpoint = (): void => {
    if (
      latestEpisode - lastCheckpointEpisode >= checkpointEvery ||
      (!trainForever && latestEpisode >= episodes)
    ) {
      saveLatest(trainForever ? "checkpoint" : "complete");
      lastCheckpointEpisode = latestEpisode;
    }
  };

  const maybeWriteDiagnostics = (
    trainStats: ReturnType<NeuralPolicy["train"]>,
    force = false,
  ): void => {
    if (diagnosticsFile === null || diagnosticsWindow.episodes === 0) {
      return;
    }

    const reachedInterval =
      latestEpisode - lastDiagnosticsEpisode >= diagnosticsEvery;
    if (!force && !reachedInterval) {
      return;
    }

    const nowMs = Date.now();
    const elapsedSeconds = (nowMs - startedAtMs) / 1000;
    const episodes = diagnosticsWindow.episodes;
    const transitions = diagnosticsWindow.transitions;
    const rewardBreakdownAverage = averageRewardBreakdown(
      diagnosticsWindow.rewardBreakdown,
      episodes,
    );
    const actionFractions = normalizedActionCounts(
      diagnosticsWindow.actionCounts,
      transitions,
    );
    const record = {
      type: "training_diagnostics",
      runId,
      writtenAt: new Date().toISOString(),
      episode: latestEpisode,
      window: {
        fromEpisode: lastDiagnosticsEpisode + 1,
        toEpisode: latestEpisode,
        episodes,
        transitions,
        elapsedSeconds,
        episodesPerSecond:
          elapsedSeconds <= 0 ? 0 : latestEpisode / elapsedSeconds,
        avgReward: diagnosticsWindow.reward / episodes,
        bestReward: diagnosticsWindow.bestReward,
        worstReward: diagnosticsWindow.worstReward,
        aliveRate: diagnosticsWindow.alive / episodes,
        winRate: diagnosticsWindow.wins / episodes,
        avgTiles: diagnosticsWindow.tiles / episodes,
        avgTicks: diagnosticsWindow.ticks / episodes,
        avgRank: diagnosticsWindow.ranks / episodes,
        avgAlliances: diagnosticsWindow.alliances / episodes,
        avgAliveOpponents: diagnosticsWindow.aliveOpponents / episodes,
        actionSuccessRate:
          transitions === 0
            ? 0
            : diagnosticsWindow.actionSucceeded / transitions,
      },
      actions: {
        counts: namedActionCounts(diagnosticsWindow.actionCounts),
        fractions: actionFractions,
      },
      rewardBreakdown: {
        totals: diagnosticsWindow.rewardBreakdown,
        averages: rewardBreakdownAverage,
      },
      latest: latestEpisodeSummary,
      training: {
        clipped: trainStats.clipped,
        gradNorm: trainStats.gradNorm,
        loss: trainStats.loss,
        maxAbsParameter: trainStats.maxAbsParameter,
        skipped: trainStats.skipped,
      },
      curriculum: curriculumPhaseForEpisode(latestEpisode, curriculumMode).name,
      hints: diagnosticHints(diagnosticsWindow),
    };

    appendJsonLine(diagnosticsFile, record);
    diagnosticsWindow = newDiagnosticsWindow();
    lastDiagnosticsEpisode = latestEpisode;
  };

  const trainRollouts = (
    rollouts: EpisodeRollout[],
  ): ReturnType<NeuralPolicy["train"]> => {
    const samples = rollouts.flatMap((rollout) =>
      discountedSamples(rollout.transitions, 0.96),
    );
    return policy.train(samples);
  };

  const allTransitions = (rollouts: EpisodeRollout[]): Transition[] =>
    rollouts.flatMap((rollout) => rollout.transitions);

  let rolloutPool: ParallelRolloutPool | null = null;
  try {
    if (workerCount <= 1) {
      for (
        let episode = latestEpisode + 1;
        !stopRequested && (trainForever || episode <= episodes);
        episode++
      ) {
        const curriculumPhase = curriculumPhaseForEpisode(
          episode,
          curriculumMode,
        );
        const rollout = await runPolicyEpisode({
          allowedActionIndexes: actionIndexes(curriculumPhase.actions),
          env,
          episode,
          epsilon: epsilonForEpisode(episode),
          policy,
          random,
          shouldStop: () => stopRequested,
        });

        if (rollout === null || stopRequested) {
          break;
        }

        const trainStats = trainRollouts([rollout]);
        latestTrainStats = trainStats;
        recordRollout(rollout);
        recordDiagnostics(rollout);
        logProgress(trainStats, rollout.transitions);
        maybeWriteDiagnostics(trainStats);
        maybeCheckpoint();
      }
    } else {
      rolloutPool = new ParallelRolloutPool(workerCount);
      while (!stopRequested && (trainForever || latestEpisode < episodes)) {
        const remainingEpisodes = trainForever
          ? workerCount
          : Math.min(workerCount, episodes - latestEpisode);
        const snapshot = policy.toJSON();
        const jobs = Array.from({ length: remainingEpisodes }, (_, index) => {
          const episode = latestEpisode + index + 1;
          const curriculumPhase = curriculumPhaseForEpisode(
            episode,
            curriculumMode,
          );
          return {
            allowedActionIndexes: actionIndexes(curriculumPhase.actions),
            envOptions,
            episode,
            epsilon: epsilonForEpisode(episode),
            policy: snapshot,
            randomSeed: seed + episode * 1_000_003 + index * 9_973,
          };
        });
        const rollouts = (await rolloutPool.runBatch(jobs)).sort(
          (left, right) => left.episode - right.episode,
        );

        if (stopRequested) {
          break;
        }

        const trainStats = trainRollouts(rollouts);
        latestTrainStats = trainStats;
        for (const rollout of rollouts) {
          recordRollout(rollout);
          recordDiagnostics(rollout);
        }
        logProgress(trainStats, allTransitions(rollouts));
        maybeWriteDiagnostics(trainStats);
        maybeCheckpoint();
      }
    }
  } finally {
    await rolloutPool?.close();
  }

  if (latestTrainStats !== null) {
    maybeWriteDiagnostics(latestTrainStats, true);
  }

  if (!stopRequested) {
    saveLatest("complete");
    console.log(`saved trained policy snapshot to ${lastSavedPath}`);
    if (publicDir !== null) {
      console.log(
        `copied live-game policy to ${path.join(publicDir, "openfront-policy.json")}`,
      );
    }
  }

  if (diagnosticsFile !== null) {
    appendJsonLine(diagnosticsFile, {
      type: "run_end",
      runId,
      endedAt: new Date().toISOString(),
      episode: latestEpisode,
      status: stopRequested ? "interrupted" : "complete",
      summary: latestSummary,
    });
  }
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
