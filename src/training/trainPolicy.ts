import fs from "node:fs";
import path from "node:path";

import {
  NEURAL_BOT_GLOBAL_MAP_SIZE,
  NEURAL_BOT_LOCAL_MAP_SIZE,
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
import { OpenFrontTrainingEnv, TRAINING_ACTIONS } from "./OpenFrontTrainingEnv";
import { ParallelRolloutPool } from "./ParallelRolloutPool";
import { EpisodeRollout, runPolicyEpisode, Transition } from "./Rollout";
import { SeededRandom } from "./TrainingRandom";

const DEFAULT_LEARNING_RATE = 0.001;

interface SavedTrainingSnapshot {
  actions?: unknown;
  episode?: unknown;
  summary?: unknown;
  policy?: SerializedNeuralPolicy;
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
  const maxTicks = numberArg("--maxTicks", 700);
  const decisionInterval = numberArg("--decisionInterval", 20);
  const opponents = numberArg("--opponents", 3);
  const seed = numberArg("--seed", 42);
  const mapName = stringArg("--map", "plains");
  const outputDir = stringArg("--outDir", "training-output");
  const publicDir = hasFlag("--noPublicCopy")
    ? null
    : stringArg("--publicDir", path.join("resources", "training"));
  const logEvery = positiveIntegerArg("--logEvery", 1000);
  const checkpointEvery = positiveIntegerArg("--checkpointEvery", 1000);
  const workerCount = positiveIntegerArg("--workers", 1);
  const learningRate = numberArg("--learningRate", DEFAULT_LEARNING_RATE);
  const restart = hasFlag("--restart");
  const restartFrom = stringArg(
    "--restartFrom",
    path.join(outputDir, "openfront-policy.json"),
  );

  const random = new SeededRandom(seed);
  const envOptions = {
    decisionInterval,
    maxTicks,
    mapName,
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

  console.log(
    `training openfront policy: episodes=${trainForever ? "until Ctrl+C" : episodes} map=${mapName} opponents=${opponents} maxTicks=${maxTicks} logEvery=${logEvery} checkpointEvery=${checkpointEvery} workers=${workerCount} learningRate=${learningRate} restart=${restart ? restartFrom : "off"} publicCopy=${publicDir ?? "off"}`,
  );
  console.log(
    `network: ${[env.inputSize, ...NEURAL_BOT_HIDDEN_LAYERS, TRAINING_ACTIONS.length].join(" -> ")} (${policy.parameterCount()} parameters)`,
  );
  console.log(
    `spatial input: global=${NEURAL_BOT_GLOBAL_MAP_SIZE}x${NEURAL_BOT_GLOBAL_MAP_SIZE} local=${NEURAL_BOT_LOCAL_MAP_SIZE}x${NEURAL_BOT_LOCAL_MAP_SIZE} channels=${NEURAL_BOT_MAP_CHANNEL_COUNT} pooled=${NEURAL_BOT_SPATIAL_POOL_SIZE}x${NEURAL_BOT_SPATIAL_POOL_SIZE} scalars=${NEURAL_BOT_SCALAR_INPUT_SIZE}`,
  );
  console.log(`actions: ${TRAINING_ACTIONS.join(", ")}`);

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
        const rollout = await runPolicyEpisode({
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
        recordRollout(rollout);
        logProgress(trainStats, rollout.transitions);
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
          return {
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
        for (const rollout of rollouts) {
          recordRollout(rollout);
        }
        logProgress(trainStats, allTransitions(rollouts));
        maybeCheckpoint();
      }
    }
  } finally {
    await rolloutPool?.close();
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
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
