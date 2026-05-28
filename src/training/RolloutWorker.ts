import process from "node:process";

import { NEURAL_BOT_HIDDEN_LAYERS } from "../core/ai/NeuralBotPolicy";
import { NeuralPolicy, SerializedNeuralPolicy } from "./NeuralPolicy";
import {
  OpenFrontTrainingEnv,
  TRAINING_ACTIONS,
  TrainingEnvOptions,
} from "./OpenFrontTrainingEnv";
import { EpisodeRollout, runPolicyEpisode } from "./Rollout";
import { SeededRandom } from "./TrainingRandom";

interface RolloutRequest {
  type: "rollout";
  id: number;
  episode: number;
  envOptions: TrainingEnvOptions;
  epsilon: number;
  policy: SerializedNeuralPolicy;
  randomSeed: number;
}

interface ShutdownRequest {
  type: "shutdown";
}

type WorkerRequest = RolloutRequest | ShutdownRequest;

interface RolloutResponse {
  type: "rolloutResult";
  id: number;
  rollout: EpisodeRollout;
}

interface RolloutErrorResponse {
  type: "rolloutError";
  id: number;
  error: string;
}

let env: OpenFrontTrainingEnv | null = null;
let envKey = "";

function optionsKey(options: TrainingEnvOptions): string {
  return JSON.stringify(options);
}

async function getEnv(
  options: TrainingEnvOptions,
): Promise<OpenFrontTrainingEnv> {
  const key = optionsKey(options);
  if (env === null || envKey !== key) {
    env = new OpenFrontTrainingEnv(options);
    await env.init();
    envKey = key;
  }
  return env;
}

function send(message: RolloutResponse | RolloutErrorResponse): void {
  process.send?.(message);
}

async function handleRollout(message: RolloutRequest): Promise<void> {
  try {
    const rolloutEnv = await getEnv(message.envOptions);
    const policy = NeuralPolicy.fromJSON(
      message.policy,
      {
        inputSize: rolloutEnv.inputSize,
        hiddenLayers: NEURAL_BOT_HIDDEN_LAYERS,
        outputSize: TRAINING_ACTIONS.length,
        learningRate: message.policy.learningRate ?? 0.001,
      },
      () => 0.5,
    );

    if (policy === null) {
      throw new Error("Worker received an incompatible neural policy snapshot");
    }

    const rollout = await runPolicyEpisode({
      env: rolloutEnv,
      episode: message.episode,
      epsilon: message.epsilon,
      policy,
      random: new SeededRandom(message.randomSeed),
    });

    if (rollout === null) {
      throw new Error("Worker rollout stopped before the episode finished");
    }

    send({
      type: "rolloutResult",
      id: message.id,
      rollout,
    });
  } catch (error) {
    send({
      type: "rolloutError",
      id: message.id,
      error:
        error instanceof Error ? (error.stack ?? error.message) : String(error),
    });
  }
}

process.on("message", (message: WorkerRequest) => {
  if (message.type === "shutdown") {
    process.exit(0);
  }
  if (message.type === "rollout") {
    void handleRollout(message);
  }
});
