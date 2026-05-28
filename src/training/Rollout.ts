import { NeuralPolicy, PolicyGradientSample } from "./NeuralPolicy";
import {
  EpisodeSummary,
  OpenFrontTrainingEnv,
  RewardBreakdown,
  TRAINING_ACTIONS,
} from "./OpenFrontTrainingEnv";
import { SeededRandom } from "./TrainingRandom";

export interface Transition extends PolicyGradientSample {
  actionSucceeded: boolean;
  reward: number;
  rewardBreakdown: RewardBreakdown;
}

export interface EpisodeRollout {
  episode: number;
  summary: EpisodeSummary;
  transitions: Transition[];
}

interface RunPolicyEpisodeOptions {
  allowedActionIndexes?: readonly number[];
  env: OpenFrontTrainingEnv;
  episode: number;
  epsilon: number;
  policy: NeuralPolicy;
  random: SeededRandom;
  shouldStop?: () => boolean;
}

function sampleAction(probs: number[], random: SeededRandom): number {
  let threshold = random.next();
  for (let index = 0; index < probs.length; index++) {
    threshold -= probs[index];
    if (threshold <= 0) {
      return index;
    }
  }
  return probs.length - 1;
}

function maskProbabilities(probs: number[], mask: boolean[]): number[] {
  const masked = probs.map((probability, index) =>
    mask[index] ? probability : 0,
  );
  const total = masked.reduce((acc, value) => acc + value, 0);
  if (Number.isFinite(total) && total > 0) {
    return masked.map((probability) => probability / total);
  }

  const validCount = mask.filter(Boolean).length;
  if (validCount === 0) {
    return probs.map(() => 1 / probs.length);
  }
  return mask.map((valid) => (valid ? 1 / validCount : 0));
}

function actionMask(
  env: OpenFrontTrainingEnv,
  allowedActionIndexes?: readonly number[],
): boolean[] {
  const allowed =
    allowedActionIndexes === undefined
      ? null
      : new Set<number>(allowedActionIndexes);
  const mask = env
    .validActionMask()
    .map((valid, index) => valid && (allowed === null || allowed.has(index)));
  if (mask.some(Boolean)) {
    return mask;
  }

  const holdIndex = TRAINING_ACTIONS.indexOf("hold");
  if (holdIndex !== -1) {
    mask[holdIndex] = true;
  }
  return mask;
}

function randomMaskedAction(mask: boolean[], random: SeededRandom): number {
  const validActions = mask
    .map((valid, index) => (valid ? index : -1))
    .filter((index) => index !== -1);
  return validActions[random.int(validActions.length)];
}

export async function runPolicyEpisode({
  allowedActionIndexes,
  env,
  episode,
  epsilon,
  policy,
  random,
  shouldStop = () => false,
}: RunPolicyEpisodeOptions): Promise<EpisodeRollout | null> {
  let state = await env.reset(episode);
  const transitions: Transition[] = [];
  let done = false;

  while (!done && !shouldStop()) {
    const pass = policy.forward(state);
    const mask = actionMask(env, allowedActionIndexes);
    const probs = maskProbabilities(pass.probs, mask);
    const action =
      random.next() < epsilon
        ? randomMaskedAction(mask, random)
        : sampleAction(probs, random);
    const result = env.step(action);

    transitions.push({
      input: pass.input,
      hidden: pass.hidden,
      activations: pass.activations,
      probs,
      action,
      actionSucceeded: result.actionSucceeded,
      advantage: 0,
      reward: result.reward,
      rewardBreakdown: result.rewardBreakdown,
    });

    state = env.state();
    done = result.done;
  }

  if (shouldStop()) {
    return null;
  }

  return {
    episode,
    summary: env.summary(),
    transitions,
  };
}
