import { NeuralPolicy, PolicyGradientSample } from "./NeuralPolicy";
import {
  EpisodeSummary,
  OpenFrontTrainingEnv,
  TRAINING_ACTIONS,
} from "./OpenFrontTrainingEnv";
import { SeededRandom } from "./TrainingRandom";

export interface Transition extends PolicyGradientSample {
  reward: number;
}

export interface EpisodeRollout {
  episode: number;
  summary: EpisodeSummary;
  transitions: Transition[];
}

interface RunPolicyEpisodeOptions {
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

export async function runPolicyEpisode({
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
    const action =
      random.next() < epsilon
        ? random.int(TRAINING_ACTIONS.length)
        : sampleAction(pass.probs, random);
    const result = env.step(action);

    transitions.push({
      input: pass.input,
      hidden: pass.hidden,
      activations: pass.activations,
      probs: pass.probs,
      action,
      advantage: 0,
      reward: result.reward,
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
