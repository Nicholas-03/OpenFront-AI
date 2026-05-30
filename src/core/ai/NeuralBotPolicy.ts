import { NEURAL_BOT_INPUT_SIZE } from "./NeuralBotFeatures";

export const NEURAL_BOT_CLIENT_ID = "NEURALAI";
export const NEURAL_BOT_NAME = "Neural AI";
export const NEURAL_BOT_POLICY_URL = "/training/openfront-policy.json";
export const NEURAL_BOT_HIDDEN_LAYERS = [256, 256, 128] as const;

export const NEURAL_BOT_ACTIONS = [
  "expand",
  "attack_nearest",
  "attack_weakest",
  "attack_strongest",
  "build_city",
  "build_factory",
  "build_port",
  "build_warship",
  "build_defense_post",
  "build_sam_launcher",
  "build_missile_silo",
  "upgrade_structure",
  "accept_alliance",
  "request_alliance_nearest",
  "request_alliance_strongest",
  "hold",
] as const;

export type NeuralBotAction = (typeof NEURAL_BOT_ACTIONS)[number];

const MAX_NEURAL_POLICY_PARAMETER = 1_000;

export interface NeuralPolicyWeights {
  inputSize: number;
  hiddenLayers?: number[];
  hiddenSize?: number;
  outputSize: number;
  weights?: number[][][];
  biases?: number[][];
  weightsInputHidden?: number[][];
  biasHidden?: number[];
  weightsHiddenOutput?: number[][];
  biasOutput?: number[];
}

export interface NormalizedNeuralPolicyWeights {
  inputSize: number;
  hiddenLayers: number[];
  outputSize: number;
  weights: number[][][];
  biases: number[][];
}

export interface NeuralPolicySnapshot {
  actions: readonly string[];
  policy: NeuralPolicyWeights;
}

export interface LoadedNeuralPolicy {
  actions: readonly NeuralBotAction[];
  weights: NormalizedNeuralPolicyWeights;
}

export interface NeuralPolicyForwardPass {
  logits: number[];
  probs: number[];
}

function isNeuralAction(value: string): value is NeuralBotAction {
  return NEURAL_BOT_ACTIONS.includes(value as NeuralBotAction);
}

function softmax(logits: number[]): number[] {
  const max = Math.max(...logits);
  const exps = logits.map((value) => Math.exp(value - max));
  const sum = exps.reduce((acc, value) => acc + value, 0);
  return exps.map((value) => value / sum);
}

function hasShape(matrix: number[][], rows: number, columns: number): boolean {
  return (
    Array.isArray(matrix) &&
    matrix.length === rows &&
    matrix.every(
      (row) =>
        Array.isArray(row) &&
        row.length === columns &&
        row.every(
          (value) =>
            Number.isFinite(value) &&
            Math.abs(value) <= MAX_NEURAL_POLICY_PARAMETER,
        ),
    )
  );
}

function sameNumbers(
  left: readonly number[],
  right: readonly number[],
): boolean {
  return (
    left.length === right.length &&
    left.every((value, index) => value === right[index])
  );
}

function normalizePolicyWeights(
  policy: NeuralPolicyWeights,
): NormalizedNeuralPolicyWeights | null {
  if (
    Array.isArray(policy.hiddenLayers) &&
    Array.isArray(policy.weights) &&
    Array.isArray(policy.biases)
  ) {
    return {
      inputSize: policy.inputSize,
      hiddenLayers: policy.hiddenLayers,
      outputSize: policy.outputSize,
      weights: policy.weights,
      biases: policy.biases,
    };
  }

  if (
    policy.hiddenSize !== undefined &&
    policy.weightsInputHidden &&
    policy.biasHidden &&
    policy.weightsHiddenOutput &&
    policy.biasOutput
  ) {
    return {
      inputSize: policy.inputSize,
      hiddenLayers: [policy.hiddenSize],
      outputSize: policy.outputSize,
      weights: [policy.weightsInputHidden, policy.weightsHiddenOutput],
      biases: [policy.biasHidden, policy.biasOutput],
    };
  }

  return null;
}

function isValidPolicyShape(weights: NormalizedNeuralPolicyWeights): boolean {
  if (
    weights.inputSize <= 0 ||
    weights.outputSize <= 0 ||
    weights.hiddenLayers.some((layer) => layer <= 0)
  ) {
    return false;
  }

  const layerSizes = [
    weights.inputSize,
    ...weights.hiddenLayers,
    weights.outputSize,
  ];
  if (
    weights.weights.length !== layerSizes.length - 1 ||
    weights.biases.length !== layerSizes.length - 1
  ) {
    return false;
  }

  for (let layer = 0; layer < weights.weights.length; layer++) {
    const rows = layerSizes[layer + 1];
    const columns = layerSizes[layer];
    if (
      !hasShape(weights.weights[layer], rows, columns) ||
      weights.biases[layer].length !== rows ||
      weights.biases[layer].some(
        (value) =>
          !Number.isFinite(value) ||
          Math.abs(value) > MAX_NEURAL_POLICY_PARAMETER,
      )
    ) {
      return false;
    }
  }

  return true;
}

export function isNeuralBotClientID(
  clientID: string | null | undefined,
): boolean {
  return clientID === NEURAL_BOT_CLIENT_ID;
}

export function parseNeuralPolicySnapshot(
  snapshot: NeuralPolicySnapshot,
): LoadedNeuralPolicy | null {
  if (!snapshot?.policy || !Array.isArray(snapshot.actions)) {
    return null;
  }
  if (
    snapshot.actions.length !== NEURAL_BOT_ACTIONS.length ||
    !snapshot.actions.every(
      (action, index) =>
        isNeuralAction(action) && action === NEURAL_BOT_ACTIONS[index],
    )
  ) {
    return null;
  }

  const weights = normalizePolicyWeights(snapshot.policy);
  if (
    !weights ||
    weights.inputSize !== NEURAL_BOT_INPUT_SIZE ||
    !sameNumbers(weights.hiddenLayers, NEURAL_BOT_HIDDEN_LAYERS) ||
    weights.outputSize !== snapshot.actions.length ||
    !isValidPolicyShape(weights)
  ) {
    return null;
  }

  return {
    actions: snapshot.actions,
    weights,
  };
}

export async function loadNeuralBotPolicy(
  url = NEURAL_BOT_POLICY_URL,
): Promise<LoadedNeuralPolicy | null> {
  if (typeof fetch !== "function") {
    return null;
  }

  try {
    const response = await fetch(url, { cache: "no-store" });
    if (!response.ok) {
      console.warn(
        `[Neural AI] No trained policy found at ${url}; using fallback behavior.`,
      );
      return null;
    }

    const snapshot = (await response.json()) as NeuralPolicySnapshot;
    const policy = parseNeuralPolicySnapshot(snapshot);
    if (!policy) {
      console.warn(`[Neural AI] Policy at ${url} is invalid.`);
      return null;
    }

    console.log(`[Neural AI] Loaded trained policy from ${url}`);
    return policy;
  } catch (error) {
    console.warn(`[Neural AI] Failed to load policy from ${url}`, error);
    return null;
  }
}

export function forwardNeuralPolicy(
  policy: LoadedNeuralPolicy,
  input: number[],
): NeuralPolicyForwardPass {
  const { weights } = policy;
  if (input.length !== weights.inputSize) {
    throw new Error(
      `Expected ${weights.inputSize} neural inputs, received ${input.length}`,
    );
  }

  let current = input;
  for (let layer = 0; layer < weights.weights.length; layer++) {
    const isOutputLayer = layer === weights.weights.length - 1;
    const next = new Array(weights.weights[layer].length);
    for (let output = 0; output < weights.weights[layer].length; output++) {
      let sum = weights.biases[layer][output];
      for (let i = 0; i < current.length; i++) {
        sum += weights.weights[layer][output][i] * current[i];
      }
      next[output] = isOutputLayer ? sum : Math.max(0, sum);
    }
    current = next;
  }

  return {
    logits: current,
    probs: softmax(current),
  };
}
