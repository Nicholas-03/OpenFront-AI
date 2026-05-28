export interface ForwardPass {
  input: number[];
  hidden: number[];
  activations: number[][];
  logits: number[];
  probs: number[];
}

export interface PolicyGradientSample {
  input: number[];
  hidden?: number[];
  activations?: number[][];
  probs: number[];
  action: number;
  advantage: number;
}

export interface TrainStats {
  loss: number;
  gradNorm: number;
}

export interface SerializedNeuralPolicy {
  inputSize: number;
  hiddenLayers?: number[];
  hiddenSize?: number;
  outputSize: number;
  learningRate?: number;
  weights?: number[][][];
  biases?: number[][];
  weightsInputHidden?: number[][];
  biasHidden?: number[];
  weightsHiddenOutput?: number[][];
  biasOutput?: number[];
  parameterCount?: number;
}

export interface NeuralPolicyArchitecture {
  inputSize: number;
  hiddenLayers: number | readonly number[];
  outputSize: number;
  learningRate: number;
}

type RandomFn = () => number;

function zeros(rows: number, columns: number): number[][] {
  return Array.from({ length: rows }, () => new Array(columns).fill(0));
}

function softmax(logits: number[]): number[] {
  const max = Math.max(...logits);
  const exps = logits.map((value) => Math.exp(value - max));
  const sum = exps.reduce((acc, value) => acc + value, 0);
  return exps.map((value) => value / sum);
}

function relu(value: number): number {
  return Math.max(0, value);
}

function hiddenLayersFrom(value: number | readonly number[]): number[] {
  return typeof value === "number" ? [value] : [...value];
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

function hasNumericVector(value: unknown, length: number): value is number[] {
  return (
    Array.isArray(value) &&
    value.length === length &&
    value.every((entry) => Number.isFinite(entry))
  );
}

function hasNumericMatrix(
  value: unknown,
  rows: number,
  columns: number,
): value is number[][] {
  return (
    Array.isArray(value) &&
    value.length === rows &&
    value.every((row) => hasNumericVector(row, columns))
  );
}

function normalizeSerializedPolicy(policy: SerializedNeuralPolicy): {
  inputSize: number;
  hiddenLayers: number[];
  outputSize: number;
  weights: number[][][];
  biases: number[][];
} | null {
  const hiddenLayers = Array.isArray(policy.hiddenLayers)
    ? [...policy.hiddenLayers]
    : policy.hiddenSize !== undefined
      ? [policy.hiddenSize]
      : null;

  if (
    hiddenLayers === null ||
    !Number.isInteger(policy.inputSize) ||
    !Number.isInteger(policy.outputSize) ||
    hiddenLayers.some((layer) => !Number.isInteger(layer))
  ) {
    return null;
  }

  if (Array.isArray(policy.weights) && Array.isArray(policy.biases)) {
    return {
      inputSize: policy.inputSize,
      hiddenLayers,
      outputSize: policy.outputSize,
      weights: policy.weights,
      biases: policy.biases,
    };
  }

  if (
    policy.weightsInputHidden !== undefined &&
    policy.biasHidden !== undefined &&
    policy.weightsHiddenOutput !== undefined &&
    policy.biasOutput !== undefined
  ) {
    return {
      inputSize: policy.inputSize,
      hiddenLayers,
      outputSize: policy.outputSize,
      weights: [policy.weightsInputHidden, policy.weightsHiddenOutput],
      biases: [policy.biasHidden, policy.biasOutput],
    };
  }

  return null;
}

function hasExpectedShape(
  weights: number[][][],
  biases: number[][],
  layerSizes: readonly number[],
): boolean {
  if (
    weights.length !== layerSizes.length - 1 ||
    biases.length !== layerSizes.length - 1
  ) {
    return false;
  }

  for (let layer = 0; layer < weights.length; layer++) {
    const rows = layerSizes[layer + 1];
    const columns = layerSizes[layer];
    if (
      !hasNumericMatrix(weights[layer], rows, columns) ||
      !hasNumericVector(biases[layer], rows)
    ) {
      return false;
    }
  }

  return true;
}

export class NeuralPolicy {
  public readonly hiddenLayers: number[];
  public readonly hiddenSize: number;

  private readonly layerSizes: number[];
  private weights: number[][][];
  private biases: number[][];

  constructor(
    public readonly inputSize: number,
    hiddenLayers: number | number[],
    public readonly outputSize: number,
    private readonly learningRate: number,
    random: RandomFn,
  ) {
    this.hiddenLayers = Array.isArray(hiddenLayers)
      ? [...hiddenLayers]
      : [hiddenLayers];
    this.hiddenSize = this.hiddenLayers[0] ?? 0;
    this.layerSizes = [inputSize, ...this.hiddenLayers, outputSize];
    this.weights = [];
    this.biases = [];

    for (let layer = 1; layer < this.layerSizes.length; layer++) {
      const inputs = this.layerSizes[layer - 1];
      const outputs = this.layerSizes[layer];
      const scale = Math.sqrt(2 / inputs);
      this.weights.push(
        Array.from({ length: outputs }, () =>
          Array.from({ length: inputs }, () => (random() * 2 - 1) * scale),
        ),
      );
      this.biases.push(new Array(outputs).fill(0));
    }
  }

  static fromJSON(
    serialized: SerializedNeuralPolicy,
    architecture: NeuralPolicyArchitecture,
    random: RandomFn,
  ): NeuralPolicy | null {
    const expectedHiddenLayers = hiddenLayersFrom(architecture.hiddenLayers);
    const expectedLayerSizes = [
      architecture.inputSize,
      ...expectedHiddenLayers,
      architecture.outputSize,
    ];
    const normalized = normalizeSerializedPolicy(serialized);
    if (
      normalized === null ||
      normalized.inputSize !== architecture.inputSize ||
      normalized.outputSize !== architecture.outputSize ||
      !sameNumbers(normalized.hiddenLayers, expectedHiddenLayers) ||
      !hasExpectedShape(
        normalized.weights,
        normalized.biases,
        expectedLayerSizes,
      )
    ) {
      return null;
    }

    const policy = new NeuralPolicy(
      architecture.inputSize,
      expectedHiddenLayers,
      architecture.outputSize,
      architecture.learningRate,
      random,
    );
    policy.weights = normalized.weights.map((layer) =>
      layer.map((row) => [...row]),
    );
    policy.biases = normalized.biases.map((bias) => [...bias]);
    return policy;
  }

  parameterCount(): number {
    let count = 0;
    for (let layer = 0; layer < this.weights.length; layer++) {
      count += this.biases[layer].length;
      for (const row of this.weights[layer]) {
        count += row.length;
      }
    }
    return count;
  }

  forward(input: number[]): ForwardPass {
    if (input.length !== this.inputSize) {
      throw new Error(
        `Expected ${this.inputSize} inputs, received ${input.length}`,
      );
    }

    const hiddenActivations: number[][] = [];
    let current = input;

    for (let layer = 0; layer < this.weights.length; layer++) {
      const isOutputLayer = layer === this.weights.length - 1;
      const next = new Array(this.weights[layer].length);
      for (let output = 0; output < this.weights[layer].length; output++) {
        let sum = this.biases[layer][output];
        for (let i = 0; i < current.length; i++) {
          sum += this.weights[layer][output][i] * current[i];
        }
        next[output] = isOutputLayer ? sum : relu(sum);
      }

      if (!isOutputLayer) {
        hiddenActivations.push(next);
      }
      current = next;
    }

    return {
      input,
      hidden: hiddenActivations[hiddenActivations.length - 1] ?? [],
      activations: hiddenActivations,
      logits: current,
      probs: softmax(current),
    };
  }

  train(samples: PolicyGradientSample[]): TrainStats {
    if (samples.length === 0) {
      return { loss: 0, gradNorm: 0 };
    }

    const gradWeights = this.weights.map((layer) =>
      zeros(layer.length, layer[0]?.length ?? 0),
    );
    const gradBiases = this.biases.map((bias) =>
      new Array(bias.length).fill(0),
    );
    let loss = 0;

    for (const sample of samples) {
      const clippedAdvantage = Math.max(-3, Math.min(3, sample.advantage));
      loss +=
        -Math.log(Math.max(sample.probs[sample.action], 1e-9)) *
        clippedAdvantage;

      const activations = [
        sample.input,
        ...(sample.activations ?? (sample.hidden ? [sample.hidden] : [])),
      ];
      if (activations.length !== this.weights.length) {
        throw new Error(
          `Expected ${this.weights.length - 1} hidden activation layers, received ${activations.length - 1}`,
        );
      }

      let delta = sample.probs.slice();
      delta[sample.action] -= 1;
      for (let i = 0; i < delta.length; i++) {
        delta[i] *= clippedAdvantage;
      }

      for (let layer = this.weights.length - 1; layer >= 0; layer--) {
        const layerInput = activations[layer];
        const nextDelta =
          layer > 0 ? new Array(this.layerSizes[layer]).fill(0) : [];

        for (let output = 0; output < this.weights[layer].length; output++) {
          gradBiases[layer][output] += delta[output];
          for (let input = 0; input < layerInput.length; input++) {
            gradWeights[layer][output][input] +=
              delta[output] * layerInput[input];
            if (layer > 0) {
              nextDelta[input] +=
                delta[output] * this.weights[layer][output][input];
            }
          }
        }

        if (layer > 0) {
          const previousActivation = activations[layer];
          for (let input = 0; input < nextDelta.length; input++) {
            if (previousActivation[input] <= 0) {
              nextDelta[input] = 0;
            }
          }
          delta = nextDelta;
        }
      }
    }

    const scale = this.learningRate / samples.length;
    let gradNormSquared = 0;

    for (let layer = 0; layer < this.weights.length; layer++) {
      for (let output = 0; output < this.weights[layer].length; output++) {
        gradNormSquared +=
          gradBiases[layer][output] * gradBiases[layer][output];
        this.biases[layer][output] -= gradBiases[layer][output] * scale;
        for (
          let input = 0;
          input < this.weights[layer][output].length;
          input++
        ) {
          gradNormSquared +=
            gradWeights[layer][output][input] *
            gradWeights[layer][output][input];
          this.weights[layer][output][input] -=
            gradWeights[layer][output][input] * scale;
        }
      }
    }

    return {
      loss: loss / samples.length,
      gradNorm: Math.sqrt(gradNormSquared),
    };
  }

  toJSON() {
    return {
      inputSize: this.inputSize,
      hiddenLayers: this.hiddenLayers,
      hiddenSize: this.hiddenSize,
      outputSize: this.outputSize,
      learningRate: this.learningRate,
      weights: this.weights,
      biases: this.biases,
      parameterCount: this.parameterCount(),
    };
  }
}
