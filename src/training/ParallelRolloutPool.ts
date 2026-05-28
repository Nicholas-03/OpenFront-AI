import { ChildProcess, fork } from "node:child_process";
import { fileURLToPath } from "node:url";

import { SerializedNeuralPolicy } from "./NeuralPolicy";
import { TrainingEnvOptions } from "./OpenFrontTrainingEnv";
import { EpisodeRollout } from "./Rollout";

interface RolloutJob {
  episode: number;
  envOptions: TrainingEnvOptions;
  epsilon: number;
  policy: SerializedNeuralPolicy;
  randomSeed: number;
}

interface RolloutRequest extends RolloutJob {
  type: "rollout";
  id: number;
}

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

type WorkerResponse = RolloutResponse | RolloutErrorResponse;

interface PendingRollout {
  id: number;
  reject: (error: Error) => void;
  resolve: (rollout: EpisodeRollout) => void;
}

interface WorkerSlot {
  child: ChildProcess;
  index: number;
  pending: PendingRollout | null;
}

export class ParallelRolloutPool {
  private readonly slots: WorkerSlot[];
  private nextJobId = 1;

  constructor(workerCount: number) {
    const workerPath = fileURLToPath(
      new URL("./RolloutWorker.ts", import.meta.url),
    );
    this.slots = Array.from({ length: workerCount }, (_, index) => {
      const child = fork(workerPath, [], {
        env: process.env,
        execArgv: ["--import", "tsx"],
        stdio: ["ignore", "inherit", "inherit", "ipc"],
      });
      const slot: WorkerSlot = {
        child,
        index,
        pending: null,
      };

      child.on("message", (message) => {
        this.handleMessage(slot, message as WorkerResponse);
      });
      child.on("exit", (code, signal) => {
        if (slot.pending !== null) {
          slot.pending.reject(
            new Error(
              `Rollout worker ${slot.index + 1} exited before finishing job ${slot.pending.id} (code=${code ?? "null"} signal=${signal ?? "null"})`,
            ),
          );
          slot.pending = null;
        }
      });

      return slot;
    });
  }

  async runBatch(jobs: RolloutJob[]): Promise<EpisodeRollout[]> {
    if (jobs.length > this.slots.length) {
      throw new Error(
        `Cannot run ${jobs.length} rollout jobs on ${this.slots.length} workers in one batch`,
      );
    }

    return Promise.all(
      jobs.map((job, index) => this.runOnSlot(this.slots[index], job)),
    );
  }

  async close(): Promise<void> {
    await Promise.all(
      this.slots.map(
        (slot) =>
          new Promise<void>((resolve) => {
            const child = slot.child;
            if (child.exitCode !== null || child.killed) {
              resolve();
              return;
            }

            const timeout = setTimeout(() => {
              child.kill();
              resolve();
            }, 1_000);

            child.once("exit", () => {
              clearTimeout(timeout);
              resolve();
            });

            child.send?.({ type: "shutdown" }, (error) => {
              if (error !== null) {
                child.kill();
              }
            });
          }),
      ),
    );
  }

  private runOnSlot(
    slot: WorkerSlot,
    job: RolloutJob,
  ): Promise<EpisodeRollout> {
    if (slot.pending !== null) {
      return Promise.reject(
        new Error(`Rollout worker ${slot.index + 1} already has a job`),
      );
    }

    const id = this.nextJobId++;
    const request: RolloutRequest = {
      ...job,
      id,
      type: "rollout",
    };

    return new Promise((resolve, reject) => {
      slot.pending = {
        id,
        reject,
        resolve,
      };

      slot.child.send?.(request, (error) => {
        if (error !== null) {
          slot.pending = null;
          reject(error);
        }
      });
    });
  }

  private handleMessage(slot: WorkerSlot, message: WorkerResponse): void {
    const pending = slot.pending;
    if (pending === null || pending.id !== message.id) {
      return;
    }

    slot.pending = null;
    if (message.type === "rolloutError") {
      pending.reject(new Error(message.error));
      return;
    }

    pending.resolve(message.rollout);
  }
}
