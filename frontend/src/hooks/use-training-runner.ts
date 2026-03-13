"use client";

import { useState, useCallback } from "react";

export interface TrainingState {
  status: "idle" | "running" | "done" | "error";
  error?: string;
}

export function useTrainingRunner<T>() {
  const [state, setState] = useState<TrainingState>({ status: "idle" });

  const run = useCallback(async (trainFn: () => Promise<T>): Promise<T | undefined> => {
    setState({ status: "running" });
    try {
      const result = await trainFn();
      setState({ status: "done" });
      return result;
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Training failed";
      setState({ status: "error", error: msg });
      throw e;
    }
  }, []);

  return { ...state, run };
}
