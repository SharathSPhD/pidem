"use client";

import { useEffect, useState, useCallback } from "react";
import { apiGet } from "@/lib/api-client";

export interface TrainingStatus {
  status: "pending" | "running" | "completed" | "failed";
  progress?: number;
  metrics?: Record<string, number>;
  results?: unknown;
  error?: string;
}

const POLL_INTERVAL_MS = 2000;

export function useAsyncTraining(runId: string | undefined) {
  const [status, setStatus] = useState<TrainingStatus | null>(null);
  const [isComplete, setIsComplete] = useState(false);

  const fetchStatus = useCallback(async () => {
    if (!runId) return null;
    try {
      const data = await apiGet<TrainingStatus>(
        `/api/v1/runs/${runId}/status`
      );
      return data;
    } catch (err) {
      return {
        status: "failed" as const,
        error: err instanceof Error ? err.message : "Unknown error",
      };
    }
  }, [runId]);

  useEffect(() => {
    if (!runId) {
      setStatus(null);
      setIsComplete(false);
      return;
    }

    let cancelled = false;

    const poll = async () => {
      const data = await fetchStatus();
      if (cancelled || !data) return;

      setStatus(data);

      if (data.status === "completed" || data.status === "failed") {
        setIsComplete(true);
        return;
      }
    };

    poll();
    const interval = setInterval(poll, POLL_INTERVAL_MS);

    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [runId, fetchStatus]);

  return {
    status,
    progress: status?.progress ?? 0,
    metrics: status?.metrics,
    results: status?.results,
    isComplete: isComplete && (status?.status === "completed" || status?.status === "failed"),
  };
}
