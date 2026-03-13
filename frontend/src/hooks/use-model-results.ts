"use client";

import { useEffect, useState, useCallback } from "react";
import { apiGet } from "@/lib/api-client";

export function useModelResults<T = unknown>(runId: string | undefined) {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  const fetchResults = useCallback(async () => {
    if (!runId) {
      setData(null);
      setLoading(false);
      setError(null);
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const result = await apiGet<T>(`/api/v1/runs/${runId}/results`);
      setData(result);
    } catch (err) {
      setError(err instanceof Error ? err : new Error(String(err)));
      setData(null);
    } finally {
      setLoading(false);
    }
  }, [runId]);

  useEffect(() => {
    fetchResults();
  }, [fetchResults]);

  return { data, loading, error, refetch: fetchResults };
}
