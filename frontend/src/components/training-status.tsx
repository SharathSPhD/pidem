"use client";

import { useEffect, useRef } from "react";
import * as Progress from "@radix-ui/react-progress";
import { useAsyncTraining } from "@/hooks/use-async-training";
import { cn } from "@/lib/utils";

interface TrainingStatusProps {
  runId?: string | undefined;
  onComplete?: () => void;
  /** When using useTrainingRunner (sync training), pass status and error directly */
  status?: "idle" | "running" | "done" | "error";
  error?: string;
  className?: string;
}

export function TrainingStatus({
  runId,
  onComplete,
  status: statusProp,
  error: errorProp,
  className,
}: TrainingStatusProps) {
  const polled = useAsyncTraining(runId);
  const onCompleteFired = useRef(false);

  const isPolling = runId !== undefined && runId !== null;
  const status = isPolling ? polled.status : null;
  const progress = isPolling ? polled.progress : 0;
  const isComplete = isPolling ? polled.isComplete : false;

  useEffect(() => {
    if (isPolling && isComplete && status?.status === "completed" && onComplete && !onCompleteFired.current) {
      onCompleteFired.current = true;
      onComplete();
    }
  }, [isPolling, isComplete, status?.status, onComplete]);

  if (!isPolling && !statusProp) return null;

  const statusLabel = isPolling
    ? (status?.status === "pending"
        ? "Queued"
        : status?.status === "running"
          ? "Training..."
          : status?.status === "completed"
            ? "Complete"
            : status?.status === "failed"
              ? `Failed: ${status.error ?? "Unknown error"}`
              : "Checking...")
    : (statusProp === "running"
        ? "Training..."
        : statusProp === "done"
          ? "Complete"
          : statusProp === "error"
            ? `Failed: ${errorProp ?? "Unknown error"}`
            : "Idle");
  const progressValue = isPolling
    ? (status?.status === "completed" ? 100 : (progress ?? 0) * 100)
    : statusProp === "running"
      ? 50
      : statusProp === "done"
        ? 100
        : 0;
  const isFailed = isPolling ? status?.status === "failed" : statusProp === "error";

  return (
    <div className={cn("space-y-2", className)}>
      <div className="flex items-center justify-between text-sm">
        <span className="text-slate-500">{statusLabel}</span>
        {(isPolling ? status?.status === "running" : statusProp === "running") && (
          <span className="text-amber-600">{Math.round(progressValue)}%</span>
        )}
      </div>
      <Progress.Root
        value={progressValue}
        className="relative h-2 w-full overflow-hidden rounded-full bg-white"
      >
        <Progress.Indicator
          className={cn(
            "h-full w-full transition-all duration-500 ease-out",
            isFailed ? "bg-rose-500" : "bg-amber-500"
          )}
          style={{
            transform: `translateX(-${100 - progressValue}%)`,
          }}
        />
      </Progress.Root>
    </div>
  );
}
