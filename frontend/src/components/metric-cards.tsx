"use client";

import { cn } from "@/lib/utils";

export interface MetricItem {
  label: string;
  value: string | number;
  delta?: string;
  status?: "neutral" | "positive" | "negative";
}

interface MetricCardsProps {
  metrics: MetricItem[];
  className?: string;
}

export function MetricCards({ metrics, className }: MetricCardsProps) {
  const statusColors = {
    neutral: "text-slate-500",
    positive: "text-emerald-400",
    negative: "text-rose-400",
  };

  return (
    <div
      className={cn(
        "grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4",
        className
      )}
    >
      {metrics.map((metric) => (
        <div
          key={metric.label}
          className="rounded-xl border border-slate-200 bg-slate-50 p-4 shadow-sm"
        >
          <p className="text-sm font-medium text-slate-500">{metric.label}</p>
          <p className="mt-1 text-2xl font-semibold text-slate-900">
            {metric.value}
          </p>
          {metric.delta !== undefined && (
            <p
              className={cn(
                "mt-1 text-sm",
                statusColors[metric.status ?? "neutral"]
              )}
            >
              {metric.delta}
            </p>
          )}
        </div>
      ))}
    </div>
  );
}
