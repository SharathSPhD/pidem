"use client";

import React from "react";
import { Eye, Sparkles, AlertTriangle, Brain } from "lucide-react";
import { cn } from "@/lib/utils";

export type InsightType = "notice" | "try" | "warning" | "think";

const INSIGHT_STYLES: Record<
  InsightType,
  { icon: React.ComponentType<{ className?: string }>; border: string; bg: string; iconColor: string; label: string }
> = {
  notice: {
    icon: Eye,
    border: "border-l-sky-500",
    bg: "bg-sky-500/5",
    iconColor: "text-sky-500",
    label: "Notice that...",
  },
  try: {
    icon: Sparkles,
    border: "border-l-emerald-500",
    bg: "bg-emerald-500/5",
    iconColor: "text-emerald-500",
    label: "Try this...",
  },
  warning: {
    icon: AlertTriangle,
    border: "border-l-amber-500",
    bg: "bg-amber-500/5",
    iconColor: "text-amber-500",
    label: "Watch out...",
  },
  think: {
    icon: Brain,
    border: "border-l-violet-500",
    bg: "bg-violet-500/5",
    iconColor: "text-violet-500",
    label: "Think about...",
  },
};

export interface GuidedInsightProps {
  type: InsightType;
  children: React.ReactNode;
  className?: string;
}

export function GuidedInsight({
  type,
  children,
  className,
}: GuidedInsightProps) {
  const { icon: Icon, border, bg, iconColor, label } = INSIGHT_STYLES[type];

  return (
    <div
      className={cn(
        "rounded-lg border border-slate-200",
        border,
        "border-l-4",
        bg,
        "p-4",
        className
      )}
    >
      <div className="flex gap-3">
        <Icon className={cn("mt-0.5 h-5 w-5 shrink-0", iconColor)} />
        <div className="min-w-0 flex-1">
          <p className="mb-1 text-sm font-semibold text-slate-700">{label}</p>
          <div className="text-sm text-slate-500 [&>p]:mb-2 [&>p:last-child]:mb-0">
            {children}
          </div>
        </div>
      </div>
    </div>
  );
}
