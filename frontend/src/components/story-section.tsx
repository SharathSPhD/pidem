"use client";

import React from "react";
import * as Collapsible from "@radix-ui/react-collapsible";
import { ChevronDown, ChevronRight } from "lucide-react";
import { cn } from "@/lib/utils";

export type Beat =
  | "FRAME"
  | "EXPLORE"
  | "BUILD"
  | "INTERROGATE"
  | "DECIDE";

const BEAT_STYLES: Record<
  Beat,
  { border: string; badge: string; label: string }
> = {
  FRAME: {
    border: "border-l-amber-500",
    badge: "bg-amber-500/10 text-amber-500 border-amber-500/30",
    label: "Frame",
  },
  EXPLORE: {
    border: "border-l-sky-500",
    badge: "bg-sky-500/10 text-sky-500 border-sky-500/30",
    label: "Explore",
  },
  BUILD: {
    border: "border-l-emerald-500",
    badge: "bg-emerald-500/10 text-emerald-500 border-emerald-500/30",
    label: "Build",
  },
  INTERROGATE: {
    border: "border-l-rose-500",
    badge: "bg-rose-500/10 text-rose-500 border-rose-500/30",
    label: "Interrogate",
  },
  DECIDE: {
    border: "border-l-violet-500",
    badge: "bg-violet-500/10 text-violet-500 border-violet-500/30",
    label: "Decide",
  },
};

export interface StorySectionProps {
  beat: Beat;
  title: string;
  children: React.ReactNode;
  defaultOpen?: boolean;
  className?: string;
}

export function StorySection({
  beat,
  title,
  children,
  defaultOpen = true,
  className,
}: StorySectionProps) {
  const [open, setOpen] = React.useState(defaultOpen);
  const styles = BEAT_STYLES[beat];

  return (
    <Collapsible.Root open={open} onOpenChange={setOpen}>
      <div
        className={cn(
          "rounded-xl border border-slate-200 bg-slate-50",
          styles.border,
          "border-l-4",
          className
        )}
      >
        <Collapsible.Trigger className="flex w-full items-center gap-3 px-6 py-4 text-left transition-colors hover:bg-slate-50">
          {open ? (
            <ChevronDown className="h-5 w-5 shrink-0 text-slate-500" />
          ) : (
            <ChevronRight className="h-5 w-5 shrink-0 text-slate-500" />
          )}
          <span
            className={cn(
              "rounded border px-2 py-0.5 text-xs font-semibold uppercase tracking-wider",
              styles.badge
            )}
          >
            {styles.label}
          </span>
          <h2 className="flex-1 text-lg font-semibold text-slate-900">{title}</h2>
        </Collapsible.Trigger>
        <Collapsible.Content>
          <div className="border-t border-slate-200 px-6 py-5">{children}</div>
        </Collapsible.Content>
      </div>
    </Collapsible.Root>
  );
}
