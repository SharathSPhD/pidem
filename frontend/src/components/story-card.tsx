"use client";

import Link from "next/link";
import { type LucideIcon } from "lucide-react";
import { cn } from "@/lib/utils";

export type ProgressStatus =
  | "not-started"
  | "visited"
  | "trained"
  | "assessed"
  | "passed";

interface StoryCardProps {
  title: string;
  /** Scenario text (for module pages) or use description for hub cards */
  scenario?: string;
  description?: string;
  icon?: LucideIcon;
  /** Module number (e.g. M00) for hub cards */
  number?: string;
  /** Link URL - when provided, card becomes a link */
  href?: string;
  progress?: ProgressStatus;
  className?: string;
}

const progressColors: Record<ProgressStatus, string> = {
  "not-started": "bg-slate-300",
  visited: "bg-amber-500",
  trained: "bg-blue-500",
  assessed: "bg-violet-500",
  passed: "bg-emerald-500",
};

export function StoryCard({
  title,
  scenario,
  description,
  icon: Icon,
  number,
  href,
  progress,
  className,
}: StoryCardProps) {
  const content = scenario ?? description ?? "";
  const paragraphs = content ? content.split("\n").filter(Boolean) : [];

  const cardContent = (
    <div
      className={cn(
        "rounded-xl border border-slate-300/50 bg-slate-50/80 p-6 text-slate-900 shadow-lg",
        href && "transition-colors hover:border-slate-300 hover:bg-slate-50",
        className
      )}
    >
      <div className="flex items-start gap-4">
        {Icon && (
          <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-white text-amber-500">
            <Icon className="h-5 w-5" />
          </div>
        )}
        {number && !Icon && (
          <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-white text-amber-500 text-sm font-semibold">
            {number}
          </div>
        )}
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <h2 className="text-lg font-semibold text-slate-900">{title}</h2>
            {progress && (
              <span
                className={cn(
                  "inline-block h-2 w-2 rounded-full shrink-0",
                  progressColors[progress]
                )}
                title={progress}
              />
            )}
          </div>
          {(paragraphs.length > 0 || description) && (
            <div className="mt-3 space-y-2 text-slate-700 leading-relaxed">
              {paragraphs.length > 0
                ? paragraphs.map((p, i) => <p key={i}>{p}</p>)
                : content && <p>{content}</p>}
            </div>
          )}
        </div>
      </div>
    </div>
  );

  if (href) {
    return <Link href={href}>{cardContent}</Link>;
  }
  return cardContent;
}
