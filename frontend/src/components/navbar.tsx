"use client";

import Link from "next/link";
import { useState } from "react";
import { Menu, X, ChevronDown } from "lucide-react";
import { useStore, type ProgressStatus } from "@/lib/store";
import { cn } from "@/lib/utils";

const MODULE_GROUPS = [
  {
    label: "I · Foundation",
    modules: [{ slug: "m00-foundations", label: "M00", title: "Bias-Variance" }],
  },
  {
    label: "II · Supervised",
    modules: [
      { slug: "m01-regression", label: "M01", title: "Regression" },
      { slug: "m02-classification", label: "M02", title: "Classification" },
      { slug: "m03-ensembles", label: "M03", title: "Ensembles" },
    ],
  },
  {
    label: "III · Unsupervised",
    modules: [
      { slug: "m04-clustering", label: "M04", title: "Clustering" },
      { slug: "m05-anomaly", label: "M05", title: "Anomaly" },
    ],
  },
  {
    label: "IV · Time Series",
    modules: [
      { slug: "m06-timeseries", label: "M06", title: "ARIMA" },
      { slug: "m07-sequence", label: "M07", title: "Sequence" },
      { slug: "m08-tft", label: "M08", title: "TFT" },
    ],
  },
  {
    label: "V · Optimization & RL",
    modules: [
      { slug: "m09-optimization", label: "M09", title: "LP/NLP" },
      { slug: "m10-bandits", label: "M10", title: "Bandits" },
      { slug: "m11-qlearning", label: "M11", title: "Q-Learn" },
      { slug: "m12-dqn", label: "M12", title: "DQN" },
    ],
  },
  {
    label: "VI · Neural",
    modules: [
      { slug: "m13-neural", label: "M13", title: "MLP" },
      { slug: "m14-ft-transformer", label: "M14", title: "FT-T" },
      { slug: "m15-transformer-zoo", label: "M15", title: "Zoo" },
    ],
  },
  {
    label: "VII · LLM & RAG",
    modules: [
      { slug: "m16-llm", label: "M16", title: "LLM" },
      { slug: "m17-rag", label: "M17", title: "RAG" },
    ],
  },
  {
    label: "VIII · Synthesis",
    modules: [{ slug: "m18-synthesis", label: "M18", title: "Capstone" }],
  },
];

function ProgressDot({ status }: { status: ProgressStatus }) {
  const colors: Record<ProgressStatus, string> = {
    not_started: "bg-slate-400",
    visited: "bg-amber-500",
    trained: "bg-blue-500",
    assessed: "bg-violet-500",
    passed: "bg-emerald-500",
  };
  return (
    <span
      className={cn("inline-block h-2 w-2 rounded-full shrink-0", colors[status])}
      title={status.replace("_", " ")}
      aria-label={status.replace("_", " ")}
    />
  );
}

function NavGroup({
  group,
  getProgress,
}: {
  group: (typeof MODULE_GROUPS)[number];
  getProgress: (slug: string) => ProgressStatus;
}) {
  const [open, setOpen] = useState(false);
  return (
    <div className="relative" onMouseEnter={() => setOpen(true)} onMouseLeave={() => setOpen(false)}>
      <button
        type="button"
        className="flex items-center gap-1 rounded-md px-2 py-1.5 text-xs font-medium text-slate-500 hover:bg-slate-100 hover:text-slate-900 transition-colors"
        onClick={() => setOpen((o) => !o)}
      >
        {group.label}
        <ChevronDown className={cn("h-3 w-3 transition-transform", open && "rotate-180")} />
      </button>
      {open && (
        <div className="absolute top-full left-0 z-50 mt-1 min-w-[180px] rounded-lg border border-slate-300 bg-slate-50 p-1 shadow-xl">
          {group.modules.map((mod) => (
            <Link
              key={mod.slug}
              href={`/modules/${mod.slug}`}
              className="flex items-center gap-2 rounded-md px-3 py-2 text-sm text-slate-700 hover:bg-slate-100 hover:text-slate-900 transition-colors"
              onClick={() => setOpen(false)}
            >
              <ProgressDot status={getProgress(mod.slug)} />
              <span className="font-mono text-xs text-slate-500">{mod.label}</span>
              <span>{mod.title}</span>
            </Link>
          ))}
        </div>
      )}
    </div>
  );
}

export function Navbar() {
  const [mobileOpen, setMobileOpen] = useState(false);
  const getProgress = useStore((s) => s.getProgress);

  return (
    <nav className="fixed top-0 z-50 w-full border-b border-slate-200 bg-white/95 shadow-sm border-b border-slate-200 backdrop-blur supports-[backdrop-filter]:bg-white/80">
      <div className="mx-auto flex h-14 max-w-7xl items-center justify-between px-4 sm:px-6 lg:px-8">
        <Link
          href="/"
          className="text-base font-semibold text-slate-900 hover:text-slate-800 transition-colors tracking-tight"
        >
          Pricing Intelligence Lab
        </Link>

        <div className="hidden lg:flex items-center gap-0.5">
          {MODULE_GROUPS.map((group) => (
            <NavGroup key={group.label} group={group} getProgress={getProgress} />
          ))}
        </div>

        <button
          type="button"
          onClick={() => setMobileOpen((o) => !o)}
          className="lg:hidden rounded-md p-2 text-slate-500 hover:bg-slate-100 hover:text-slate-900"
          aria-label={mobileOpen ? "Close menu" : "Open menu"}
        >
          {mobileOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
        </button>
      </div>

      {mobileOpen && (
        <div className="lg:hidden border-t border-slate-200 bg-white shadow-sm border-b border-slate-200 px-4 py-3 max-h-[70vh] overflow-y-auto">
          {MODULE_GROUPS.map((group) => (
            <div key={group.label} className="mb-3">
              <p className="mb-1 text-xs font-medium uppercase tracking-wider text-slate-500">{group.label}</p>
              <div className="flex flex-wrap gap-1.5">
                {group.modules.map((mod) => (
                  <Link
                    key={mod.slug}
                    href={`/modules/${mod.slug}`}
                    onClick={() => setMobileOpen(false)}
                    className="flex items-center gap-2 rounded-lg bg-slate-50 px-3 py-2 text-sm text-slate-700 hover:bg-slate-100 hover:text-slate-900"
                  >
                    <ProgressDot status={getProgress(mod.slug)} />
                    <span className="font-mono text-xs text-slate-500">{mod.label}</span>
                    <span>{mod.title}</span>
                  </Link>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
    </nav>
  );
}
