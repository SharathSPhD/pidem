"use client";

import React from "react";
import { ChapterHeader } from "./chapter-header";
import { ChapterNav } from "./chapter-nav";
import { cn } from "@/lib/utils";

const MODULE_LIST = [
  { slug: "m00-foundations", title: "Bias-Variance Tradeoff" },
  { slug: "m01-regression", title: "Price Elasticity" },
  { slug: "m02-classification", title: "Threat Detection" },
  { slug: "m03-ensembles", title: "Ensemble Intelligence" },
  { slug: "m04-clustering", title: "Station Segmentation" },
  { slug: "m05-anomaly", title: "Anomaly Detection" },
  { slug: "m06-timeseries", title: "Classical Forecasting" },
  { slug: "m07-sequence", title: "Sequence Models" },
  { slug: "m08-tft", title: "Temporal Fusion Transformer" },
  { slug: "m09-optimization", title: "Price Optimization" },
  { slug: "m10-bandits", title: "Price Experimentation" },
  { slug: "m11-qlearning", title: "Dynamic Pricing" },
  { slug: "m12-dqn", title: "Deep Reinforcement Learning" },
  { slug: "m13-neural", title: "Neural Networks" },
  { slug: "m14-ft-transformer", title: "FT-Transformer" },
  { slug: "m15-transformer-zoo", title: "Transformer Architectures" },
  { slug: "m16-llm", title: "LLM Capabilities" },
  { slug: "m17-rag", title: "RAG Pipeline" },
  { slug: "m18-synthesis", title: "System Design" },
];

export interface ModuleLayoutProps {
  chapterNumber: number;
  title: string;
  subtitle: string;
  estimatedMinutes: number;
  learningObjectives: string[];
  currentSlug?: string;
  children: React.ReactNode;
  className?: string;
}

export function ModuleLayout({
  chapterNumber,
  title,
  subtitle,
  estimatedMinutes,
  learningObjectives,
  currentSlug,
  children,
  className,
}: ModuleLayoutProps) {
  const slug =
    currentSlug ?? MODULE_LIST[chapterNumber]?.slug ?? MODULE_LIST[0].slug;

  return (
    <div className={cn("min-h-screen bg-white text-slate-900", className)}>
      <ChapterHeader
        chapterNumber={chapterNumber}
        totalChapters={MODULE_LIST.length}
        title={title}
        subtitle={subtitle}
        estimatedMinutes={estimatedMinutes}
        learningObjectives={learningObjectives}
      />
      <main className="mx-auto max-w-4xl space-y-6 px-6 py-8 pb-24">
        {children}
      </main>
      <ChapterNav currentSlug={slug} moduleList={MODULE_LIST} />
    </div>
  );
}
