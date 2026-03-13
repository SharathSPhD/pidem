"use client";

import React from "react";
import { cn } from "@/lib/utils";

export interface ChapterHeaderProps {
  chapterNumber: number;
  totalChapters?: number;
  title: string;
  subtitle: string;
  estimatedMinutes: number;
  learningObjectives: string[];
}

export function ChapterHeader({
  chapterNumber,
  totalChapters = 19,
  title,
  subtitle,
  estimatedMinutes,
  learningObjectives,
}: ChapterHeaderProps) {
  return (
    <header className="w-full bg-gradient-to-b from-slate-50 to-white py-12 px-6">
      <div className="mx-auto max-w-4xl">
        <div className="mb-6 flex items-baseline gap-4">
          <span className="text-5xl font-bold text-amber-500 tabular-nums">
            {String(chapterNumber).padStart(2, "0")}
          </span>
          <span className="text-sm font-medium text-slate-500">
            / {totalChapters} chapters
          </span>
        </div>
        <h1 className="mb-2 text-3xl font-bold text-slate-900 md:text-4xl">
          {title}
        </h1>
        <p className="mb-6 text-lg text-slate-500">{subtitle}</p>
        <div className="flex flex-wrap items-center gap-3">
          <span className="rounded-full bg-white px-3 py-1 text-sm font-medium text-slate-600">
            ~{estimatedMinutes} min
          </span>
          {learningObjectives.map((objective) => (
            <span
              key={objective}
              className="rounded-full bg-white/80 px-3 py-1 text-sm text-slate-500"
            >
              {objective}
            </span>
          ))}
        </div>
      </div>
    </header>
  );
}
