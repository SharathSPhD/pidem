"use client";

import React from "react";
import Link from "next/link";
import { ChevronLeft, ChevronRight } from "lucide-react";

export interface ChapterNavProps {
  currentSlug: string;
  moduleList: { slug: string; title: string }[];
}

export function ChapterNav({ currentSlug, moduleList }: ChapterNavProps) {
  const currentIndex = moduleList.findIndex((m) => m.slug === currentSlug);
  const prevModule =
    currentIndex > 0 ? moduleList[currentIndex - 1] : null;
  const nextModule =
    currentIndex >= 0 && currentIndex < moduleList.length - 1
      ? moduleList[currentIndex + 1]
      : null;
  const progress =
    moduleList.length > 0
      ? ((currentIndex + 1) / moduleList.length) * 100
      : 0;

  return (
    <nav className="fixed bottom-0 left-0 right-0 z-40 h-16 border-t border-slate-200 bg-white">
      <div
        className="absolute left-0 top-0 h-0.5 bg-amber-500/80 transition-all duration-300"
        style={{ width: `${progress}%` }}
      />
      <div className="mx-auto flex h-full max-w-4xl items-center justify-between px-6">
        {prevModule ? (
          <Link
            href={`/modules/${prevModule.slug}`}
            className="flex items-center gap-2 text-sm font-medium text-slate-500 transition-colors hover:text-slate-900"
          >
            <ChevronLeft className="h-4 w-4" />
            <span className="hidden sm:inline">Previous:</span>
            <span className="max-w-[140px] truncate sm:max-w-[200px]">
              {prevModule.title}
            </span>
          </Link>
        ) : (
          <div />
        )}
        <span className="text-xs text-slate-500">
          {currentIndex + 1} / {moduleList.length}
        </span>
        {nextModule ? (
          <Link
            href={`/modules/${nextModule.slug}`}
            className="flex items-center gap-2 text-sm font-medium text-slate-500 transition-colors hover:text-slate-900"
          >
            <span className="max-w-[140px] truncate sm:max-w-[200px]">
              {nextModule.title}
            </span>
            <span className="hidden sm:inline">Next:</span>
            <ChevronRight className="h-4 w-4" />
          </Link>
        ) : (
          <div />
        )}
      </div>
    </nav>
  );
}
