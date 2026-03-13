"use client";

import { useState } from "react";
import * as Collapsible from "@radix-ui/react-collapsible";
import { ChevronDown, ChevronUp } from "lucide-react";
import { cn } from "@/lib/utils";

export interface AssessmentQuestion {
  id: string;
  type?: "conceptual" | "applied" | "critical";
  question: string;
  options?: string[];
  modelAnswer?: string;
}

interface AssessmentProps {
  questions: AssessmentQuestion[];
  onSubmit?: (answers: Record<string, string>) => void;
  className?: string;
}

export function Assessment({ questions, onSubmit, className }: AssessmentProps) {
  const [open, setOpen] = useState(false);
  const [answers, setAnswers] = useState<Record<string, string>>({});
  const [submitted, setSubmitted] = useState(false);
  const [revealed, setRevealed] = useState<Record<string, boolean>>({});

  const handleAnswerChange = (id: string, value: string) => {
    setAnswers((prev) => ({ ...prev, [id]: value }));
  };

  const handleSubmit = () => {
    setSubmitted(true);
    onSubmit?.(answers);
  };

  const handleReveal = (id: string) => {
    setRevealed((prev) => ({ ...prev, [id]: true }));
  };

  return (
    <Collapsible.Root
      open={open}
      onOpenChange={setOpen}
      className={cn(
        "rounded-xl border border-slate-200 bg-slate-50",
        className
      )}
    >
      <Collapsible.Trigger asChild>
        <button
          type="button"
          className="flex w-full items-center justify-between gap-4 p-4 text-left text-slate-900 hover:bg-slate-50 transition-colors rounded-t-xl"
        >
          <span className="font-semibold">Module Assessment</span>
          {open ? (
            <ChevronUp className="h-5 w-5 shrink-0 text-slate-500" />
          ) : (
            <ChevronDown className="h-5 w-5 shrink-0 text-slate-500" />
          )}
        </button>
      </Collapsible.Trigger>
      <Collapsible.Content>
        <div className="border-t border-slate-200 p-6 space-y-6">
          {questions.map((q, idx) => (
            <div key={q.id} className="space-y-3">
              <div className="flex items-baseline gap-2">
                <span className="text-xs font-medium uppercase tracking-wider text-amber-600">
                  {q.type === "applied" ? "Applied" : q.type === "critical" ? "Critical Thinking" : "Conceptual"}
                </span>
                <span className="text-xs text-slate-500">Q{idx + 1}</span>
              </div>
              <p className="text-slate-500">{q.question}</p>
              {q.options ? (
                <div className="space-y-2">
                  {q.options.map((opt) => (
                    <label
                      key={opt}
                      className="flex items-center gap-2 rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 cursor-pointer hover:bg-slate-100"
                    >
                      <input
                        type="radio"
                        name={q.id}
                        value={opt}
                        checked={answers[q.id] === opt}
                        onChange={(e) => handleAnswerChange(q.id, e.target.value)}
                        className="h-4 w-4 rounded border-slate-300 text-amber-500 focus:ring-amber-500"
                      />
                      <span className="text-slate-600">{opt}</span>
                    </label>
                  ))}
                </div>
              ) : (
                <textarea
                  value={answers[q.id] ?? ""}
                  onChange={(e) => handleAnswerChange(q.id, e.target.value)}
                  placeholder="Your answer..."
                  rows={3}
                  className="w-full rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-slate-600 placeholder-slate-400 focus:border-amber-500 focus:outline-none focus:ring-1 focus:ring-amber-500"
                />
              )}
              {submitted && q.modelAnswer && (
                <div className="mt-2">
                  <button
                    type="button"
                    onClick={() => handleReveal(q.id)}
                    className="text-sm text-amber-600 hover:text-amber-300"
                  >
                    {revealed[q.id] ? "Model answer:" : "Reveal model answer"}
                  </button>
                  {revealed[q.id] && (
                    <div className="mt-2 rounded-lg border border-slate-200 bg-slate-50 p-3 text-slate-600 text-sm">
                      {q.modelAnswer}
                    </div>
                  )}
                </div>
              )}
            </div>
          ))}
          <button
            type="button"
            onClick={handleSubmit}
            disabled={submitted}
            className={cn(
              "rounded-lg px-4 py-2 text-sm font-medium transition-colors",
              submitted
                ? "cursor-not-allowed bg-slate-100 text-slate-500"
                : "bg-amber-500 text-slate-900 hover:bg-amber-400"
            )}
          >
            {submitted ? "Submitted" : "Submit"}
          </button>
        </div>
      </Collapsible.Content>
    </Collapsible.Root>
  );
}
