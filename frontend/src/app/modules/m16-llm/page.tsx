"use client";

import { useState, useCallback, useEffect } from "react";
import { ModuleLayout } from "@/components/module-layout";
import { StorySection } from "@/components/story-section";
import { GuidedInsight } from "@/components/guided-insight";
import { ChartPanel } from "@/components/chart-panel";
import { Assessment, type AssessmentQuestion } from "@/components/assessment";
import { TrainingStatus } from "@/components/training-status";
import { apiPost } from "@/lib/api-client";
import { useTrainingRunner } from "@/hooks/use-training-runner";
import type { PlotlyFigure } from "@/components/chart-panel";

interface M16AnalyzeResponse {
  figures?: Record<string, PlotlyFigure>;
  data?: {
    tokens?: Array<{ token: string; id: number; position: number }>;
    n_tokens?: number;
    llm_analysis?: string;
    capabilities?: string[];
  };
}

const DEFAULT_TEXT = "The diesel price at motorway stations is 1.65 EUR/L";

const ASSESSMENT_QUESTIONS: AssessmentQuestion[] = [
  {
    id: "q1",
    type: "conceptual",
    question: "How does temperature affect output diversity?",
    options: [
      "Higher temperature = more deterministic",
      "Higher temperature = more diverse/creative",
      "Temperature has no effect",
      "Temperature only affects speed",
    ],
    modelAnswer:
      "Higher temperature increases randomness in token sampling. At 0, outputs are deterministic; at 1.5, responses become creative but may hallucinate.",
  },
  {
    id: "q2",
    type: "applied",
    question: "What does tokenization reveal about model capabilities?",
    options: [
      "Domain terms like 'EUR/litre' may split into multiple tokens",
      "Tokenization is always 1:1 with words",
      "Tokenization is irrelevant to understanding",
      "All models use the same tokenizer",
    ],
    modelAnswer:
      "Domain-specific terms often split into subword tokens. 'EUR/litre' may become multiple tokens, affecting how well the model handles pricing terminology.",
  },
  {
    id: "q3",
    type: "critical",
    question: "Your manager asks the LLM 'What should Station 42's price be?' The LLM gives a confident answer. Should you trust it? Why or why not?",
    modelAnswer:
      "No. LLMs don't know your actual prices. Without RAG (M17), they generate plausible but potentially wrong numbers. The model has no access to your data — it's pattern-matching from training, not reasoning over your facts.",
  },
];

export default function M16LlmPage() {
  const [input, setInput] = useState(DEFAULT_TEXT);
  const [temperature, setTemperature] = useState(0.7);
  const [tokenFigure, setTokenFigure] = useState<PlotlyFigure | null>(null);
  const [llmAnalysis, setLlmAnalysis] = useState<string | null>(null);
  const [tokens, setTokens] = useState<Array<{ token: string; id: number; position: number }>>([]);

  const { status, error, run } = useTrainingRunner<M16AnalyzeResponse>();
  const llmOffline = (llmAnalysis ?? "").toLowerCase().includes("offline");

  const analyze = useCallback(async () => {
    const res = await run(() =>
      apiPost<M16AnalyzeResponse>("/api/m16/analyze", {
        text: input,
        temperature,
      })
    );
    if (res?.figures?.primary) setTokenFigure(res.figures.primary as PlotlyFigure);
    if (res?.data?.llm_analysis != null) setLlmAnalysis(res.data.llm_analysis);
    if (res?.data?.tokens) setTokens(res.data.tokens);
  }, [input, temperature, run]);

  useEffect(() => {
    let cancelled = false;
    apiPost<M16AnalyzeResponse>("/api/m16/analyze", {
      text: DEFAULT_TEXT,
      temperature: 0.7,
    })
      .then((res) => {
        if (!cancelled && res?.figures?.primary) setTokenFigure(res.figures.primary as PlotlyFigure);
        if (!cancelled && res?.data?.llm_analysis != null) setLlmAnalysis(res.data!.llm_analysis!);
        if (!cancelled && res?.data?.tokens) setTokens(res.data!.tokens!);
      })
      .catch(() => {});
    return () => { cancelled = true; };
  }, []);

  return (
    <ModuleLayout
      chapterNumber={16}
      title="LLM Capabilities"
      subtitle="What can a 9-billion-parameter model understand about pricing?"
      estimatedMinutes={20}
      learningObjectives={[
        "Tokenization",
        "Temperature control",
        "Prompt engineering basics",
      ]}
      currentSlug="m16-llm"
    >
      <StorySection beat="FRAME" title="Asking the right questions">
        <div className="space-y-4 text-slate-500">
          <p>
            You&apos;ve built models that predict, classify, and optimize. Now you have a 9B-parameter language model (Nemotron) that can reason about pricing in natural language. The question isn&apos;t whether it&apos;s smart — it&apos;s whether you can ask the right questions.
          </p>
          <p>
            LLMs are transforming how pricing teams work. Instead of building SQL queries to analyze competitor moves, you ask a question in English. Instead of writing PowerPoint summaries of pricing performance, the LLM drafts them. The technology isn&apos;t replacing pricing expertise — it&apos;s amplifying it.
          </p>
          <p>
            Three concepts matter. <strong className="text-slate-700">Tokenization</strong> is how the model sees text: it breaks words into subword units (tokens). Domain terms like &quot;EUR/litre&quot; may split into multiple tokens, which affects how well the model handles pricing terminology. <strong className="text-slate-700">Temperature</strong> controls the creativity vs consistency trade-off: 0 gives deterministic, report-like answers; higher values (e.g., 1.0–1.5) produce creative but risky output — the model might invent prices that don&apos;t exist. <strong className="text-slate-700">Prompt engineering</strong> is the art of asking the right question: a vague prompt yields vague answers; a well-structured prompt with context gets actionable insights.
          </p>
        </div>
      </StorySection>

      <StorySection beat="EXPLORE" title="Tokenization and analysis">
        <div className="space-y-4">
          {llmOffline && (
            <GuidedInsight type="notice">
              LLM features are running in graceful offline mode on this machine. Core modules still work, and you can enable live LLM by starting the NIM service in a supported environment.
            </GuidedInsight>
          )}
          <ChartPanel
            figure={tokenFigure}
            title="Token IDs"
            loading={!tokenFigure && status !== "error"}
          />
          {llmAnalysis && (
            <div className="rounded-xl border border-slate-200 bg-slate-50 p-4">
              <h3 className="mb-2 text-sm font-semibold uppercase tracking-wider text-slate-500">
                LLM Analysis
              </h3>
              <p className="whitespace-pre-wrap text-sm text-slate-600">{llmAnalysis}</p>
            </div>
          )}
          <GuidedInsight type="notice">
            The model breaks &quot;EUR/litre&quot; into multiple tokens. Tokenization affects how well the model handles domain-specific terminology.
          </GuidedInsight>
        </div>
      </StorySection>

      <StorySection beat="BUILD" title="Pricing query and temperature">
        <div className="space-y-4">
          <div className="flex flex-wrap gap-6">
            <label className="flex flex-col gap-1">
              <span className="text-xs text-slate-500">Pricing query</span>
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Enter pricing text..."
                className="w-80 rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm text-slate-600 placeholder-slate-400"
              />
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-xs text-slate-500">Temperature (0.0–1.5)</span>
              <input
                type="range"
                min={0}
                max={1.5}
                step={0.1}
                value={temperature}
                onChange={(e) => setTemperature(Number(e.target.value))}
                className="w-40"
              />
              <span className="text-xs text-slate-500">{temperature.toFixed(1)}</span>
            </label>
            <div className="flex flex-col justify-end gap-1">
              <button
                onClick={analyze}
                disabled={status === "running"}
                className="rounded-lg bg-amber-500 px-4 py-2 text-sm font-medium text-slate-900 hover:bg-amber-400 disabled:opacity-50"
              >
                Analyze
              </button>
              <TrainingStatus status={status} error={error} />
            </div>
          </div>
          <ChartPanel
            figure={tokenFigure}
            title="Token visualization"
            loading={status === "running"}
          />
          {tokens.length > 0 && (
            <div className="flex flex-wrap gap-2">
              {tokens.map((t, i) => (
                <span
                  key={i}
                  className="rounded bg-slate-100 px-2 py-1 text-xs text-slate-600"
                >
                  {t.token} ({t.id})
                </span>
              ))}
            </div>
          )}
          {llmAnalysis && (
            <div className="rounded-xl border border-slate-200 bg-slate-50 p-4">
              <h3 className="mb-2 text-sm font-semibold uppercase tracking-wider text-slate-500">
                LLM Analysis
              </h3>
              <p className="whitespace-pre-wrap text-sm text-slate-600">{llmAnalysis}</p>
            </div>
          )}
        </div>
      </StorySection>

      <StorySection beat="INTERROGATE" title="Temperature and hallucination">
        <div className="space-y-4">
          <p className="text-slate-500">
            LLMs excel at pattern-matching and language fluency, but they have clear limits. They don&apos;t have access to your live data unless you provide it (e.g., via RAG). They can confidently state false facts — a phenomenon called hallucination. In pricing, that means invented competitor prices, fabricated volume figures, or plausible-sounding but wrong recommendations. The same technology that powers a retailer&apos;s customer-service chatbot can mislead a pricing analyst if used without guardrails.
          </p>
          <GuidedInsight type="try">
            Set temperature to 0. The response is deterministic but may be repetitive.
          </GuidedInsight>
          <GuidedInsight type="try">
            Set temperature to 1.5. The response becomes creative but may hallucinate pricing data.
          </GuidedInsight>
          <GuidedInsight type="warning">
            LLMs don&apos;t know your actual prices. Without RAG (M17), they generate plausible but potentially wrong numbers.
          </GuidedInsight>
          <GuidedInsight type="think">
            Hallucination risk is highest when the model is asked for specific numbers it cannot verify — e.g., &quot;What is Competitor X&apos;s current price?&quot; or &quot;What should we charge tomorrow?&quot; Always validate numeric outputs against known data before they influence decisions.
          </GuidedInsight>
        </div>
      </StorySection>

      <StorySection beat="DECIDE" title="Should you trust the LLM?">
        <div className="space-y-4">
          <p className="text-slate-500">
            Practical deployment requires deciding what to delegate to LLMs vs keep manual. <strong className="text-slate-700">Delegate</strong>: drafting reports, summarizing long documents, suggesting alternative phrasings, answering conceptual questions about pricing theory, and generating first-pass analyses for human review. <strong className="text-slate-700">Keep manual</strong>: final price decisions, numeric outputs that drive automated systems, compliance-critical statements, and any output that will be used without human verification. The LLM is a powerful assistant, not a replacement for domain judgment.
          </p>
          <p className="text-slate-500">
            Your manager asks the LLM &quot;What should Station 42&apos;s price be?&quot; The LLM gives a confident answer. Should you trust it? Why or why not?
          </p>
          <Assessment questions={ASSESSMENT_QUESTIONS} />
        </div>
      </StorySection>
    </ModuleLayout>
  );
}
