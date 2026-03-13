"use client";

import { useState, useEffect } from "react";
import { ModuleLayout } from "@/components/module-layout";
import { StorySection } from "@/components/story-section";
import { GuidedInsight } from "@/components/guided-insight";
import { Assessment, type AssessmentQuestion } from "@/components/assessment";
import { apiGet } from "@/lib/api-client";

interface OverviewData {
  architecture?: {
    layers?: string[];
    components_per_layer?: number[];
  };
  governance_checklist?: { item: string; status: string }[];
  modules_summary?: {
    total?: number;
    categories?: Record<string, number>;
  };
}

const LAYERS = [
  { name: "Data Layer", desc: "Ingestion, validation, synthetic generation, caching", color: "border-sky-500" },
  { name: "ML Layer", desc: "14 models: regression, classification, clustering, time series, optimization, RL, neural", color: "border-emerald-500" },
  { name: "API Layer", desc: "FastAPI routers, Pydantic validation, async task management", color: "border-amber-500" },
  { name: "Frontend Layer", desc: "Next.js, Plotly.js, Zustand state, 19 interactive chapters", color: "border-violet-500" },
  { name: "LLM Layer", desc: "Nemotron NIM, RAG pipeline, LoRA fine-tuning, prompt engineering", color: "border-rose-500" },
];

const ASSESSMENT: AssessmentQuestion[] = [
  {
    id: "q1",
    type: "critical",
    question: "You have budget to deploy 5 of the 18 models. Which 5 would you choose for maximum business impact? Justify your selection.",
    modelAnswer:
      "A strong portfolio: (1) Regression for elasticity measurement, (2) Time Series for demand forecasting, (3) Optimization for automated pricing, (4) Anomaly Detection for competitor monitoring, (5) RAG for analyst Q&A. These cover the core pricing workflow: measure → forecast → optimize → monitor → explain.",
  },
  {
    id: "q2",
    type: "applied",
    question: "What happens when the regression model (M01) disagrees with the optimization model (M09)?",
    modelAnswer:
      "The optimization model uses elasticity from regression as an input. If they disagree, it usually means the optimization is using stale elasticity estimates. The resolution is to re-run regression with fresh data and feed updated coefficients into the optimizer. A production system needs automated pipeline orchestration to prevent this.",
  },
  {
    id: "q3",
    type: "conceptual",
    question: "Why is model monitoring listed as 'partial' in the governance checklist? What risks does this create?",
    modelAnswer:
      "Without comprehensive monitoring, you can't detect model drift — when the relationship between inputs and outputs changes over time. In pricing, this could mean your elasticity model becomes stale after a market shock, leading to systematically wrong price recommendations.",
  },
];

export default function M18SynthesisPage() {
  const [overview, setOverview] = useState<OverviewData | null>(null);
  const [activeLayer, setActiveLayer] = useState<number | null>(null);

  useEffect(() => {
    apiGet<OverviewData>("/api/m18/overview").then(setOverview).catch(() => {});
  }, []);

  const checklist = overview?.governance_checklist ?? [];
  const categories = overview?.modules_summary?.categories ?? {};

  return (
    <ModuleLayout
      chapterNumber={18}
      title="System Design"
      subtitle="How do you ship 18 models into a system a VP would approve?"
      estimatedMinutes={30}
      learningObjectives={["Production ML architecture", "Governance checklist", "Deployment strategy"]}
    >
      <StorySection beat="FRAME" title="From notebooks to production">
        <div className="space-y-3 text-slate-600">
          <p>
            Over 17 chapters, you've built models that predict demand, detect threats, forecast
            volumes, optimize prices, learn strategies, and answer questions in natural language.
            Each works in isolation in a Jupyter notebook or API endpoint.
          </p>
          <p>
            Your VP asks: <em className="text-slate-900">"Give me one system that handles pricing
            decisions end-to-end — from data ingestion to price recommendation to explainability."</em>
          </p>
          <p>
            This chapter is about the architecture that ties everything together. It&apos;s not about
            building another model — it&apos;s about building the <strong className="text-amber-600">system</strong> that
            deploys, monitors, and governs all of them.
          </p>
          <p>
            Whether you&apos;re building a pricing system for a fuel retailer, a category management platform for a grocery chain, or a procurement optimizer for an industrial distributor, the architectural decisions are the same: how do models talk to each other? How do you monitor drift? How do you maintain stakeholder trust?
          </p>
        </div>
      </StorySection>

      <StorySection beat="EXPLORE" title="The five-layer architecture">
        <div className="space-y-4">
          <p className="text-slate-500">
            System architecture isn&apos;t just boxes and arrows — it determines where failures propagate, how fast you can iterate, and whether the VP will sign off. A well-designed layer boundary isolates risk: a bug in the ML layer doesn&apos;t crash the frontend; a data pipeline delay doesn&apos;t corrupt model outputs. The &quot;so what?&quot; is that your deployment strategy, monitoring plan, and governance model all flow from these boundaries.
          </p>
          <div className="space-y-2">
            {LAYERS.map((layer, i) => (
              <button
                key={layer.name}
                type="button"
                onClick={() => setActiveLayer(activeLayer === i ? null : i)}
                className={`w-full text-left rounded-lg border-l-4 ${layer.color} border border-slate-200 bg-slate-50 p-4 transition-all hover:bg-slate-50 ${activeLayer === i ? "ring-1 ring-slate-300" : ""}`}
              >
                <div className="flex items-center justify-between">
                  <h3 className="font-semibold text-slate-900">{layer.name}</h3>
                  {overview?.architecture?.components_per_layer?.[i] != null && (
                    <span className="text-xs text-slate-500">
                      {overview.architecture.components_per_layer[i]} components
                    </span>
                  )}
                </div>
                {activeLayer === i && (
                  <p className="mt-2 text-sm text-slate-500">{layer.desc}</p>
                )}
              </button>
            ))}
          </div>
          <GuidedInsight type="notice">
            <p>
              Five layers: Data → ML → API → Frontend → LLM. Each layer has specific
              responsibilities and failure modes. A failure in the Data Layer cascades
              to every model downstream.
            </p>
          </GuidedInsight>

          {Object.keys(categories).length > 0 && (
            <div className="rounded-lg border border-slate-200 bg-slate-50 p-4">
              <h4 className="mb-3 text-sm font-medium text-slate-500">Module distribution</h4>
              <div className="flex flex-wrap gap-2">
                {Object.entries(categories).map(([cat, count]) => (
                  <span key={cat} className="rounded-full bg-white px-3 py-1 text-xs text-slate-600">
                    {cat}: {count}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      </StorySection>

      <StorySection beat="BUILD" title="Governance checklist">
        <div className="space-y-4">
          <p className="text-slate-500">
            A production ML system requires more than working models. Governance isn&apos;t bureaucracy — it&apos;s how you earn trust from finance, legal, and executive stakeholders. Each checklist item maps to a business risk: model versioning prevents &quot;what changed?&quot; confusion; drift monitoring catches stale models before they recommend wrong prices; audit trails support compliance and dispute resolution. Review each governance item and assess your readiness.
          </p>
          <div className="space-y-2">
            {checklist.length > 0 ? (
              checklist.map((item, i) => {
                const statusColors: Record<string, string> = {
                  complete: "bg-emerald-500",
                  partial: "bg-amber-500",
                  pending: "bg-slate-300",
                };
                return (
                  <div
                    key={i}
                    className="flex items-center gap-3 rounded-lg border border-slate-200 bg-slate-50 px-4 py-3"
                  >
                    <span className={`h-2.5 w-2.5 rounded-full shrink-0 ${statusColors[item.status] ?? "bg-slate-300"}`} />
                    <span className="flex-1 text-sm text-slate-600">{item.item}</span>
                    <span className="text-xs text-slate-500 capitalize">{item.status}</span>
                  </div>
                );
              })
            ) : (
              <div className="text-sm text-slate-500">Loading governance checklist...</div>
            )}
          </div>
          <GuidedInsight type="try">
            <p>
              Click each architecture layer above. Think about what happens if that layer
              fails. Which failure has the highest business impact?
            </p>
          </GuidedInsight>
        </div>
      </StorySection>

      <StorySection beat="INTERROGATE" title="Where the system breaks">
        <div className="space-y-4">
          <p className="text-slate-500">
            Real-world deployment faces challenges beyond code: data quality varies by region, stakeholders resist &quot;black box&quot; recommendations, and legacy systems don&apos;t expose clean APIs. The technical architecture is necessary but not sufficient — organizational readiness matters as much as model accuracy.
          </p>
          <GuidedInsight type="think">
            <p>
              What happens when the regression model (M01) estimates elasticity at −1.5
              but the optimization model (M09) was calibrated with elasticity −1.2?
              The optimizer will recommend prices that are too aggressive. You need
              pipeline orchestration to keep models in sync.
            </p>
          </GuidedInsight>
          <GuidedInsight type="warning">
            <p>
              Model monitoring is listed as "partial" — this means you can't detect
              model drift in production. If the relationship between price and volume
              changes (due to a new competitor, regulation, or macro shift), your models
              will silently produce wrong recommendations. This is the #1 risk in
              production ML.
            </p>
          </GuidedInsight>
          <GuidedInsight type="think">
            <p>
              The LLM layer adds a new class of risk: hallucination. If a pricing analyst
              asks the RAG pipeline &quot;What should Station 42 charge?&quot; and the LLM fabricates
              a number, the consequence is a real pricing decision based on fiction. How do
              you add guardrails?
            </p>
          </GuidedInsight>
          <GuidedInsight type="think">
            <p>
              Organizational readiness: Does your team have the skills to maintain 18 models? Are stakeholders prepared to trust automated recommendations, or do they need explainability first? A phased rollout — start with decision support, then move to automation — often succeeds where a big-bang deployment fails.
            </p>
          </GuidedInsight>
        </div>
      </StorySection>

      <StorySection beat="DECIDE" title="Your deployment strategy">
        <div className="space-y-4">
          <div className="rounded-lg border border-slate-200 bg-slate-50 p-5 text-slate-600 space-y-3">
            <p>
              Congratulations — you&apos;ve completed the full curriculum. You now have hands-on
              experience with regression, classification, clustering, anomaly detection,
              time series forecasting, optimization, reinforcement learning, neural networks,
              transformers, LLMs, and RAG.
            </p>
            <p>
              An implementation roadmap helps: <strong className="text-slate-900">Phase 1</strong> — ship the highest-impact models (elasticity, forecasting, optimization) with human-in-the-loop; <strong className="text-slate-900">Phase 2</strong> — add monitoring, drift detection, and automated retraining; <strong className="text-slate-900">Phase 3</strong> — introduce RAG and LLM-assisted analysis for explainability and Q&amp;A. Each phase delivers value while building confidence for the next.
            </p>
            <p className="font-medium text-slate-900">
              Final challenge: Design the deployment strategy for your pricing intelligence
              platform. Which models ship first? What monitoring do you need? How do you
              get VP approval?
            </p>
          </div>
          <Assessment questions={ASSESSMENT} />
        </div>
      </StorySection>
    </ModuleLayout>
  );
}
