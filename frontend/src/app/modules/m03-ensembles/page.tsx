"use client";

import { useState, useCallback, useEffect } from "react";
import { ModuleLayout } from "@/components/module-layout";
import { StorySection } from "@/components/story-section";
import { GuidedInsight } from "@/components/guided-insight";
import { ChartPanel } from "@/components/chart-panel";
import { Assessment, type AssessmentQuestion } from "@/components/assessment";
import { apiPost } from "@/lib/api-client";
import type { PlotlyFigure } from "@/components/chart-panel";

interface TrainResponse {
  figures?: Record<string, PlotlyFigure>;
  metrics?: { r2?: number; rmse?: number };
  data?: { importance?: Record<string, number> };
}

const ASSESSMENT_QUESTIONS: AssessmentQuestion[] = [
  {
    id: "q1",
    type: "conceptual",
    question: "What does feature importance tell you vs SHAP?",
    options: [
      "Importance = SHAP",
      "Importance tells you WHICH features matter; SHAP tells you HOW they affect the prediction",
      "SHAP is only for tree models",
      "Importance is more accurate",
    ],
    modelAnswer:
      "Importance tells you which features matter overall (e.g. price_gap is top). SHAP tells you how each feature pushes a prediction up or down for each observation — e.g. 'for this station, high price_gap pushed volume UP'.",
  },
  {
    id: "q2",
    type: "applied",
    question: "A VP asks: 'Why did the model predict low volume for Station 42?' How do you answer?",
    options: [
      "Show the feature importance chart",
      "Show the SHAP beeswarm or waterfall for that observation",
      "Show the raw feature values",
      "Explain the R² of the model",
    ],
    modelAnswer:
      "SHAP gives you the answer. A SHAP waterfall or beeswarm for that observation shows which features pushed the prediction down (e.g. low price_gap, high our_price) and by how much.",
  },
  {
    id: "q3",
    type: "critical",
    question: "When might you prefer a single regression model over an ensemble?",
    options: [
      "When interpretability is critical",
      "When you have very little data",
      "When the single model already achieves acceptable R²",
      "All of the above",
    ],
    modelAnswer:
      "All of the above. Ensembles are black boxes; a single linear model has clear coefficients. With little data, ensembles can overfit. If R²=0.65 is good enough for the business, simplicity wins.",
  },
];

export default function M03EnsemblesPage() {
  const [nEstimators, setNEstimators] = useState(100);
  const [learningRate, setLearningRate] = useState(0.1);
  const [loading, setLoading] = useState(false);
  const [exploreFigure, setExploreFigure] = useState<PlotlyFigure | null>(null);
  const [buildFigure, setBuildFigure] = useState<PlotlyFigure | null>(null);
  const [shapFigure, setShapFigure] = useState<PlotlyFigure | null>(null);
  const [r2, setR2] = useState<number | null>(null);
  const [rmse, setRmse] = useState<number | null>(null);

  const train = useCallback(async () => {
    setLoading(true);
    setBuildFigure(null);
    setShapFigure(null);
    setR2(null);
    setRmse(null);
    try {
      const res = await apiPost<TrainResponse>("/api/m03/train", {
        n_estimators: nEstimators,
        learning_rate: learningRate,
        shap_sample: 100,
      });
      const figs = res?.figures ?? {};
      if (figs.primary) setBuildFigure(figs.primary as PlotlyFigure);
      if (figs.importance) setBuildFigure(figs.importance as PlotlyFigure);
      if (figs.shap_beeswarm) setShapFigure(figs.shap_beeswarm as PlotlyFigure);
      if (res?.metrics?.r2 != null) setR2(res.metrics.r2);
      if (res?.metrics?.rmse != null) setRmse(res.metrics.rmse);
    } catch {
      // Error handled by UI
    } finally {
      setLoading(false);
    }
  }, [nEstimators, learningRate]);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const res = await apiPost<TrainResponse>("/api/m03/train", {
          n_estimators: 100,
          learning_rate: 0.1,
          shap_sample: 100,
        });
        if (!cancelled && res?.figures?.primary) {
          setExploreFigure(res.figures.primary as PlotlyFigure);
        }
      } catch {
        // Ignore explore load errors
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <ModuleLayout
      chapterNumber={3}
      title="Ensemble Intelligence"
      subtitle="What if combining weak models creates a strong one?"
      estimatedMinutes={25}
      learningObjectives={[
        "Gradient boosting intuition",
        "SHAP explainability",
        "Feature importance vs SHAP",
      ]}
      currentSlug="m03-ensembles"
    >
      <StorySection beat="FRAME" title="Beyond single models">
        <div className="space-y-4 text-slate-500">
          <p>
            Your single regression model gives R²=0.65. The pricing committee
            needs R²&gt;0.80 to trust automated recommendations. Can you close
            the gap without sacrificing explainability?
          </p>
          <p>
            Ensemble methods are used across all pricing domains. A retailer
            combining store-level models for promotional pricing. A logistics
            company blending route-level demand forecasts. The principle is
            universal: combining diverse weak learners creates a strong one.
          </p>
          <p>
            Think of gradient boosting like a committee of analysts, where each
            new analyst focuses specifically on the cases the previous analysts
            got wrong. Each tree fits the residuals of the previous ones —
            correcting errors sequentially. The result: a model that captures
            nonlinear patterns and interactions that a single linear or shallow
            tree would miss.
          </p>
        </div>
      </StorySection>

      <StorySection beat="EXPLORE" title="Feature importance">
        <div className="space-y-4">
          <ChartPanel
            figure={exploreFigure}
            title="Feature importance (default model)"
            loading={!exploreFigure && !loading}
          />
          <GuidedInsight type="notice">
            Importance tells you WHICH features matter, but not HOW they affect
            the prediction. SHAP fills that gap.
          </GuidedInsight>
        </div>
      </StorySection>

      <StorySection beat="BUILD" title="Train the ensemble">
        <div className="space-y-4">
          <div className="flex flex-wrap gap-6">
            <label className="flex flex-col gap-1">
              <span className="text-xs text-slate-500">n_estimators (10–200)</span>
              <input
                type="range"
                min={10}
                max={200}
                step={10}
                value={nEstimators}
                onChange={(e) => setNEstimators(Number(e.target.value))}
                className="w-40"
              />
              <span className="text-xs text-slate-500">{nEstimators}</span>
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-xs text-slate-500">Learning rate (0.01–0.5)</span>
              <input
                type="range"
                min={0.01}
                max={0.5}
                step={0.01}
                value={learningRate}
                onChange={(e) => setLearningRate(Number(e.target.value))}
                className="w-40"
              />
              <span className="text-xs text-slate-500">{learningRate.toFixed(2)}</span>
            </label>
            <button
              onClick={train}
              disabled={loading}
              className="mt-2 rounded-lg bg-amber-500 px-4 py-2 text-sm font-medium text-slate-900 hover:bg-amber-400 disabled:opacity-50"
            >
              Train
            </button>
          </div>
          <div className="flex gap-4">
            {r2 != null && (
              <div className="rounded-lg border border-slate-200 bg-slate-50 px-4 py-2">
                <span className="text-xs text-slate-500">R²</span>
                <p className="text-lg font-semibold text-emerald-400">
                  {r2.toFixed(3)}
                </p>
              </div>
            )}
            {rmse != null && (
              <div className="rounded-lg border border-slate-200 bg-slate-50 px-4 py-2">
                <span className="text-xs text-slate-500">RMSE</span>
                <p className="text-lg font-semibold text-sky-400">
                  {rmse.toFixed(3)}
                </p>
              </div>
            )}
          </div>
          <ChartPanel
            figure={buildFigure}
            title="Feature importance"
            loading={loading}
          />
          {shapFigure && (
            <ChartPanel figure={shapFigure} title="SHAP beeswarm" />
          )}
          <GuidedInsight type="try">
            Set learning_rate to 0.5. The model overfits faster — fewer trees
            needed but less stable.
          </GuidedInsight>
        </div>
      </StorySection>

      <StorySection beat="INTERROGATE" title="SHAP beeswarm">
        <div className="space-y-4">
          <ChartPanel figure={shapFigure} title="SHAP beeswarm" />
          <GuidedInsight type="notice">
            Red dots on the right for price_gap mean higher price gaps push
            volume UP. This makes business sense — when competitors are
            expensive, you gain volume.
          </GuidedInsight>
        </div>
      </StorySection>

      <StorySection beat="DECIDE" title="Explaining predictions">
        <div className="space-y-4">
          <p className="text-slate-500">
            A VP asks: &quot;Why did the model predict low volume for Station
            42?&quot; SHAP gives you the answer — use the beeswarm or a
            per-observation waterfall to explain which features drove that
            prediction.
          </p>
          <Assessment questions={ASSESSMENT_QUESTIONS} />
        </div>
      </StorySection>
    </ModuleLayout>
  );
}
