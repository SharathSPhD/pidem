"use client";

import { useState, useCallback, useEffect } from "react";
import { ModuleLayout } from "@/components/module-layout";
import { StorySection } from "@/components/story-section";
import { GuidedInsight } from "@/components/guided-insight";
import { ChartPanel } from "@/components/chart-panel";
import { Assessment, type AssessmentQuestion } from "@/components/assessment";
import { apiPost } from "@/lib/api-client";
import type { PlotlyFigure } from "@/components/chart-panel";

interface M14Response {
  figures?: Record<string, PlotlyFigure>;
  charts?: Record<string, PlotlyFigure>;
  metrics?: { r2?: number };
}

const ASSESSMENT_QUESTIONS: AssessmentQuestion[] = [
  {
    id: "q1",
    type: "critical",
    question:
      "FT-Transformer matches XGBoost at R²=0.78 but trains 10x slower. For production pricing, which would you choose?",
    modelAnswer:
      "XGBoost is typically preferred for production: faster training, lower compute, mature tooling. Choose FT-Transformer when: (1) you need attention-based interpretability, (2) feature interactions are complex and trees underperform, (3) you have abundant data and compute. For most pricing use cases, XGBoost is the pragmatic choice.",
  },
];

export default function M14FtTransformerPage() {
  const [nHeads, setNHeads] = useState(4);
  const [nLayers, setNLayers] = useState(2);
  const [compare, setCompare] = useState(false);
  const [figures, setFigures] = useState<Record<string, PlotlyFigure | null>>({});
  const [metrics, setMetrics] = useState<M14Response["metrics"]>({});
  const [loading, setLoading] = useState(false);
  const [exploreLoading, setExploreLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const train = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await apiPost<M14Response>("/api/m14/train", {
        n_heads: nHeads,
        n_layers: nLayers,
        compare_mode: compare,
      });
      const figs = res?.figures ?? res?.charts ?? {};
      setFigures(figs as Record<string, PlotlyFigure | null>);
      if (res?.metrics) setMetrics(res.metrics);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Training failed");
    } finally {
      setLoading(false);
    }
  }, [nHeads, nLayers, compare]);

  useEffect(() => {
    let cancelled = false;
    setExploreLoading(true);
    apiPost<M14Response>("/api/m14/train", {
      n_heads: 4,
      n_layers: 2,
      compare_mode: false,
    })
      .then((res) => {
        if (!cancelled) {
          const figs = res?.figures ?? res?.charts ?? {};
          setFigures(figs as Record<string, PlotlyFigure | null>);
          if (res?.metrics) setMetrics(res.metrics);
        }
      })
      .catch(() => {})
      .finally(() => {
        if (!cancelled) setExploreLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const attentionFigure = figures.primary ?? figures.attention ?? null;
  const comparisonFigure = figures.comparison ?? null;

  return (
    <ModuleLayout
      chapterNumber={14}
      title="FT-Transformer"
      subtitle="Can attention on tabular data beat gradient boosting?"
      estimatedMinutes={20}
      learningObjectives={[
        "Feature tokenization",
        "Self-attention on tabular data",
        "Comparison with XGBoost",
      ]}
      currentSlug="m14-ft-transformer"
    >
      <StorySection beat="FRAME" title="Attention on tabular features">
        <div className="space-y-4 text-slate-500">
          <p>
            FT-Transformer treats each tabular feature as a &quot;token&quot; (like a word in NLP)
            and applies self-attention. The breakthrough is feature interaction discovery — the
            model can learn that the combination of &quot;competitor_price_gap&quot; and
            &quot;day_of_week&quot; matters more than either alone. Trees discover interactions
            through splits; attention discovers them through learned weights.
          </p>
          <p>
            Attention on tabular data is transforming every industry. Banks use it for credit
            scoring. Retailers use it for personalized pricing. The question for your pricing team:
            does attention beat gradient boosting on YOUR data?
          </p>
          <p>
            Self-attention lets each feature &quot;look at&quot; every other feature and decide
            which ones matter most for this specific prediction. When predicting demand for a
            product, the model might attend strongly to price when cost is high, and to competitor
            gap when it&apos;s low. That context-dependent weighting is what trees cannot easily
            express.
          </p>
        </div>
      </StorySection>

      <StorySection beat="EXPLORE" title="Attention heatmap">
        <div className="space-y-4">
          <p className="text-slate-500">
            The attention heatmap shows which features attend to which others. Each row is a
            &quot;query&quot; feature; each column is a &quot;key&quot; it attends to. Brighter
            cells mean stronger attention. Unlike SHAP, which gives a single global importance
            score, attention reveals pairwise relationships — e.g., does price attend to cost, or to
            competitor gap?
          </p>
          <ChartPanel
            figure={attentionFigure}
            title="Feature attention"
            loading={exploreLoading}
          />
          <GuidedInsight type="notice">
            The attention map shows which features attend to each other. High attention between
            price and crude_eur means the model learned the cost-passthrough relationship.
          </GuidedInsight>
        </div>
      </StorySection>

      <StorySection beat="BUILD" title="Train the FT-Transformer">
        <div className="space-y-4">
          <div className="flex flex-wrap items-center gap-6">
            <label className="flex flex-col gap-1">
              <span className="text-xs text-slate-500">Attention heads (1–8)</span>
              <input
                type="range"
                min={1}
                max={8}
                value={nHeads}
                onChange={(e) => setNHeads(Number(e.target.value))}
                className="w-32"
              />
              <span className="text-xs text-slate-500">{nHeads}</span>
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-xs text-slate-500">Layers (1–4)</span>
              <input
                type="range"
                min={1}
                max={4}
                value={nLayers}
                onChange={(e) => setNLayers(Number(e.target.value))}
                className="w-32"
              />
              <span className="text-xs text-slate-500">{nLayers}</span>
            </label>
            <label className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={compare}
                onChange={(e) => setCompare(e.target.checked)}
                className="rounded border-slate-200"
              />
              <span className="text-sm text-slate-600">Compare with XGBoost</span>
            </label>
            <button
              onClick={train}
              disabled={loading}
              className="rounded-lg bg-amber-500 px-4 py-2 text-sm font-medium text-slate-900 hover:bg-amber-400 disabled:opacity-50"
            >
              {loading ? "Training…" : "Train"}
            </button>
          </div>
          {error && <p className="text-sm text-rose-400">{error}</p>}
          {metrics?.r2 != null && (
            <div className="rounded-lg border border-slate-200 bg-slate-50 px-4 py-2">
              <span className="text-xs text-slate-500">R²</span>
              <p className="text-lg font-semibold text-emerald-400">{metrics.r2.toFixed(3)}</p>
            </div>
          )}
          <div className="grid gap-4 sm:grid-cols-2">
            <ChartPanel
              figure={attentionFigure}
              title="Attention heatmap"
              loading={loading}
            />
            {compare && (
              <ChartPanel
                figure={comparisonFigure}
                title="FT-Transformer vs XGBoost"
                loading={loading}
              />
            )}
          </div>
        </div>
      </StorySection>

      <StorySection beat="INTERROGATE" title="Interpreting attention">
        <div className="space-y-4">
          <p className="text-slate-500">
            Use the heatmap to validate business logic. If price attends strongly to crude_eur, the
            model has learned cost passthrough — expected for commodity pricing. If
            competitor_price_gap attends to day_of_week, the model may have captured that
            competitors react differently on weekdays vs weekends. Unexpected patterns (e.g., a
            weak feature dominating) can signal data leakage or spurious correlations.
          </p>
          <GuidedInsight type="think">
            The attention mechanism provides interpretability that matches SHAP — you can see which
            features influence each prediction.
          </GuidedInsight>
          <GuidedInsight type="notice">
            With 1 head and 1 layer, the model is essentially a weighted average. More heads
            capture more diverse feature interactions.
          </GuidedInsight>
        </div>
      </StorySection>

      <StorySection beat="DECIDE" title="FT-Transformer or XGBoost?">
        <div className="space-y-4">
          <p className="text-slate-500">
            Production tradeoffs matter. XGBoost trains in seconds, deploys on CPU, and has mature
            tooling (feature importance, SHAP, hyperparameter tuning). FT-Transformer typically
            needs GPU for reasonable training times, requires more data to shine, and offers
            attention-based interpretability as a different lens. When both achieve similar R²,
            choose XGBoost for speed and simplicity unless you specifically need attention
            interpretability or have evidence that FT-Transformer outperforms on your data.
            FT-Transformer matches XGBoost at R²=0.78 but trains 10x slower. For production
            pricing, which would you choose?
          </p>
          <Assessment questions={ASSESSMENT_QUESTIONS} />
        </div>
      </StorySection>
    </ModuleLayout>
  );
}
