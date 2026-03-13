"use client";

import { useState, useCallback, useEffect } from "react";
import { ModuleLayout } from "@/components/module-layout";
import { StorySection } from "@/components/story-section";
import { GuidedInsight } from "@/components/guided-insight";
import { ChartPanel } from "@/components/chart-panel";
import { Assessment, type AssessmentQuestion } from "@/components/assessment";
import { apiPost } from "@/lib/api-client";
import type { PlotlyFigure } from "@/components/chart-panel";

interface M13Response {
  figures?: Record<string, PlotlyFigure>;
  charts?: Record<string, PlotlyFigure>;
  metrics?: { accuracy?: number; r2?: number };
}

const ASSESSMENT_QUESTIONS: AssessmentQuestion[] = [
  {
    id: "q1",
    type: "critical",
    question:
      "A neural network achieves R²=0.85 but you can't explain individual predictions to regulators. Would you deploy it for automated pricing?",
    modelAnswer:
      "Consider: (1) Use SHAP or LIME for post-hoc explainability. (2) Deploy as a decision-support tool with human approval, not full automation. (3) Hybrid: use the NN for ranking/candidates, a simpler model for the final explainable recommendation. (4) Regulatory context matters — some jurisdictions require explainability for automated decisions.",
  },
];

export default function M13NeuralPage() {
  const [layers, setLayers] = useState(2);
  const [units, setUnits] = useState(64);
  const [activation, setActivation] = useState<"relu" | "tanh" | "selu">("relu");
  const [figures, setFigures] = useState<Record<string, PlotlyFigure | null>>({});
  const [metrics, setMetrics] = useState<M13Response["metrics"]>({});
  const [loading, setLoading] = useState(false);
  const [exploreLoading, setExploreLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const train = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await apiPost<M13Response>("/api/m13/train", {
        n_layers: layers,
        units,
        activation,
      });
      const figs = res?.figures ?? res?.charts ?? {};
      setFigures(figs as Record<string, PlotlyFigure | null>);
      if (res?.metrics) setMetrics(res.metrics);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Training failed");
    } finally {
      setLoading(false);
    }
  }, [layers, units, activation]);

  useEffect(() => {
    let cancelled = false;
    setExploreLoading(true);
    apiPost<M13Response>("/api/m13/train", {
      n_layers: 2,
      units: 64,
      activation: "relu",
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

  const boundaryFigure =
    figures.primary ?? figures.decision_boundary ?? figures.decisionBoundary ?? null;

  return (
    <ModuleLayout
      chapterNumber={13}
      title="Neural Networks"
      subtitle="How does a neural network decide whether to cut price or hold?"
      estimatedMinutes={25}
      learningObjectives={["MLP architecture", "Activation functions", "Decision boundaries"]}
      currentSlug="m13-neural"
    >
      <StorySection beat="FRAME" title="Universal function approximators">
        <div className="space-y-4 text-slate-500">
          <p>
            You&apos;ve used tree-based models (M03) and linear models (M01). Neural networks are
            universal function approximators — they can theoretically learn ANY relationship between
            inputs and outputs. But this power comes with opacity. A VP can&apos;t inspect
            &quot;weights in layer 3&quot; the way they can inspect a regression coefficient.
          </p>
          <p>
            Neural networks power everything from fraud detection in banking to recommendation
            engines in retail. For pricing, they excel when the relationship between features and
            outcomes is highly nonlinear — such as demand response near psychological price points
            (EUR 1.99 vs 2.01) or during promotional events.
          </p>
          <p>
            The architecture is straightforward: the input layer receives your features (price,
            cost, competitor gap, etc.); hidden layers learn increasingly abstract patterns; the
            output layer produces the prediction. Activation functions — ReLU, tanh, SELU — determine
            what patterns each neuron can capture. ReLU is fast and avoids vanishing gradients;
            tanh squashes values to [-1, 1] and can model smoother boundaries; SELU enables
            self-normalizing networks that train more stably.
          </p>
        </div>
      </StorySection>

      <StorySection beat="EXPLORE" title="Decision boundary">
        <div className="space-y-4">
          <p className="text-slate-500">
            The decision boundary visualizes how the network partitions the feature space into
            &quot;hold&quot; vs &quot;cut&quot; regions. Unlike a linear model, which draws a
            single straight line, a neural network can carve out curved, even disjoint regions.
            The shape of the boundary reflects the model&apos;s capacity: too little capacity and
            it underfits; too much and it memorizes noise.
          </p>
          <ChartPanel
            figure={boundaryFigure}
            title="Hold vs cut decision boundary"
            loading={exploreLoading}
          />
          <GuidedInsight type="notice">
            The colored regions show where the network predicts &quot;hold&quot; vs
            &quot;cut.&quot; Compare this to the linear decision boundary from M01.
          </GuidedInsight>
        </div>
      </StorySection>

      <StorySection beat="BUILD" title="Train the neural network">
        <div className="space-y-4">
          <p className="text-slate-500">
            Layers control depth — more layers let the network learn hierarchical patterns (e.g.,
            raw features → price elasticity → regional effects). Units (hidden dimension) control
            width — more units allow finer-grained distinctions. Activation shapes the boundary:
            ReLU yields piecewise-linear regions; tanh yields smoother curves; SELU often trains
            better in deep stacks. For pricing, start with 2 layers and 64 units; increase only if
            validation performance justifies the added complexity.
          </p>
          <div className="flex flex-wrap items-center gap-6">
            <label className="flex flex-col gap-1">
              <span className="text-xs text-slate-500">Layers (1–5)</span>
              <input
                type="range"
                min={1}
                max={5}
                value={layers}
                onChange={(e) => setLayers(Number(e.target.value))}
                className="w-32"
              />
              <span className="text-xs text-slate-500">{layers}</span>
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-xs text-slate-500">Units (16–128)</span>
              <input
                type="range"
                min={16}
                max={128}
                step={16}
                value={units}
                onChange={(e) => setUnits(Number(e.target.value))}
                className="w-32"
              />
              <span className="text-xs text-slate-500">{units}</span>
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-xs text-slate-500">Activation</span>
              <select
                value={activation}
                onChange={(e) => setActivation(e.target.value as "relu" | "tanh" | "selu")}
                className="rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm text-slate-600"
              >
                <option value="relu">ReLU</option>
                <option value="tanh">Tanh</option>
                <option value="selu">SELU</option>
              </select>
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
          {(metrics?.accuracy != null || metrics?.r2 != null) && (
            <div className="rounded-lg border border-slate-200 bg-white/50 px-4 py-2">
              <span className="text-xs text-slate-500">Metric</span>
              <p className="text-lg font-semibold text-emerald-400">
                {metrics.accuracy != null
                  ? `Accuracy: ${(metrics.accuracy * 100).toFixed(1)}%`
                  : `R²: ${metrics.r2?.toFixed(3) ?? "—"}`}
              </p>
            </div>
          )}
          <ChartPanel
            figure={boundaryFigure}
            title="Decision boundary"
            loading={loading}
          />
        </div>
      </StorySection>

      <StorySection beat="INTERROGATE" title="Architecture tradeoffs">
        <div className="space-y-4">
          <GuidedInsight type="try">
            Set layers=1, units=16 with tanh. The boundary is smooth but can&apos;t capture complex
            patterns.
          </GuidedInsight>
          <GuidedInsight type="try">
            Set layers=5, units=128 with relu. The boundary becomes jagged — overfitting.
          </GuidedInsight>
          <GuidedInsight type="warning">
            Deep narrow networks train faster but underfit. Shallow wide networks overfit. The sweet
            spot is problem-dependent.
          </GuidedInsight>
        </div>
      </StorySection>

      <StorySection beat="DECIDE" title="Deploy for automated pricing?">
        <div className="space-y-4">
          <p className="text-slate-500">
            A neural network achieves R²=0.85 but you can&apos;t explain individual predictions to
            regulators. Would you deploy it for automated pricing?
          </p>
          <Assessment questions={ASSESSMENT_QUESTIONS} />
        </div>
      </StorySection>
    </ModuleLayout>
  );
}
