"use client";

import { useState, useCallback, useEffect } from "react";
import { ModuleLayout } from "@/components/module-layout";
import { StorySection } from "@/components/story-section";
import { GuidedInsight } from "@/components/guided-insight";
import { ChartPanel } from "@/components/chart-panel";
import { Assessment, type AssessmentQuestion } from "@/components/assessment";
import { apiPost } from "@/lib/api-client";
import type { PlotlyFigure } from "@/components/chart-panel";

interface M08Response {
  figures?: Record<string, PlotlyFigure>;
  metrics?: Record<string, number | string | null | undefined>;
}

const STATIONS = ["STN_001", "STN_002"];
const QUANTILE_OPTIONS = [0.1, 0.25, 0.5, 0.75, 0.9];

const ASSESSMENT_QUESTIONS: AssessmentQuestion[] = [
  {
    id: "q1",
    type: "conceptual",
    question: "What does the attention heatmap reveal?",
    modelAnswer:
      "The attention heatmap shows which input features the model relies on most when making predictions. High attention on price and holiday features indicates the model has learned that these drive volume.",
  },
  {
    id: "q2",
    type: "applied",
    question: "What happens if you ablate the holiday input?",
    modelAnswer:
      "The attention shifts to other features (e.g. price, temperature), but forecast accuracy typically drops because the model loses a key explanatory signal.",
  },
  {
    id: "q3",
    type: "critical",
    question: "How would you communicate Q10/Q50/Q90 to a supply chain manager who thinks in 'single numbers'?",
    modelAnswer:
      "Frame it as a range: 'We expect 80% of outcomes to fall between Q10 and Q90.' Use Q50 as the planning number and the interval for risk buffers. For procurement, Q90 helps size safety stock.",
  },
];

export default function M08TftPage() {
  const [station, setStation] = useState("STN_001");
  const [quantiles, setQuantiles] = useState<number[]>([0.1, 0.5, 0.9]);
  const [figures, setFigures] = useState<Record<string, PlotlyFigure | null>>({});
  const [metrics, setMetrics] = useState<Record<string, number | string | null | undefined>>({});
  const [loading, setLoading] = useState(false);
  const [exploreLoading, setExploreLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const toggleQuantile = (q: number) => {
    setQuantiles((prev) =>
      prev.includes(q) ? prev.filter((x) => x !== q) : [...prev, q].sort((a, b) => a - b)
    );
  };

  const train = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await apiPost<M08Response>("/api/m08/train", {
        station_id: station,
        quantiles,
      });
      if (res?.figures) setFigures(res.figures as Record<string, PlotlyFigure | null>);
      if (res?.metrics) setMetrics(res.metrics ?? {});
    } catch (e) {
      setError(e instanceof Error ? e.message : "Training failed");
    } finally {
      setLoading(false);
    }
  }, [station, quantiles]);

  useEffect(() => {
    let cancelled = false;
    setExploreLoading(true);
    apiPost<M08Response>("/api/m08/train", {
      station_id: "STN_001",
      quantiles: [0.1, 0.5, 0.9],
    })
      .then((res) => {
        if (!cancelled && res?.figures) {
          setFigures(res.figures as Record<string, PlotlyFigure | null>);
          if (res?.metrics) setMetrics(res.metrics ?? {});
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

  return (
    <ModuleLayout
      chapterNumber={8}
      title="Temporal Fusion Transformer"
      subtitle="What if a model could tell you which inputs it's paying attention to?"
      estimatedMinutes={25}
      learningObjectives={["Multi-horizon forecasting", "Attention mechanism intuition", "Quantile prediction"]}
      currentSlug="m08-tft"
    >
      <StorySection beat="FRAME" title="Interpretable probabilistic forecasting">
        <div className="space-y-4 text-slate-500">
          <p>
            Traditional forecasting models give you a single number. But a supply chain manager
            doesn&apos;t just want to know &quot;we&apos;ll sell 50,000 litres next week&quot; — they
            need to know &quot;we&apos;re 90% confident we&apos;ll sell between 42,000 and 58,000, and
            here&apos;s what&apos;s driving the uncertainty.&quot; The Temporal Fusion Transformer
            delivers exactly this.
          </p>
          <p>
            Probabilistic forecasting matters wherever downside risk is asymmetric. A hotel that
            over-projects occupancy by 10% just has empty rooms. A hotel that under-projects by 10%
            turns guests away. A retailer ordering perishable goods has a similar asymmetry —
            overstock is waste, understock is lost revenue. Knowing the full distribution, not just
            the mean, changes how you plan.
          </p>
          <p>
            The TFT combines attention (from transformers) with known future inputs like holidays and
            promotions. It produces quantile forecasts — Q10, Q50, Q90 — so you get prediction
            intervals, not just point estimates. And it shows attention heatmaps: which inputs the
            model is &quot;paying attention to&quot; when making each prediction. That interpretability
            is rare in deep learning.
          </p>
        </div>
      </StorySection>

      <StorySection beat="EXPLORE" title="Multi-quantile forecast">
        <div className="space-y-4">
          <p className="text-slate-500">
            The chart below shows a multi-quantile forecast: instead of one line, you see several.
            Q50 is the median (your best single guess); Q10 and Q90 form an 80% prediction interval.
            The shaded region between them captures uncertainty — wider bands mean the model is less
            confident.
          </p>
          <GuidedInsight type="notice">
            The shaded region between Q10 and Q90 is the 80% prediction interval — wider
            intervals mean more uncertainty.
          </GuidedInsight>
          <ChartPanel
            figure={figures.primary ?? null}
            title="Multi-quantile forecast"
            loading={exploreLoading}
          />
        </div>
      </StorySection>

      <StorySection beat="BUILD" title="Select quantiles and station">
        <div className="space-y-4">
          <p className="text-slate-500">
            <strong>Quantiles</strong> define which parts of the forecast distribution you want to
            see. Q10 (10th percentile) is your downside case — useful for safety stock and
            worst-case planning. Q50 is the median — the typical planning number. Q90 is the upside
            — useful for capacity and staffing. Select the quantiles that match your risk appetite:
            conservative planners focus on Q10–Q50; growth-oriented teams may care more about Q75–Q90.
          </p>
          <div className="flex flex-wrap items-center gap-4">
            <label className="flex flex-col gap-1">
              <span className="text-xs text-slate-500">Station</span>
              <select
                value={station}
                onChange={(e) => setStation(e.target.value)}
                className="rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm text-slate-600"
              >
                {STATIONS.map((s) => (
                  <option key={s} value={s}>{s}</option>
                ))}
              </select>
            </label>
            <div className="flex flex-col gap-1">
              <span className="text-xs text-slate-500">Quantiles</span>
              <div className="flex flex-wrap gap-2">
                {QUANTILE_OPTIONS.map((q) => (
                  <label key={q} className="flex items-center gap-1">
                    <input
                      type="checkbox"
                      checked={quantiles.includes(q)}
                      onChange={() => toggleQuantile(q)}
                      className="rounded border-slate-200 text-amber-500 focus:ring-amber-500"
                    />
                    <span className="text-sm text-slate-600">Q{q * 100}</span>
                  </label>
                ))}
              </div>
            </div>
            <button
              onClick={train}
              disabled={loading}
              className="rounded-lg bg-amber-500 px-4 py-2 text-sm font-medium text-slate-900 hover:bg-amber-400 disabled:opacity-50"
            >
              {loading ? "Training…" : "Train"}
            </button>
          </div>
          {error && <p className="text-sm text-rose-400">{error}</p>}
          <div className="grid gap-4 sm:grid-cols-2">
            <ChartPanel
              figure={figures.primary ?? null}
              title="Forecast"
              loading={loading}
            />
            <ChartPanel
              figure={figures.attention ?? null}
              title="Attention heatmap"
              loading={loading}
            />
          </div>
        </div>
      </StorySection>

      <StorySection beat="INTERROGATE" title="Interpreting attention">
        <div className="space-y-4">
          <p className="text-slate-500">
            The attention heatmap shows which input features the model relies on most at each
            timestep. High attention on price and holidays means the model has learned that these
            drive volume. This interpretability is powerful: in retail, you might see attention spike
            on promotions; in energy, on temperature. The same mechanism applies across domains —
            the model reveals what it &quot;cares about&quot; for each prediction.
          </p>
          <GuidedInsight type="notice">
            The model pays high attention to price and holiday features — it&apos;s learned that
            pricing and holidays drive volume.
          </GuidedInsight>
          <GuidedInsight type="think">
            What happens if you ablate the holiday input? The attention shifts elsewhere but
            accuracy drops. This suggests the model genuinely uses that signal — removing it
            degrades forecasts. Use ablation to validate which inputs matter.
          </GuidedInsight>
          <ChartPanel figure={figures.attention ?? null} title="Attention heatmap" />
        </div>
      </StorySection>

      <StorySection beat="DECIDE" title="Communicating probabilistic forecasts">
        <div className="space-y-4 text-slate-500">
          <p>
            TFT gives probabilistic forecasts. How would you communicate Q10/Q50/Q90 to a supply
            chain manager who thinks in &quot;single numbers&quot;? Use a communication framework:
            <strong> Q50</strong> is your planning number — the most likely outcome.
            <strong> Q10–Q90</strong> is the 80% prediction interval — frame it as &quot;we expect
            80% of outcomes to fall between these bounds.&quot; For procurement, Q90 helps size safety
            stock; for capacity, Q10 helps plan for low-demand scenarios.
          </p>
          <p>
            Emphasize that a single number hides risk. A point forecast of 50,000 could mean
            &quot;almost always 50,000&quot; or &quot;ranges from 30,000 to 70,000.&quot; Quantiles make that
            distinction explicit and support better decisions.
          </p>
        </div>
      </StorySection>

      <Assessment questions={ASSESSMENT_QUESTIONS} />
    </ModuleLayout>
  );
}
