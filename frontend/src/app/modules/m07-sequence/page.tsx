"use client";

import { useState, useCallback, useEffect } from "react";
import { ModuleLayout } from "@/components/module-layout";
import { StorySection } from "@/components/story-section";
import { GuidedInsight } from "@/components/guided-insight";
import { ChartPanel } from "@/components/chart-panel";
import { Assessment, type AssessmentQuestion } from "@/components/assessment";
import { apiPost } from "@/lib/api-client";
import type { PlotlyFigure } from "@/components/chart-panel";

interface M07Response {
  figures?: Record<string, PlotlyFigure>;
  metrics?: { mape?: number; rmse?: number; r2?: number; [k: string]: number | string | undefined };
  data?: { importance?: Record<string, number> };
}

const STATIONS = ["STN_001", "STN_002"];

const ASSESSMENT_QUESTIONS: AssessmentQuestion[] = [
  {
    id: "q1",
    type: "conceptual",
    question: "Why does lag-7 (same day last week) capture weekly patterns?",
    modelAnswer:
      "Volume often follows day-of-week patterns (e.g. Sundays lower than weekdays). Lag-7 provides the model with the value from the same weekday in the previous week, directly encoding this cycle.",
  },
  {
    id: "q2",
    type: "applied",
    question: "What happens to MAPE when you set lags=3?",
    modelAnswer:
      "With only 3 lags, you lose the lag-7 (weekly) signal. MAPE typically increases because the model cannot capture the weekly cycle.",
  },
  {
    id: "q3",
    type: "critical",
    question: "Would you choose ARIMA (M06) or LightGBM lags for your procurement forecast? Consider interpretability, maintenance, and accuracy.",
    modelAnswer:
      "LightGBM lags: simpler to maintain, no stationarity assumptions, handles exogenous features easily. ARIMA: interpretable parameters, probabilistic forecasts. Choice depends on whether you need interpretability and uncertainty quantification vs. ease of deployment and feature flexibility.",
  },
];

export default function M07SequencePage() {
  const [lags, setLags] = useState(7);
  const [station, setStation] = useState("STN_001");
  const [figures, setFigures] = useState<Record<string, PlotlyFigure | null>>({});
  const [metrics, setMetrics] = useState<{ mape?: number; rmse?: number; r2?: number; [k: string]: number | string | undefined }>({});
  const [importance, setImportance] = useState<Record<string, number>>({});
  const [loading, setLoading] = useState(false);
  const [exploreLoading, setExploreLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const train = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await apiPost<M07Response>("/api/m07/train", {
        station_id: station,
        lags,
        lookback: 14,
      });
      if (res?.figures) setFigures(res.figures as Record<string, PlotlyFigure | null>);
      if (res?.metrics) setMetrics(res.metrics ?? {});
      if (res?.data?.importance) setImportance(res.data.importance);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Training failed");
    } finally {
      setLoading(false);
    }
  }, [station, lags]);

  useEffect(() => {
    let cancelled = false;
    setExploreLoading(true);
    apiPost<M07Response>("/api/m07/train", {
      station_id: "STN_001",
      lags: 7,
      lookback: 14,
    })
      .then((res) => {
        if (!cancelled && res?.figures) {
          setFigures(res.figures as Record<string, PlotlyFigure | null>);
          if (res?.metrics) setMetrics(res.metrics ?? {});
          if (res?.data?.importance) setImportance(res.data.importance);
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

  const mape = metrics?.mape as number | undefined;

  return (
    <ModuleLayout
      chapterNumber={7}
      title="Sequence Models"
      subtitle="Can yesterday's lag features outperform a full ARIMA model?"
      estimatedMinutes={20}
      learningObjectives={["Lag feature engineering", "LightGBM for time series", "Feature importance for lags"]}
      currentSlug="m07-sequence"
    >
      <StorySection beat="FRAME" title="The lag-feature alternative">
        <div className="space-y-4 text-slate-500">
          <p>
            ARIMA requires stationarity checks, parameter tuning, and careful diagnostics. What if
            you could skip all that complexity and just use &quot;yesterday&apos;s volume&quot; and
            &quot;same day last week&quot; as features in a gradient booster? This remarkably simple
            approach often matches or beats ARIMA — and it scales to thousands of locations without
            manual tuning.
          </p>
          <p>
            The lag-feature approach works for any time series. A retailer uses yesterday&apos;s
            foot traffic and last week&apos;s same-day sales. A logistics company uses last
            month&apos;s shipping volume and same-month-last-year patterns. The idea is the same: let
            the machine learning algorithm discover temporal patterns through engineered features.
          </p>
          <p>
            Lag features encode temporal patterns because they explicitly give the model access to
            past values. Lag-1 captures yesterday; lag-7 captures the same weekday last week. The
            model learns how these relate to today. The trade-off: statistical models like ARIMA
            provide interpretable parameters and probabilistic forecasts, but lag-based gradient
            boosting is simpler to maintain, handles exogenous features easily, and often achieves
            comparable or better accuracy with far less manual effort.
          </p>
        </div>
      </StorySection>

      <StorySection beat="EXPLORE" title="Lag model predictions">
        <div className="space-y-4">
          <p className="text-slate-500">
            Below you see predictions from a LightGBM model trained on lag features. The model uses
            past values (lag-1, lag-2, … lag-7 and beyond) as inputs to predict the next day. Notice
            how well it tracks the actual series — especially the weekly ups and downs.
          </p>
          <GuidedInsight type="notice">
            The lag model captures day-of-week patterns automatically through the lag-7 feature.
          </GuidedInsight>
          <ChartPanel
            figure={figures.primary ?? null}
            title="Predictions vs actual"
            loading={exploreLoading}
          />
        </div>
      </StorySection>

      <StorySection beat="BUILD" title="Tune lags and station">
        <div className="space-y-4">
          <p className="text-slate-500">
            Use the controls to experiment. <strong>Station</strong> switches between locations with
            different demand patterns. <strong>Lags</strong> controls how many past days the model
            uses: fewer lags (e.g. 3) may miss weekly cycles; more lags (e.g. 14) capture longer
            history but add noise. The sweet spot for weekly data is often 7 (yesterday through same
            day last week).
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
            <label className="flex flex-col gap-1">
              <span className="text-xs text-slate-500">Lags (3–14)</span>
              <input
                type="range"
                min={3}
                max={14}
                value={lags}
                onChange={(e) => setLags(Number(e.target.value))}
                className="w-32"
              />
              <span className="text-xs text-slate-500">{lags}</span>
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
          <div className="grid gap-4 sm:grid-cols-2">
            <ChartPanel
              figure={figures.primary ?? null}
              title="Predictions"
              loading={loading}
            />
            <ChartPanel
              figure={figures.importance ?? null}
              title="Lag importance"
              loading={loading}
            />
          </div>
          {typeof mape === "number" && (
            <p className="text-sm text-slate-500">
              MAPE: <span className="font-mono text-amber-600">{mape}%</span>
              {metrics?.rmse != null && (
                <span>{` · RMSE: ${metrics.rmse}`}</span>
              )}
              {metrics?.r2 != null && (
                <span>{` · R²: ${metrics.r2}`}</span>
              )}
            </p>
          )}
        </div>
      </StorySection>

      <StorySection beat="INTERROGATE" title="Interpreting lag importance">
        <div className="space-y-4">
          <p className="text-slate-500">
            Feature importance tells you which lags the model relies on most. In fuel volume (and
            many retail series), lag-1 and lag-7 typically dominate: yesterday&apos;s value is the
            best single predictor, and same-day-last-week captures the weekly rhythm. The same
            pattern appears in other domains — a restaurant&apos;s lag-7 is last Saturday&apos;s
            covers; a utility&apos;s lag-7 is last week&apos;s demand.
          </p>
          <GuidedInsight type="notice">
            Lag-1 (yesterday) dominates, but lag-7 (same day last week) is the second most
            important — this is the weekly cycle.
          </GuidedInsight>
          <GuidedInsight type="try">
            Set lags=3. You lose the weekly signal and MAPE increases.
          </GuidedInsight>
          <GuidedInsight type="think">
            Why might lag-7 matter more than lag-2 or lag-3? Day-of-week effects create a 7-day
            cycle. Lag-7 gives the model the value from the same weekday; lag-2 and lag-3 give
            different weekdays, which are less informative for predicting today.
          </GuidedInsight>
          <ChartPanel figure={figures.importance ?? null} title="Lag importance" />
        </div>
      </StorySection>

      <StorySection beat="DECIDE" title="ARIMA vs LightGBM lags">
        <div className="space-y-4 text-slate-500">
          <p>
            Would you choose ARIMA (M06) or LightGBM lags for your procurement forecast? Use this
            decision framework: <strong>ARIMA</strong> offers interpretable parameters (autoregressive
            order, differencing), built-in probabilistic forecasts, and a rigorous statistical
            foundation — but it assumes stationarity, requires manual tuning, and struggles with
            exogenous features. <strong>LightGBM lags</strong> are simpler to deploy, scale to many
            locations without per-series tuning, and easily incorporate price, weather, or holidays
            — but they lack native uncertainty quantification and the parameters are less
            interpretable.
          </p>
          <p>
            Choose ARIMA when you need explainable forecasts and uncertainty intervals for
            inventory or capacity planning. Choose LightGBM lags when you have many series, rich
            exogenous data, and prefer operational simplicity over statistical elegance.
          </p>
        </div>
      </StorySection>

      <Assessment questions={ASSESSMENT_QUESTIONS} />
    </ModuleLayout>
  );
}
