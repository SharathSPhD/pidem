"use client";

import { useState, useCallback, useEffect } from "react";
import { ModuleLayout } from "@/components/module-layout";
import { StorySection } from "@/components/story-section";
import { GuidedInsight } from "@/components/guided-insight";
import { ChartPanel } from "@/components/chart-panel";
import { Assessment, type AssessmentQuestion } from "@/components/assessment";
import { apiPost } from "@/lib/api-client";
import type { PlotlyFigure } from "@/components/chart-panel";

interface M06Metrics {
  mape?: number;
  rmse?: number;
  [key: string]: number | string | undefined;
}

interface M06Response {
  figures?: Record<string, PlotlyFigure>;
  metrics?: M06Metrics;
}

const STATIONS = ["STN_001", "STN_002"];

const ASSESSMENT_QUESTIONS: AssessmentQuestion[] = [
  {
    id: "q1",
    type: "conceptual",
    question: "What do ARIMA (p,d,q) orders represent?",
    modelAnswer:
      "p = autoregressive order (how many past values predict the next). d = differencing order (how many times to difference for stationarity). q = moving average order (how many past errors influence the forecast).",
  },
  {
    id: "q2",
    type: "applied",
    question: "How does STL decomposition help forecasting?",
    modelAnswer:
      "STL separates the series into trend, seasonal, and residual components. This clarifies patterns (e.g. weekly seasonality) and allows modeling each part separately or removing seasonality before fitting ARIMA.",
  },
  {
    id: "q3",
    type: "critical",
    question: "Given MAPE of X%, is this forecast accurate enough for procurement decisions worth EUR 500K/week?",
    modelAnswer:
      "It depends on the MAPE level and risk tolerance. A 5% MAPE on EUR 500K implies ±EUR 25K uncertainty per week. Teams typically set accuracy thresholds (e.g. MAPE < 10%) before using forecasts for high-stakes decisions.",
  },
];

export default function M06TimeseriesPage() {
  const [station, setStation] = useState("STN_001");
  const [p, setP] = useState(2);
  const [d, setD] = useState(1);
  const [q, setQ] = useState(2);
  const [horizon, setHorizon] = useState(30);
  const [stlFigure, setStlFigure] = useState<PlotlyFigure | null>(null);
  const [forecastFigure, setForecastFigure] = useState<PlotlyFigure | null>(null);
  const [metrics, setMetrics] = useState<M06Metrics>({});
  const [loading, setLoading] = useState(false);
  const [exploreLoading, setExploreLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const train = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await apiPost<M06Response>("/api/m06/train", {
        station_id: station,
        method: "arima",
        p,
        d,
        q,
        horizon,
      });
      if (res?.figures?.primary) setForecastFigure(res.figures.primary);
      if (res?.metrics) setMetrics(res.metrics);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Training failed");
    } finally {
      setLoading(false);
    }
  }, [station, p, d, q, horizon]);

  useEffect(() => {
    let cancelled = false;
    setExploreLoading(true);
    apiPost<M06Response>("/api/m06/train", {
      station_id: "STN_001",
      method: "stl",
    })
      .then((res) => {
        if (!cancelled && res?.figures?.primary) {
          setStlFigure(res.figures.primary);
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
      chapterNumber={6}
      title="Classical Forecasting"
      subtitle="How far ahead can ARIMA see, and when does it go blind?"
      estimatedMinutes={25}
      learningObjectives={["ARIMA parameters (p,d,q)", "STL decomposition", "Forecast uncertainty"]}
      currentSlug="m06-timeseries"
    >
      <StorySection beat="FRAME" title="The procurement forecast problem">
        <div className="space-y-4 text-slate-500">
          <p>
            Your supply chain needs a 30-day volume forecast for procurement across 200 locations.
            Over-order by 10% and you tie up EUR 50K in working capital. Under-order by 10% and
            locations run dry — losing EUR 100K in missed sales. The cost of forecasting errors is
            asymmetric and real.
          </p>
          <p>
            Demand forecasting is universal. A grocery chain forecasts sandwich demand —
            perishable goods with zero tolerance for overstock. An industrial supplier forecasts
            monthly order volumes to negotiate capacity with manufacturers. A hotel revenue manager
            forecasts occupancy 90 days out to set room rates. The mechanics differ, but the question
            is the same: what will demand look like, and how much uncertainty should we plan for?
          </p>
          <p>
            <strong>ARIMA</strong> (AutoRegressive Integrated Moving Average) models capture
            autocorrelation — how today&apos;s value depends on past values and past errors. The
            (p,d,q) orders control that structure. <strong>STL</strong> (Seasonal-Trend
            decomposition using Loess) splits the series into three parts: trend (long-term
            direction), seasonal (repeating patterns like weekly or monthly cycles), and residual
            (what&apos;s left). Decomposition tells you what drives the signal: is the drop in
            volume a trend shift or just a seasonal dip? Seasonality matters because it&apos;s
            predictable — if Sundays are always slow, you don&apos;t need to &quot;forecast&quot;
            that; you need to forecast the trend and residual.
          </p>
        </div>
      </StorySection>

      <StorySection beat="EXPLORE" title="STL decomposition">
        <div className="space-y-4">
          <p className="text-slate-500">
            STL decomposes a time series into trend, seasonal, and residual components. The trend
            shows the underlying direction — is volume growing or declining? The seasonal component
            captures repeating patterns (e.g. weekly cycles, where Sundays differ from weekdays).
            The residual is the noise left after removing both. Understanding this structure helps
            you choose the right forecasting approach: if seasonality is strong, you may want to
            model it explicitly; if the residual is large, forecasts will be less reliable.
          </p>
          <GuidedInsight type="notice">
            The seasonal component shows clear weekly patterns — volume dips on Sundays.
          </GuidedInsight>
          <ChartPanel
            figure={stlFigure}
            title="STL decomposition"
            loading={exploreLoading}
          />
        </div>
      </StorySection>

      <StorySection beat="BUILD" title="Tune ARIMA and forecast horizon">
        <div className="space-y-4">
          <p className="text-slate-500">
            <strong>p</strong> (autoregressive order): how many past values predict the next. Higher
            p captures longer memory — useful when yesterday&apos;s volume strongly predicts
            today&apos;s. <strong>d</strong> (differencing order): how many times to difference the
            series for stationarity. d=1 often suffices for trending data. <strong>q</strong>
            (moving average order): how many past forecast errors influence the next prediction.
            Higher q helps when shocks persist. <strong>Horizon</strong> is how far ahead you
            forecast — 7 days for short-term replenishment, 30–60 days for procurement. Longer
            horizons have wider confidence intervals; forecasts degrade as you extend further.
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
              <span className="text-xs text-slate-500">p (0–4)</span>
              <input
                type="range"
                min={0}
                max={4}
                value={p}
                onChange={(e) => setP(Number(e.target.value))}
                className="w-24"
              />
              <span className="text-xs text-slate-500">{p}</span>
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-xs text-slate-500">d (0–4)</span>
              <input
                type="range"
                min={0}
                max={4}
                value={d}
                onChange={(e) => setD(Number(e.target.value))}
                className="w-24"
              />
              <span className="text-xs text-slate-500">{d}</span>
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-xs text-slate-500">q (0–4)</span>
              <input
                type="range"
                min={0}
                max={4}
                value={q}
                onChange={(e) => setQ(Number(e.target.value))}
                className="w-24"
              />
              <span className="text-xs text-slate-500">{q}</span>
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-xs text-slate-500">Horizon (7–60)</span>
              <input
                type="range"
                min={7}
                max={60}
                value={horizon}
                onChange={(e) => setHorizon(Number(e.target.value))}
                className="w-32"
              />
              <span className="text-xs text-slate-500">{horizon}</span>
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
          <ChartPanel
            figure={forecastFigure}
            title="Forecast"
            loading={loading}
          />
          {typeof mape === "number" && (
            <p className="text-sm text-slate-500">
              MAPE: <span className="font-mono text-amber-400">{mape}%</span>
              {metrics?.rmse != null && (
                <span>{` · RMSE: ${metrics.rmse}`}</span>
              )}
            </p>
          )}
        </div>
      </StorySection>

      <StorySection beat="INTERROGATE" title="Interpreting ARIMA parameters">
        <div className="space-y-4">
          <p className="text-slate-500">
            Parameters aren&apos;t arbitrary — they encode assumptions about how the series
            behaves. Set p=0, d=0, q=0 and the model has no structure to learn from; it collapses
            to a flat line. Overfit (e.g. p=4, q=4 on short data) and you capture noise as signal.
            The right balance depends on your data length and how much autocorrelation exists.
          </p>
          <GuidedInsight type="try">
            Set p=0, d=0, q=0. The model becomes a flat line — it can&apos;t learn anything.
          </GuidedInsight>
          <GuidedInsight type="warning">
            Forecasts beyond 14 days degrade rapidly. The confidence interval widens until it&apos;s
            useless.
          </GuidedInsight>
          <GuidedInsight type="think">
            Forecast horizon limits matter for planning. Use 7-day forecasts for replenishment;
            30-day for procurement with buffer; beyond 60 days, consider scenario planning or
            external indicators rather than pure extrapolation.
          </GuidedInsight>
        </div>
      </StorySection>

      <StorySection beat="DECIDE" title="Forecast accuracy for procurement">
        <div className="space-y-4 text-slate-500">
          <p>
            Given MAPE of {typeof mape === "number" ? `${mape}%` : "X%"}, is this forecast accurate
            enough for procurement decisions worth EUR 500K/week?
          </p>
          <p>
            This is a decision framework. <strong>Accuracy thresholds</strong> should be set before
            using forecasts for high-stakes decisions. A 5% MAPE on EUR 500K implies ±EUR 25K
            uncertainty per week — acceptable for some teams, unacceptable for others. A 15% MAPE
            implies ±EUR 75K — likely too wide for procurement without additional buffers. Many
            teams set a rule: MAPE &lt; 10% before using forecasts for capital allocation; above
            that, use scenarios or human judgment. The threshold depends on your risk tolerance and
            the cost of being wrong.
          </p>
        </div>
      </StorySection>

      <Assessment questions={ASSESSMENT_QUESTIONS} />
    </ModuleLayout>
  );
}
