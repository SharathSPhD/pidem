"use client";

import { useState, useCallback, useEffect } from "react";
import { ModuleLayout } from "@/components/module-layout";
import { StorySection } from "@/components/story-section";
import { GuidedInsight } from "@/components/guided-insight";
import { ChartPanel } from "@/components/chart-panel";
import { Assessment, type AssessmentQuestion } from "@/components/assessment";
import { apiPost } from "@/lib/api-client";
import type { PlotlyFigure } from "@/components/chart-panel";

interface M05Event {
  date?: string;
  volume_litres?: number;
  anomaly_score?: number;
}

interface M05Response {
  figures?: Record<string, PlotlyFigure>;
  metrics?: Record<string, number | string | null | undefined>;
  data?: { events?: M05Event[] };
}

const ASSESSMENT_QUESTIONS: AssessmentQuestion[] = [
  {
    id: "q1",
    type: "conceptual",
    question: "How does contamination affect sensitivity?",
    modelAnswer:
      "Contamination is the expected proportion of anomalies in the data. Lower values (e.g. 0.01) make the model stricter — fewer points are flagged, so you may miss real competitive attacks. Higher values (e.g. 0.15) flag more points, increasing false alarms.",
  },
  {
    id: "q2",
    type: "applied",
    question: "What do the control chart UCL/LCL represent?",
    modelAnswer:
      "UCL (Upper Control Limit) and LCL (Lower Control Limit) define the expected range of variation. Points outside these limits are statistically unusual and warrant investigation — they may indicate anomalies.",
  },
  {
    id: "q3",
    type: "critical",
    question: "How many false alarms per week can your pricing team tolerate before they start ignoring the system?",
    modelAnswer:
      "This is an operational decision. Too many false positives erode trust; too few detections mean missed competitive threats. Teams often aim for 1–3 actionable alerts per week.",
  },
];

export default function M05AnomalyPage() {
  const [contamination, setContamination] = useState(0.05);
  const [windowSize, setWindowSize] = useState(14);
  const [figures, setFigures] = useState<Record<string, PlotlyFigure | null>>({});
  const [metrics, setMetrics] = useState<Record<string, number | string | null | undefined>>({});
  const [events, setEvents] = useState<M05Event[]>([]);
  const [loading, setLoading] = useState(false);
  const [exploreLoading, setExploreLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const train = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await apiPost<M05Response>("/api/m05/train", {
        contamination,
        window_size: windowSize,
      });
      if (res?.figures) setFigures(res.figures as Record<string, PlotlyFigure | null>);
      if (res?.metrics) setMetrics(res.metrics ?? {});
      if (res?.data?.events) setEvents(res.data.events);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Training failed");
    } finally {
      setLoading(false);
    }
  }, [contamination, windowSize]);

  useEffect(() => {
    let cancelled = false;
    setExploreLoading(true);
    apiPost<M05Response>("/api/m05/train", { contamination: 0.05, window_size: 14 })
      .then((res) => {
        if (!cancelled && res?.figures) {
          setFigures(res.figures as Record<string, PlotlyFigure | null>);
          if (res?.metrics) setMetrics(res.metrics ?? {});
          if (res?.data?.events) setEvents(res.data.events ?? []);
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

  const nAnomalies = metrics?.n_anomalies as number | undefined;

  return (
    <ModuleLayout
      chapterNumber={5}
      title="Anomaly Detection"
      subtitle="Which volume drops are natural and which signal a competitive attack?"
      estimatedMinutes={15}
      learningObjectives={["Isolation Forest", "Control charts", "False positive management"]}
      currentSlug="m05-anomaly"
    >
      <StorySection beat="FRAME" title="Signal vs noise">
        <div className="space-y-4 text-slate-500">
          <p>
            Station STN_001 drops 25% in volume on a Tuesday. Is it a bank holiday? A road closure?
            Or a competitor waging a price war 2km away? Every pricing team faces this daily —
            separating genuine anomalies from routine noise.
          </p>
          <p>
            Anomaly detection isn&apos;t just about fuel volumes. A convenience retailer monitors
            basket size — a sudden 15% drop in average transaction value could signal a
            competitor&apos;s new loyalty program. A supply chain manager watches lead times — a
            3-day spike in delivery time from a key supplier might precede a stockout. In each case,
            the question is the same: is this random variation or something you need to act on?
          </p>
          <p>
            This module uses two complementary approaches. <strong>Isolation Forest</strong> treats
            anomalies as points that are &quot;easy to isolate&quot; — they sit far from the crowd
            in feature space, so a random tree can split them off in few steps. It&apos;s
            model-free, works with multivariate data, and flags outliers without assuming a
            distribution. <strong>Control charts</strong> take a different view: they define an
            expected range (UCL/LCL) from historical variation. Points outside those limits are
            statistically unusual. Isolation Forest answers &quot;which points are weird?&quot;
            Control charts answer &quot;is today&apos;s value within normal bounds?&quot; Both views
            matter — one for exploratory detection, the other for ongoing monitoring.
          </p>
        </div>
      </StorySection>

      <StorySection beat="EXPLORE" title="Time series with anomalies marked">
        <div className="space-y-4">
          <p className="text-slate-500">
            Before tuning parameters, inspect the raw signal. The chart below overlays anomaly markers
            on daily volume — each red point was flagged by Isolation Forest as statistically
            unusual. Notice how some anomalies cluster around known events (holidays, weekends);
            others stand alone and may warrant deeper investigation.
          </p>
          <GuidedInsight type="notice">
            The red markers show detected anomalies. Not all are problems — some are holidays.
          </GuidedInsight>
          <ChartPanel
            figure={figures.primary ?? null}
            title="Volume with anomaly markers"
            loading={exploreLoading}
          />
        </div>
      </StorySection>

      <StorySection beat="BUILD" title="Tune contamination and window size">
        <div className="space-y-4">
          <p className="text-slate-500">
            <strong>Contamination</strong> is the expected proportion of anomalies in your data. Set
            it to 0.01 and the model assumes only 1% of points are anomalous — it becomes strict,
            flagging fewer points. Set it to 0.15 and it assumes 15% are anomalies — more points get
            flagged, including borderline cases. In business terms: low contamination reduces false
            alarms but risks missing real competitive attacks; high contamination catches more
            threats but can overwhelm your team with noise. <strong>Window size</strong> controls
            the rolling window for the control chart — how many days of history define
            &quot;normal.&quot; A 7-day window adapts quickly to recent shifts but is noisy; a 30-day
            window is smoother but slower to react to regime changes.
          </p>
          <div className="flex flex-wrap items-center gap-4">
            <label className="flex flex-col gap-1">
              <span className="text-xs text-slate-500">Contamination (0.01–0.15)</span>
              <input
                type="range"
                min={0.01}
                max={0.15}
                step={0.01}
                value={contamination}
                onChange={(e) => setContamination(Number(e.target.value))}
                className="w-32"
              />
              <span className="text-xs text-slate-500">{contamination}</span>
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-xs text-slate-500">Window size (7–30)</span>
              <input
                type="range"
                min={7}
                max={30}
                value={windowSize}
                onChange={(e) => setWindowSize(Number(e.target.value))}
                className="w-32"
              />
              <span className="text-xs text-slate-500">{windowSize}</span>
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
            figure={figures.primary ?? null}
            title="Control chart"
            loading={loading}
          />
          {typeof nAnomalies === "number" && (
            <p className="text-sm text-slate-500">
              Detected anomalies: <span className="font-mono text-amber-600">{nAnomalies}</span>
            </p>
          )}
        </div>
      </StorySection>

      <StorySection beat="INTERROGATE" title="Contamination tradeoffs">
        <div className="space-y-4">
          <p className="text-slate-500">
            The table below lists detected anomalies with their scores. Higher scores mean the point
            was easier to isolate — more likely a true outlier. Compare dates across contamination
            settings: at 0.01 you&apos;ll see only the most extreme events; at 0.15 you&apos;ll see
            borderline cases too. The tradeoff is operational: how many investigations can your team
            realistically run per week?
          </p>
          <GuidedInsight type="warning">
            Lower contamination = fewer detected anomalies. At 0.01, you&apos;ll miss real
            competitive attacks. At 0.15, you&apos;ll drown in false alarms.
          </GuidedInsight>
          <GuidedInsight type="think">
            Operational tolerance varies by domain. A fuel retailer might accept 2–3 alerts per week
            — each triggers a site visit or price check. A convenience chain monitoring 500 stores
            might need stricter thresholds to avoid alert fatigue. A supply chain manager watching
            lead times might tolerate more false positives because missing a supplier disruption is
            costlier than investigating a benign spike.
          </GuidedInsight>
          {events.length > 0 && (
            <div className="overflow-x-auto rounded-lg border border-slate-200">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-slate-200 bg-slate-50">
                    <th className="px-4 py-2 text-left text-slate-500">Date</th>
                    <th className="px-4 py-2 text-left text-slate-500">Volume (L)</th>
                    <th className="px-4 py-2 text-left text-slate-500">Anomaly score</th>
                  </tr>
                </thead>
                <tbody>
                  {events.map((ev, i) => (
                    <tr key={i} className="border-b border-slate-200">
                      <td className="px-4 py-2 text-slate-600">{ev.date ?? "—"}</td>
                      <td className="px-4 py-2 text-slate-600">{ev.volume_litres ?? "—"}</td>
                      <td className="px-4 py-2 text-slate-600">{ev.anomaly_score ?? "—"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </StorySection>

      <StorySection beat="DECIDE" title="Operational tolerance">
        <div className="space-y-4 text-slate-500">
          <p>
            How many false alarms per week can your pricing team tolerate before they start
            ignoring the system?
          </p>
          <p>
            This is a decision framework, not a technical one. <strong>Alert fatigue</strong> is
            real: when 90% of alerts are false positives, teams stop trusting the system and miss
            the one that mattered. Conversely, <strong>missed threats</strong> are costly — a
            competitor&apos;s price war that goes undetected for two weeks can cost thousands in
            lost volume. The right balance depends on your capacity to investigate, the cost of a
            missed detection, and the cost of a false alarm. Many teams aim for 1–3 actionable
            alerts per week — enough to stay vigilant, few enough to act on each one.
          </p>
        </div>
      </StorySection>

      <Assessment questions={ASSESSMENT_QUESTIONS} />
    </ModuleLayout>
  );
}
