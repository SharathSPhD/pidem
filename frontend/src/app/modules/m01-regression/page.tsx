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
  data?: {
    coefficients?: Array<{ name: string; value: number; ci_95_low?: number; ci_95_high?: number }>;
    elasticity_price_gap?: number;
    vif?: Array<{ feature: string; vif?: number }>;
  };
}

const ASSESSMENT_QUESTIONS: AssessmentQuestion[] = [
  {
    id: "q1",
    type: "conceptual",
    question: "What does a VIF > 10 indicate?",
    options: [
      "The feature is highly predictive",
      "Features are collinear — the elasticity estimate becomes unreliable",
      "The model is overfitting",
      "The model needs more data",
    ],
    modelAnswer:
      "VIF > 10 indicates multicollinearity. When features are highly correlated, the coefficient estimates become unstable and the elasticity confidence interval widens.",
  },
  {
    id: "q2",
    type: "applied",
    question: "The elasticity CI is [-1.8, -1.2]. If margin is 0.04 EUR/L, should you approve the 2ct price cut?",
    options: [
      "Yes — elasticity suggests volume gain outweighs margin loss",
      "No — the margin loss is too high",
      "Need more data — the CI is too wide",
      "Depends on the volume elasticity at current price",
    ],
    modelAnswer:
      "With elasticity around -1.5, a 2ct cut (e.g. 1.50→1.48) is ~1.3% price drop. Volume would rise ~2% (1.5 × 1.3). At 0.04 EUR/L margin, the volume gain must offset the margin loss — run the numbers for your specific scenario.",
  },
  {
    id: "q3",
    type: "critical",
    question: "When might you prefer a simpler elasticity model over one with more features?",
    options: [
      "When stakeholders need a single number to explain",
      "When VIF values are high",
      "When the extra features don't improve out-of-sample R²",
      "All of the above",
    ],
    modelAnswer:
      "All of the above. Simplicity aids communication; high VIF suggests collinearity; and if extra features don't improve generalization, they add noise.",
  },
];

const STATION_TYPES = [
  { value: "all", label: "All" },
  { value: "motorway", label: "Motorway" },
  { value: "urban", label: "Urban" },
  { value: "rural", label: "Rural" },
];

export default function M01RegressionPage() {
  const [stationType, setStationType] = useState("all");
  const [regularization, setRegularization] = useState(1);
  const [loading, setLoading] = useState(false);
  const [exploreFigure, setExploreFigure] = useState<PlotlyFigure | null>(null);
  const [buildFigure, setBuildFigure] = useState<PlotlyFigure | null>(null);
  const [vifFigure, setVifFigure] = useState<PlotlyFigure | null>(null);
  const [residualFigure, setResidualFigure] = useState<PlotlyFigure | null>(null);
  const [elasticity, setElasticity] = useState<number | null>(null);
  const [vif, setVif] = useState<Array<{ feature: string; vif?: number }>>([]);

  const train = useCallback(async () => {
    setLoading(true);
    setBuildFigure(null);
    setVifFigure(null);
    setResidualFigure(null);
    setElasticity(null);
    setVif([]);
    try {
      const res = await apiPost<TrainResponse>("/api/m01/train", {
        station_type: stationType,
        regularization,
      });
      const figs = res?.figures ?? {};
      if (figs.primary) setBuildFigure(figs.primary as PlotlyFigure);
      if (figs.vif) setVifFigure(figs.vif as PlotlyFigure);
      if (figs.residual) setResidualFigure(figs.residual as PlotlyFigure);
      if (res?.data?.elasticity_price_gap != null)
        setElasticity(res.data.elasticity_price_gap);
      if (res?.data?.vif) setVif(res.data.vif);
    } catch {
      // Error handled by UI
    } finally {
      setLoading(false);
    }
  }, [stationType, regularization]);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const res = await apiPost<TrainResponse>("/api/m01/train", {
          station_type: "all",
          regularization: 1,
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
      chapterNumber={1}
      title="Price Elasticity"
      subtitle="When does cutting your price actually pay off in volume?"
      estimatedMinutes={25}
      learningObjectives={[
        "Log-log demand models",
        "Elasticity interpretation",
        "Feature diagnostics (VIF, Cook's D)",
      ]}
      currentSlug="m01-regression"
    >
      <StorySection beat="FRAME" title="The EUR 2M question">
        <div className="space-y-4 text-slate-500">
          <p>
            Your VP asks: &quot;If we cut diesel 2 cents, how much volume do we
            gain?&quot; This isn&apos;t an academic exercise — the answer drives
            a EUR 2M pricing decision across 200 locations. Get the elasticity
            wrong by even 0.3, and you either leave money on the table or
            trigger a price war you can&apos;t win.
          </p>
          <p>
            Elasticity isn&apos;t unique to fuel. A convenience retailer asks
            the same question about coffee: &quot;If we raise the price 20p, how
            many fewer cups do we sell?&quot; A supply chain manager wonders:
            &quot;If we increase minimum order quantities, do customers
            consolidate orders or walk away?&quot; The math is identical — a
            log-log regression that quantifies the percentage change in demand
            for a 1% change in price.
          </p>
          <p>
            What does elasticity really mean? It&apos;s the sensitivity of
            demand to price. An elasticity of -1.5 says: for every 1% you cut
            price, volume rises by about 1.5%. That&apos;s the headline number.
            But the confidence interval matters more for decision-making: a
            narrow CI [-1.4, -1.6] tells you the estimate is trustworthy; a
            wide CI [-2.2, -0.8] means you&apos;re guessing. Before you approve
            a EUR 2M move, you need to know which world you&apos;re in.
          </p>
          <p>
            This module teaches you to build log-log demand models, interpret
            elasticity, and diagnose when collinearity makes your estimates
            unreliable.
          </p>
        </div>
      </StorySection>

      <StorySection beat="EXPLORE" title="Price-gap vs volume">
        <div className="space-y-4">
          <p className="text-slate-500">
            The elasticity coefficients below show how strongly each feature
            drives volume. The price-gap coefficient is the one that answers
            your VP&apos;s question: it tells you the percentage change in
            demand for a 1% change in the price gap vs competitors. A more
            negative value means customers are more price-sensitive — they
            switch stations when you&apos;re expensive. Why does this matter?
            Because the magnitude of that coefficient directly feeds your
            revenue model: get it right, and you optimize; get it wrong, and you
            bleed margin or lose share.
          </p>
          <ChartPanel
            figure={exploreFigure}
            title="Elasticity coefficients"
            loading={!exploreFigure && !loading}
          />
          <GuidedInsight type="notice">
            Motorway stations show flatter elasticity — captive demand. Urban
            stations respond more to price gaps.
          </GuidedInsight>
        </div>
      </StorySection>

      <StorySection beat="BUILD" title="Train the elasticity model">
        <div className="space-y-4">
          <p className="text-slate-500">
            Station type filters let you see if elasticity differs by location
            type — captive motorway customers respond very differently to price
            changes than urban commuters with 5 alternatives within 2km.
            Regularization controls how much the model penalizes large
            coefficients — think of it as a skepticism dial: turn it up, and the
            model prefers smaller, more conservative estimates.
          </p>
          <div className="flex flex-wrap gap-6">
            <label className="flex flex-col gap-1">
              <span className="text-xs text-slate-500">Station type</span>
              <select
                value={stationType}
                onChange={(e) => setStationType(e.target.value)}
                className="rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm text-slate-600"
              >
                {STATION_TYPES.map((t) => (
                  <option key={t.value} value={t.value}>
                    {t.label}
                  </option>
                ))}
              </select>
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-xs text-slate-500">Regularization (0.01–10)</span>
              <input
                type="range"
                min={0.01}
                max={10}
                step={0.01}
                value={regularization}
                onChange={(e) => setRegularization(Number(e.target.value))}
                className="w-40"
              />
              <span className="text-xs text-slate-500">{regularization.toFixed(2)}</span>
            </label>
            <button
              onClick={train}
              disabled={loading}
              className="mt-2 rounded-lg bg-amber-500 px-4 py-2 text-sm font-medium text-slate-900 hover:bg-amber-400 disabled:opacity-50"
            >
              Train
            </button>
          </div>
          {elasticity != null && (
            <div className="rounded-lg border border-slate-200 bg-slate-50 px-4 py-2">
              <span className="text-xs text-slate-500">Price-gap elasticity</span>
              <p className="text-lg font-semibold text-emerald-400">
                {elasticity.toFixed(3)}
              </p>
            </div>
          )}
          <ChartPanel
            figure={buildFigure}
            title="Coefficients"
            loading={loading}
          />
          <GuidedInsight type="try">
            Set regularization to 0.01. Watch VIF values spike — collinearity
            becomes visible.
          </GuidedInsight>
        </div>
      </StorySection>

      <StorySection beat="INTERROGATE" title="VIF and residual diagnostics">
        <div className="space-y-4">
          <p className="text-slate-500">
            VIF measures whether your features are telling redundant stories.
            When two predictors (e.g. price gap and competitor count) move
            together, the model can&apos;t cleanly separate their effects — the
            elasticity estimate becomes unstable. The same logic applies in other
            domains: a retailer modeling coffee demand might see VIF spike if
            &quot;store size&quot; and &quot;foot traffic&quot; are highly
            correlated; a supply chain model might suffer if &quot;lead time&quot;
            and &quot;inventory level&quot; are collinear. Diagnostics are
            universal — the numbers change, the principles don&apos;t.
          </p>
          <div className="grid gap-4 sm:grid-cols-2">
            <ChartPanel figure={vifFigure} title="VIF" />
            <ChartPanel figure={residualFigure} title="Residual plot" />
          </div>
          {vif.length > 0 && (
            <div className="rounded-lg border border-slate-200 bg-slate-50 p-4">
              <p className="text-xs text-slate-500">VIF values</p>
              <div className="mt-2 flex flex-wrap gap-2">
                {vif.map((v) => (
                  <span
                    key={v.feature}
                    className={`rounded px-2 py-1 text-xs ${
                      (v.vif ?? 0) > 10
                        ? "bg-amber-500/20 text-amber-600"
                        : "bg-slate-100 text-slate-500"
                    }`}
                  >
                    {v.feature}: {v.vif?.toFixed(1) ?? "—"}
                  </span>
                ))}
              </div>
            </div>
          )}
          <GuidedInsight type="warning">
            If VIF &gt; 10, features are collinear. The elasticity estimate
            becomes unreliable.
          </GuidedInsight>
          <GuidedInsight type="think">
            What are the implications? If your elasticity CI is wide and VIF is
            high, you&apos;re not ready to act. Either collect more data, drop
            redundant features, or segment by station type to reduce
            collinearity — but don&apos;t bet EUR 2M on a shaky estimate.
          </GuidedInsight>
        </div>
      </StorySection>

      <StorySection beat="DECIDE" title="Approve the 2ct price cut?">
        <div className="space-y-4">
          <p className="text-slate-500">
            The elasticity CI is [-1.8, -1.2]. If margin is 0.04 EUR/L, should
            you approve the 2ct price cut? Use the model and diagnostics to
            decide.
          </p>
          <p className="text-slate-500">
            When are elasticity estimates actionable? When the CI is narrow,
            VIF is low, and the residual plot shows no systematic bias. That&apos;s
            when you can plug the midpoint into your revenue model and make a
            call. When do you need more data? When the CI spans both &quot;cut
            price&quot; and &quot;hold price&quot; territory, when VIF &gt; 10, or
            when you&apos;ve never validated the model on a holdout period.
            Deployment matters too: elasticity drifts over time as competitors
            react, so plan to retrain quarterly and monitor out-of-sample
            performance.
          </p>
          <Assessment questions={ASSESSMENT_QUESTIONS} />
        </div>
      </StorySection>
    </ModuleLayout>
  );
}
