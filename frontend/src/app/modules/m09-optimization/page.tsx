"use client";

import { useState, useCallback, useEffect } from "react";
import { ModuleLayout } from "@/components/module-layout";
import { StorySection } from "@/components/story-section";
import { GuidedInsight } from "@/components/guided-insight";
import { ChartPanel } from "@/components/chart-panel";
import { Assessment, type AssessmentQuestion } from "@/components/assessment";
import { apiPost } from "@/lib/api-client";
import type { PlotlyFigure } from "@/components/chart-panel";

interface M09Response {
  figures?: Record<string, PlotlyFigure>;
  charts?: Record<string, PlotlyFigure>;
  metrics?: { total_margin?: number; total_volume?: number; status?: string };
}

const ASSESSMENT_QUESTIONS: AssessmentQuestion[] = [
  {
    id: "q1",
    type: "critical",
    question:
      "The optimizer says Station 42 should price at EUR 1.72 when competitors are at 1.70. Would you override the model? Under what circumstances?",
    modelAnswer:
      "Consider overriding when: brand positioning forbids premium pricing, local regulations cap prices, or you have qualitative intelligence (e.g., competitor about to cut). The model optimizes margin; business policy may prioritize volume or market share.",
  },
];

export default function M09OptimizationPage() {
  const [priceMin, setPriceMin] = useState(1.4);
  const [priceMax, setPriceMax] = useState(1.8);
  const [volumeFloor, setVolumeFloor] = useState(0.8);
  const [solverMode, setSolverMode] = useState<"lp" | "nlp">("lp");
  const [figures, setFigures] = useState<Record<string, PlotlyFigure | null>>({});
  const [metrics, setMetrics] = useState<M09Response["metrics"]>({});
  const [loading, setLoading] = useState(false);
  const [exploreLoading, setExploreLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const solve = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await apiPost<M09Response>("/api/m09/solve", {
        price_band: [priceMin, priceMax],
        volume_floor: volumeFloor,
        solver_mode: solverMode,
      });
      const figs = res?.figures ?? res?.charts ?? {};
      setFigures(figs as Record<string, PlotlyFigure | null>);
      if (res?.metrics) setMetrics(res.metrics);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Solve failed");
    } finally {
      setLoading(false);
    }
  }, [priceMin, priceMax, volumeFloor, solverMode]);

  useEffect(() => {
    let cancelled = false;
    setExploreLoading(true);
    apiPost<M09Response>("/api/m09/solve", {
      price_band: [1.4, 1.8],
      volume_floor: 0.8,
      solver_mode: "lp",
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

  const primaryFigure = figures.primary ?? figures.price_map ?? null;

  return (
    <ModuleLayout
      chapterNumber={9}
      title="Price Optimization"
      subtitle="Given constraints, what's the mathematically optimal price for each station?"
      estimatedMinutes={20}
      learningObjectives={["Linear programming", "Constraint formulation", "Shadow prices"]}
      currentSlug="m09-optimization"
    >
      <StorySection beat="FRAME" title="The optimization challenge">
        <div className="space-y-4 text-slate-500">
          <p>
            You have 100 locations, each with a different competitive landscape, cost base, and
            volume sensitivity. Setting prices manually — or with simple rules like &quot;match the
            competitor minus 1ct&quot; — leaves money on the table. Linear programming finds the
            margin-maximizing price for every location simultaneously, subject to constraints you
            define.
          </p>
          <p>
            Price optimization is used across retail, logistics, and B2B. A supermarket optimizes
            thousands of SKU prices weekly. An airline sets ticket prices for hundreds of routes. A
            chemical supplier optimizes contract pricing across customer tiers. The mathematical
            framework — objective function plus constraints — is identical.
          </p>
          <p>
            The <strong>objective function</strong> is what you maximize (e.g. total margin). The
            <strong> constraints</strong> are the rules: price within a band, volume above a floor,
            stay near competitors. <strong>Shadow prices</strong> tell you the marginal value of
            relaxing each constraint — how much extra margin you&apos;d gain if you allowed one more
            unit. <strong>Feasibility</strong> means a solution exists that satisfies all
            constraints; if not, you must relax some.
          </p>
        </div>
      </StorySection>

      <StorySection beat="EXPLORE" title="Price recommendations">
        <div className="space-y-4">
          <p className="text-slate-500">
            The chart below shows optimal prices by station. Each bar is the price that maximizes
            total margin across the network, given the constraints you set. Notice how motorway and
            urban stations get different recommendations — the optimizer exploits differences in
            demand elasticity and competition.
          </p>
          <ChartPanel
            figure={primaryFigure}
            title="Optimal price by station"
            loading={exploreLoading}
          />
          <GuidedInsight type="notice">
            Each bar shows the optimal price. Notice that motorway stations get higher prices — they
            have captive demand.
          </GuidedInsight>
        </div>
      </StorySection>

      <StorySection beat="BUILD" title="Run the optimizer">
        <div className="space-y-4">
          <p className="text-slate-500">
            <strong>Price band</strong> bounds the allowed price range — e.g. 1.40–1.80 EUR/L. This
            encodes policy (don&apos;t go below cost, don&apos;t exceed a cap) or competitive
            constraints. <strong>Volume floor</strong> (0.5–1.0) limits how much volume you&apos;re
            willing to sacrifice: 1.0 means no drop from baseline; 0.8 allows up to 20% volume loss
            for margin gain. <strong>Solver mode</strong>: LP assumes linear demand curves; NLP
            handles nonlinear elasticity for more realistic responses.
          </p>
          <div className="flex flex-wrap items-center gap-6">
            <label className="flex flex-col gap-1">
              <span className="text-xs text-slate-500">Price band (EUR/L)</span>
              <div className="flex gap-2">
                <input
                  type="number"
                  step={0.05}
                  min={1.2}
                  max={2.0}
                  value={priceMin}
                  onChange={(e) => setPriceMin(Number(e.target.value))}
                  className="w-20 rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm text-slate-600"
                />
                <input
                  type="number"
                  step={0.05}
                  min={1.2}
                  max={2.0}
                  value={priceMax}
                  onChange={(e) => setPriceMax(Number(e.target.value))}
                  className="w-20 rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm text-slate-600"
                />
              </div>
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-xs text-slate-500">Volume floor (0.5–1.0)</span>
              <input
                type="range"
                min={0.5}
                max={1.0}
                step={0.05}
                value={volumeFloor}
                onChange={(e) => setVolumeFloor(Number(e.target.value))}
                className="w-32"
              />
              <span className="text-xs text-slate-500">{volumeFloor}</span>
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-xs text-slate-500">Solver mode</span>
              <select
                value={solverMode}
                onChange={(e) => setSolverMode(e.target.value as "lp" | "nlp")}
                className="rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm text-slate-600"
              >
                <option value="lp">LP</option>
                <option value="nlp">NLP</option>
              </select>
            </label>
            <button
              onClick={solve}
              disabled={loading}
              className="rounded-lg bg-amber-500 px-4 py-2 text-sm font-medium text-slate-900 hover:bg-amber-400 disabled:opacity-50"
            >
              {loading ? "Solving…" : "Solve"}
            </button>
          </div>
          {error && <p className="text-sm text-rose-400">{error}</p>}
          {(metrics?.total_margin != null || metrics?.total_volume != null) && (
            <div className="flex gap-4">
              {metrics.total_margin != null && (
                <div className="rounded-lg border border-slate-200 bg-slate-50 px-4 py-2">
                  <span className="text-xs text-slate-500">Total margin</span>
                  <p className="text-lg font-semibold text-emerald-400">
                    EUR {metrics.total_margin.toFixed(2)}
                  </p>
                </div>
              )}
              {metrics.total_volume != null && (
                <div className="rounded-lg border border-slate-200 bg-slate-50 px-4 py-2">
                  <span className="text-xs text-slate-500">Total volume</span>
                  <p className="text-lg font-semibold text-emerald-400">
                    {metrics.total_volume.toFixed(0)} L
                  </p>
                </div>
              )}
            </div>
          )}
          <div className="grid gap-4 sm:grid-cols-2">
            <ChartPanel
              figure={figures.frontier ?? figures.margin_frontier ?? null}
              title="Margin frontier"
              loading={loading}
            />
            <ChartPanel
              figure={figures.shadow_prices ?? null}
              title="Shadow prices"
              loading={loading}
            />
          </div>
        </div>
      </StorySection>

      <StorySection beat="INTERROGATE" title="Interpreting shadow prices">
        <div className="space-y-4">
          <p className="text-slate-500">
            Shadow prices tell you the marginal value of relaxing each constraint by one unit. A
            high shadow price on the volume floor means volume preservation is expensive — you&apos;re
            giving up significant margin to hold volume. The same logic applies in other domains: a
            logistics optimizer might show a high shadow price on a capacity constraint, meaning
            adding one truck would yield substantial value.
          </p>
          <ChartPanel figure={figures.shadow_prices ?? null} title="Shadow prices" />
          <GuidedInsight type="notice">
            Shadow prices tell you the EUR value of relaxing each constraint by one unit. A high
            shadow price on the volume floor means volume preservation is expensive.
          </GuidedInsight>
          <GuidedInsight type="try">
            Loosen the volume floor to 0.5. Total margin jumps — you&apos;re sacrificing volume for
            profit.
          </GuidedInsight>
          <GuidedInsight type="think">
            If the shadow price on the volume floor is EUR 500, what does that mean? Relaxing the
            floor by 1% (e.g. from 0.80 to 0.81) would increase total margin by roughly EUR 5. Use
            shadow prices to prioritize which constraints to revisit in negotiations or policy.
          </GuidedInsight>
        </div>
      </StorySection>

      <StorySection beat="DECIDE" title="Override the model?">
        <div className="space-y-4">
          <div className="space-y-4 text-slate-500">
            <p>
              The optimizer says Station 42 should price at EUR 1.72 when competitors are at 1.70.
              Would you override the model? Real-world override scenarios: <strong>Brand
              positioning</strong> — you never want to be the premium outlier in a price-sensitive
              market. <strong>Regulatory caps</strong> — local rules may limit prices. <strong>Qualitative
              intelligence</strong> — a competitor is about to cut; the model doesn&apos;t know. <strong>Volume
              targets</strong> — the model optimizes margin; you may prioritize market share.
            </p>
            <p>
              The model optimizes what you tell it to. Overrides are valid when business policy,
              regulation, or information the model lacks should take precedence. Document overrides
              and feed them back to refine constraints for next time.
            </p>
          </div>
          <Assessment questions={ASSESSMENT_QUESTIONS} />
        </div>
      </StorySection>
    </ModuleLayout>
  );
}
