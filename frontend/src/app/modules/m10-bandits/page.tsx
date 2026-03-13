"use client";

import { useState, useCallback, useEffect } from "react";
import { ModuleLayout } from "@/components/module-layout";
import { StorySection } from "@/components/story-section";
import { GuidedInsight } from "@/components/guided-insight";
import { ChartPanel } from "@/components/chart-panel";
import { Assessment, type AssessmentQuestion } from "@/components/assessment";
import { apiPost } from "@/lib/api-client";
import type { PlotlyFigure } from "@/components/chart-panel";

interface M10Response {
  figures?: Record<string, PlotlyFigure>;
  metrics?: { total_regret?: number; algorithm?: string };
}

const ALGORITHMS = [
  { value: "thompson_sampling", label: "Thompson Sampling" },
  { value: "ucb1", label: "UCB1" },
  { value: "epsilon_greedy", label: "ε-Greedy" },
];

const ASSESSMENT: AssessmentQuestion[] = [
  {
    id: "q1",
    type: "conceptual",
    question: "What is the explore-exploit tradeoff in multi-armed bandits?",
    modelAnswer:
      "Exploration tries new arms to learn their rewards; exploitation picks the arm with the best known reward. Too much exploration wastes opportunities; too little locks you into a suboptimal arm.",
  },
  {
    id: "q2",
    type: "applied",
    question: "You have 168 hours (1 week) to run a pricing experiment. Which algorithm minimizes regret?",
    options: ["ε-Greedy", "UCB1", "Thompson Sampling", "Random"],
    modelAnswer: "Thompson Sampling typically minimizes regret because it naturally balances exploration and exploitation through posterior sampling, concentrating trials on promising arms faster than UCB1 or ε-Greedy.",
  },
  {
    id: "q3",
    type: "critical",
    question: "When does non-stationarity break Thompson Sampling, and what can you do about it?",
    modelAnswer:
      "When the best price level shifts over time (e.g., weekday vs weekend), Thompson Sampling's posterior becomes stale. FDSW-Thompson uses a sliding window to discount old observations, adapting to shifts.",
  },
];

export default function M10BanditsPage() {
  const [algorithm, setAlgorithm] = useState("thompson_sampling");
  const [horizon, setHorizon] = useState(500);
  const [nonStationary, setNonStationary] = useState(false);
  const [loading, setLoading] = useState(false);
  const [figures, setFigures] = useState<Record<string, PlotlyFigure>>({});
  const [metrics, setMetrics] = useState<M10Response["metrics"]>({});

  const run = useCallback(async (alg?: string, h?: number, ns?: boolean) => {
    setLoading(true);
    try {
      const res = await apiPost<M10Response>("/api/m10/run", {
        algorithm: alg ?? algorithm,
        horizon: h ?? horizon,
        non_stationary: ns ?? nonStationary,
      });
      if (res?.figures) setFigures(res.figures);
      if (res?.metrics) setMetrics(res.metrics);
    } catch { /* handled by UI */ }
    setLoading(false);
  }, [algorithm, horizon, nonStationary]);

  useEffect(() => { run("thompson_sampling", 500, false); }, []);

  return (
    <ModuleLayout
      chapterNumber={10}
      title="Price Experimentation"
      subtitle="Should you exploit what works or explore what might work better?"
      estimatedMinutes={20}
      learningObjectives={["Explore-exploit tradeoff", "Thompson Sampling", "Cumulative regret"]}
    >
      <StorySection beat="FRAME" title="The experimentation dilemma">
        <div className="space-y-3 text-slate-600">
          <p>
            Your LP model (M09) gives optimal prices — but it assumes you know the exact demand elasticity.
            What if your elasticity estimate is wrong by 20%? You'd set the wrong price and never know.
          </p>
          <p>
            Multi-armed bandits learn the true elasticity by experimenting with price levels in real time.
            Each hour, the bandit picks one of five arms: −4ct, −2ct, parity, +2ct, +4ct. It observes
            the resulting revenue and updates its beliefs about which arm is best.
          </p>
          <p>
            The challenge: every hour spent exploring a suboptimal arm costs real revenue. This is
            <strong className="text-amber-600"> regret</strong> — the gap between what you earned and
            what you would have earned by always picking the best arm.
          </p>
          <p>
            Whether you&apos;re experimenting with coffee prices at a convenience chain, testing
            promotional discounts on an e-commerce platform, or varying contractual terms with
            industrial customers, the explore-exploit dilemma is universal. How much revenue are you
            willing to sacrifice on experiments to learn something that could generate much more
            revenue in the future?
          </p>
        </div>
      </StorySection>

      <StorySection beat="EXPLORE" title="How fast does the bandit learn?">
        <div className="space-y-4">
          <ChartPanel figure={figures.primary ?? null} title="Cumulative Regret" loading={loading} />
          <GuidedInsight type="notice">
            <p>
              Regret flattens as the bandit learns which arm is best. A flat curve means
              the algorithm found the optimal price quickly. A steep curve means it wasted
              experiments on suboptimal arms.
            </p>
          </GuidedInsight>
          <GuidedInsight type="think">
            <p>
              So what? The speed at which regret flattens determines how quickly you can stop
              experimenting and start earning. A pricing manager who waits for statistical
              significance before committing to a price may never act — bandits give you a
              principled way to learn and earn simultaneously.
            </p>
          </GuidedInsight>
        </div>
      </StorySection>

      <StorySection beat="BUILD" title="Compare algorithms">
        <div className="space-y-4">
          <div className="flex flex-wrap gap-4 rounded-lg border border-slate-200 bg-slate-50 p-4">
            <label className="flex flex-col gap-1">
              <span className="text-xs text-slate-500">Algorithm</span>
              <select
                value={algorithm}
                onChange={(e) => setAlgorithm(e.target.value)}
                className="rounded border border-slate-200 bg-white px-3 py-1.5 text-sm text-slate-700"
              >
                {ALGORITHMS.map((a) => (
                  <option key={a.value} value={a.value}>{a.label}</option>
                ))}
              </select>
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-xs text-slate-500">Horizon (hours)</span>
              <input
                type="range" min={100} max={1000} step={50}
                value={horizon}
                onChange={(e) => setHorizon(Number(e.target.value))}
                className="w-40"
              />
              <span className="text-xs text-slate-500">{horizon}</span>
            </label>
            <label className="flex items-center gap-2">
              <input
                type="checkbox" checked={nonStationary}
                onChange={(e) => setNonStationary(e.target.checked)}
                className="rounded border-slate-200"
              />
              <span className="text-sm text-slate-600">Non-stationary</span>
            </label>
            <button
              onClick={() => run()}
              disabled={loading}
              className="rounded-lg bg-amber-500 px-4 py-2 text-sm font-medium text-slate-900 hover:bg-amber-400 disabled:opacity-50"
            >
              Run
            </button>
          </div>

          <ChartPanel figure={figures.primary ?? null} title="Cumulative Regret" loading={loading} />

          {metrics?.total_regret != null && (
            <p className="text-sm text-slate-500">
              Total regret: <span className="font-mono text-amber-600">{metrics.total_regret}</span>
              {metrics.algorithm && <span> ({metrics.algorithm})</span>}
            </p>
          )}

          <GuidedInsight type="try">
            <p>
              Switch to ε-Greedy. It explores randomly, wasting experiments on poor arms.
              Thompson Sampling concentrates on promising arms much faster.
            </p>
          </GuidedInsight>
          <div className="space-y-2 text-slate-600 text-sm">
            <p className="font-medium text-slate-800">What each algorithm means for pricing:</p>
            <p>
              <strong>Thompson Sampling</strong> — Samples from a probability distribution over
              which price is best. Naturally favors arms that look promising while still trying
              underdogs. For pricing managers: &quot;We&apos;re more likely to try the price that
              seems best, but we&apos;ll occasionally test others to avoid getting stuck.&quot;
            </p>
            <p>
              <strong>UCB1</strong> — Picks the arm with the highest upper confidence bound. Explicitly
              balances known reward vs uncertainty. For pricing managers: &quot;We favor prices we
              know work, but we give extra weight to prices we haven&apos;t tried enough.&quot;
            </p>
            <p>
              <strong>ε-Greedy</strong> — With probability ε, explore randomly; otherwise exploit the
              best known arm. Simple but blunt. For pricing managers: &quot;Most of the time we use
              our best guess; occasionally we try something random.&quot; Easiest to explain, but
              wastes more experiments than Thompson or UCB1.
            </p>
          </div>
        </div>
      </StorySection>

      <StorySection beat="INTERROGATE" title="What breaks in the real world?">
        <div className="space-y-4">
          <ChartPanel figure={figures.heatmap ?? null} title="Arm Selection Over Time" loading={loading} />
          <GuidedInsight type="warning">
            <p>
              Enable non-stationary mode. The best arm shifts mid-simulation —
              only algorithms with forgetting mechanisms (like FDSW-Thompson) adapt.
              Standard Thompson Sampling keeps exploiting an arm that's no longer optimal.
            </p>
          </GuidedInsight>
          <GuidedInsight type="think">
            <p>
              In pricing, demand elasticity shifts with seasons, competitor behavior, and
              macro events. A bandit deployed on Monday might face different dynamics by Friday.
              How would you detect and adapt to non-stationarity?
            </p>
          </GuidedInsight>
          <p className="text-slate-600">
            The same logic applies beyond fuel retail. A/B tests on landing pages, promotional
            campaigns, or subscription tiers all face non-stationarity: user preferences change,
            competitors react, and external events shift demand. A bandit that forgets old data
            (like FDSW-Thompson) adapts; one that doesn&apos;t keeps exploiting a strategy that
            no longer works.
          </p>
          <GuidedInsight type="think">
            <p>
              Business implication: if your environment is stable (e.g., a mature product with
              predictable demand), standard bandits suffice. If it shifts (seasonal, competitive,
              or volatile markets), you need forgetting mechanisms — or accept that your bandit
              will degrade over time.
            </p>
          </GuidedInsight>
        </div>
      </StorySection>

      <StorySection beat="DECIDE" title="Choosing your experiment design">
        <div className="space-y-4">
          <div className="rounded-lg border border-slate-200 bg-slate-50 p-5 text-slate-600 space-y-3">
            <p>
              You have 168 hours (1 week) to run a pricing experiment at 5 stations.
              Each hour of exploration at a suboptimal price costs roughly EUR 15 in
              lost margin.
            </p>
            <p className="font-medium text-slate-900">
              Which algorithm do you recommend, and what&apos;s the maximum acceptable regret?
            </p>
            <p>
              <strong>When bandits are practical:</strong> You have a small, discrete set of
              options (e.g., 5 price levels). Each decision is independent — today&apos;s price
              doesn&apos;t meaningfully change tomorrow&apos;s state. You can run experiments in
              parallel (e.g., different stations). Bandits are simpler to deploy and interpret than
              full reinforcement learning.
            </p>
            <p>
              <strong>When full RL is needed:</strong> Today&apos;s price affects tomorrow&apos;s
              demand, competitor response, or inventory. The state space is large or continuous.
              You need a policy that maps many states to actions — not just &quot;which arm is
              best.&quot; In those cases, Q-Learning (M11) or DQN (M12) are the right tools.
            </p>
          </div>
          <Assessment questions={ASSESSMENT} />
        </div>
      </StorySection>
    </ModuleLayout>
  );
}
