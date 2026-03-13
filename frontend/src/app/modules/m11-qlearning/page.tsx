"use client";

import { useState, useCallback, useEffect } from "react";
import { ModuleLayout } from "@/components/module-layout";
import { StorySection } from "@/components/story-section";
import { GuidedInsight } from "@/components/guided-insight";
import { ChartPanel } from "@/components/chart-panel";
import { Assessment, type AssessmentQuestion } from "@/components/assessment";
import { apiPost } from "@/lib/api-client";
import type { PlotlyFigure } from "@/components/chart-panel";

interface M11Response {
  figures?: Record<string, PlotlyFigure>;
  charts?: Record<string, PlotlyFigure>;
  metrics?: { final_avg_reward?: number; episodes?: number };
}

const ASSESSMENT_QUESTIONS: AssessmentQuestion[] = [
  {
    id: "q1",
    type: "critical",
    question:
      "The Q-Learning policy says to undercut competitors by 2ct in state (low_volume, high_competition). But your brand positioning says 'never undercut.' How do you reconcile model recommendations with business policy?",
    modelAnswer:
      "Options: (1) Add a hard constraint to the MDP that forbids undercutting. (2) Post-process the policy to map undercut actions to parity. (3) Use a penalty term in the reward for undercutting. (4) Accept the tension and use the model as input to human decision-making rather than automation.",
  },
];

export default function M11QLearningPage() {
  const [gamma, setGamma] = useState(0.9);
  const [episodes, setEpisodes] = useState(1000);
  const [epsilonDecay, setEpsilonDecay] = useState(0.995);
  const [competitorModel, setCompetitorModel] = useState<"static" | "reactive">("static");
  const [figures, setFigures] = useState<Record<string, PlotlyFigure | null>>({});
  const [metrics, setMetrics] = useState<M11Response["metrics"]>({});
  const [loading, setLoading] = useState(false);
  const [exploreLoading, setExploreLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const train = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await apiPost<M11Response>("/api/m11/train", {
        gamma,
        episodes,
        epsilon_decay: epsilonDecay,
        competitor_model: competitorModel,
      });
      const figs = res?.figures ?? res?.charts ?? {};
      setFigures(figs as Record<string, PlotlyFigure | null>);
      if (res?.metrics) setMetrics(res.metrics);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Training failed");
    } finally {
      setLoading(false);
    }
  }, [gamma, episodes, epsilonDecay, competitorModel]);

  useEffect(() => {
    let cancelled = false;
    setExploreLoading(true);
    apiPost<M11Response>("/api/m11/train", {
      gamma: 0.9,
      episodes: 500,
      epsilon_decay: 0.995,
      competitor_model: "static",
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

  const rewardFigure = figures.primary ?? figures.reward ?? null;
  const policyFigure = figures.policy ?? null;

  return (
    <ModuleLayout
      chapterNumber={11}
      title="Dynamic Pricing"
      subtitle="Can an agent learn a pricing policy by trial and error?"
      estimatedMinutes={25}
      learningObjectives={["Markov Decision Processes", "Q-table visualization", "Policy interpretation"]}
      currentSlug="m11-qlearning"
    >
      <StorySection beat="FRAME" title="Sequential pricing decisions">
        <div className="space-y-4 text-slate-500">
          <p>
            Bandits optimize a single price decision. But pricing is sequential: today&apos;s price
            affects tomorrow&apos;s demand and competitor response. Cut your price today and
            competitors may undercut you tomorrow; hold margin and you may lose volume. The
            optimal action depends not just on the current state but on how it shapes future
            states. Q-Learning models pricing as a Markov Decision Process (MDP) where the agent
            learns a complete policy mapping every possible state to the best action.
          </p>
          <p>
            Dynamic pricing is everywhere. Airlines adjust ticket prices thousands of times daily.
            Hotels change room rates based on occupancy forecasts. Ride-sharing platforms
            surge-price in real time. The underlying framework — model the world as states,
            actions, and rewards, then learn which action to take in each state — is the same.
          </p>
          <p>
            In business terms: <strong>states</strong> are market conditions (e.g., low volume,
            high competition). <strong>Actions</strong> are price levels (undercut, parity,
            premium). <strong>Rewards</strong> are margin or revenue. The agent learns a Q-table:
            for each state-action pair, the expected cumulative reward. The policy is simple —
            in each state, pick the action with the highest Q-value.
          </p>
        </div>
      </StorySection>

      <StorySection beat="EXPLORE" title="Reward trajectory">
        <div className="space-y-4">
          <p className="text-slate-600">
            Each episode is a simulated pricing run: the agent starts in a random state, takes
            actions, receives rewards, and transitions to new states. Over episodes, the Q-table
            is updated. Early on, the agent explores (tries random actions); over time, it
            exploits (picks the best known action). The reward curve shows whether the agent
            is learning — rising and stabilizing means convergence.
          </p>
          <ChartPanel figure={rewardFigure} title="Reward over episodes" loading={exploreLoading} />
          <GuidedInsight type="notice">
            The reward stabilizes around episode 200 — the agent has converged on a policy.
          </GuidedInsight>
          <GuidedInsight type="think">
            <p>
              So what? A converged Q-Learning policy gives you a lookup table: for every market
              condition, the best price. No need to re-optimize when conditions change — the
              policy already encodes the right response. The catch: the state space must be
              discrete and small enough to store. When it isn&apos;t, you need DQN (M12).
            </p>
          </GuidedInsight>
        </div>
      </StorySection>

      <StorySection beat="BUILD" title="Train the Q-Learning agent">
        <div className="space-y-4">
          <div className="flex flex-wrap items-center gap-6">
            <label className="flex flex-col gap-1">
              <span className="text-xs text-slate-500">Gamma (0.5–0.99)</span>
              <input
                type="range"
                min={0.5}
                max={0.99}
                step={0.01}
                value={gamma}
                onChange={(e) => setGamma(Number(e.target.value))}
                className="w-32"
              />
              <span className="text-xs text-slate-500">{gamma}</span>
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-xs text-slate-500">Episodes (100–2000)</span>
              <input
                type="range"
                min={100}
                max={2000}
                step={100}
                value={episodes}
                onChange={(e) => setEpisodes(Number(e.target.value))}
                className="w-32"
              />
              <span className="text-xs text-slate-500">{episodes}</span>
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-xs text-slate-500">Epsilon decay (0.99–0.999)</span>
              <input
                type="range"
                min={0.99}
                max={0.999}
                step={0.001}
                value={epsilonDecay}
                onChange={(e) => setEpsilonDecay(Number(e.target.value))}
                className="w-32"
              />
              <span className="text-xs text-slate-500">{epsilonDecay}</span>
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-xs text-slate-500">Competitor model</span>
              <select
                value={competitorModel}
                onChange={(e) => setCompetitorModel(e.target.value as "static" | "reactive")}
                className="rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm text-slate-600"
              >
                <option value="static">Static</option>
                <option value="reactive">Reactive</option>
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
          {metrics?.final_avg_reward != null && (
            <div className="rounded-lg border border-slate-200 bg-white/50 px-4 py-2">
              <span className="text-xs text-slate-500">Final avg reward</span>
              <p className="text-lg font-semibold text-emerald-400">
                {metrics.final_avg_reward.toFixed(2)}
              </p>
            </div>
          )}
          <div className="grid gap-4 sm:grid-cols-2">
            <ChartPanel figure={rewardFigure} title="Reward trajectory" loading={loading} />
            <ChartPanel figure={policyFigure} title="Policy heatmap" loading={loading} />
          </div>
          <div className="space-y-2 text-slate-600 text-sm">
            <p className="font-medium text-slate-800">What the controls mean for pricing:</p>
            <p>
              <strong>Gamma (discount factor)</strong> — How much the agent values future rewards.
              Gamma=0.9 means a reward 10 steps from now is worth 0.9^10 ≈ 35% of an immediate
              reward. Low gamma: short-sighted, favors immediate margin. High gamma: patient,
              willing to sacrifice today for long-term gains. For pricing: if competitors react
              slowly, use high gamma; if the market resets daily, lower gamma is fine.
            </p>
            <p>
              <strong>Epsilon decay</strong> — Controls exploration vs exploitation. Early
              episodes: high epsilon, lots of random actions to fill the Q-table. Later: low
              epsilon, mostly exploit. Slower decay (e.g., 0.999) explores longer; faster decay
              (e.g., 0.99) locks in sooner. For pricing: if the environment is noisy, explore
              longer; if it&apos;s stable, decay faster.
            </p>
          </div>
        </div>
      </StorySection>

      <StorySection beat="INTERROGATE" title="Interpreting the policy">
        <div className="space-y-4">
          <ChartPanel figure={policyFigure} title="Policy heatmap" />
          <p className="text-slate-600">
            The policy heatmap shows which action the agent picks in each state. Rows might be
            volume levels; columns, competition intensity. Each cell is a color-coded action:
            undercut, parity, premium. A good policy has clear patterns — e.g., &quot;when volume
            is low and competition high, undercut&quot; — that you can validate against business
            intuition.
          </p>
          <GuidedInsight type="notice">
            In high-volume states, the policy chooses to hold or increase price. In low-volume
            states, it cuts price to win back customers.
          </GuidedInsight>
          <p className="text-slate-600">
            The same framework applies across domains. In airline pricing, states might be
            days-to-departure and seat occupancy; in ride-sharing, demand and supply imbalance.
            The policy tells you: given this state, what price maximizes long-term reward?
          </p>
          <GuidedInsight type="try">
            Set gamma=0.5 (short-sighted). The agent becomes greedy and misses long-term revenue
            gains.
          </GuidedInsight>
          <GuidedInsight type="think">
            <p>
              Business implication: the policy is a decision support tool, not a black box. You
              can inspect each state-action pair and ask: &quot;Does this make sense?&quot; If the
              model says undercut in a state where your brand forbids it, you need to reconcile —
              via constraints, post-processing, or human override.
            </p>
          </GuidedInsight>
        </div>
      </StorySection>

      <StorySection beat="DECIDE" title="Reconciling model and brand policy">
        <div className="space-y-4">
          <p className="text-slate-500">
            The Q-Learning policy says to undercut competitors by 2ct in state (low_volume,
            high_competition). But your brand positioning says &quot;never undercut.&quot; How do you
            reconcile model recommendations with business policy?
          </p>
          <p className="text-slate-600">
            <strong>When Q-Learning works:</strong> The state space is discrete and small (e.g.,
            a few dozen or few hundred states). You have enough episodes to visit each state
            many times. The environment is Markovian — the next state depends only on the current
            state and action, not the full history. Under these conditions, Q-Learning converges
            to the optimal policy.
          </p>
          <p className="text-slate-600">
            <strong>Limitations:</strong> Q-Learning does not scale to continuous states (e.g.,
            raw price, competitor price, temperature). Discretizing creates a combinatorial
            explosion: 10 bins per dimension, 5 dimensions → 100,000 states. Training becomes
            slow; the policy may miss nuances. For high-dimensional or continuous state spaces,
            use DQN (M12) instead.
          </p>
          <Assessment questions={ASSESSMENT_QUESTIONS} />
        </div>
      </StorySection>
    </ModuleLayout>
  );
}
