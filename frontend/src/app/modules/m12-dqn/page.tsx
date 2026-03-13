"use client";

import { useState, useCallback, useEffect } from "react";
import { ModuleLayout } from "@/components/module-layout";
import { StorySection } from "@/components/story-section";
import { GuidedInsight } from "@/components/guided-insight";
import { ChartPanel } from "@/components/chart-panel";
import { Assessment, type AssessmentQuestion } from "@/components/assessment";
import { apiPost } from "@/lib/api-client";
import type { PlotlyFigure } from "@/components/chart-panel";

interface M12Response {
  figures?: Record<string, PlotlyFigure>;
  charts?: Record<string, PlotlyFigure>;
  metrics?: { final_reward?: number; episodes?: number };
}

const ASSESSMENT_QUESTIONS: AssessmentQuestion[] = [
  {
    id: "q1",
    type: "critical",
    question:
      "DQN can handle continuous states but requires more data and compute. For a network of 100 stations, would you deploy DQN or stick with tabular Q-Learning?",
    modelAnswer:
      "Tabular Q-Learning is sufficient when state space is discrete and small (e.g., 405 states). DQN shines when states are continuous or high-dimensional. For 100 stations with discrete state bins, tabular is simpler, faster to train, and easier to interpret. Choose DQN when you need to incorporate raw features (price, volume, time) without manual discretization.",
  },
];

export default function M12DqnPage() {
  const [hiddenLayers, setHiddenLayers] = useState(2);
  const [units, setUnits] = useState(64);
  const [replayBufferSize, setReplayBufferSize] = useState(2000);
  const [targetFreq, setTargetFreq] = useState(50);
  const [figures, setFigures] = useState<Record<string, PlotlyFigure | null>>({});
  const [metrics, setMetrics] = useState<M12Response["metrics"]>({});
  const [loading, setLoading] = useState(false);
  const [exploreLoading, setExploreLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const train = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await apiPost<M12Response>("/api/m12/train", {
        hidden_layers: hiddenLayers,
        units,
        replay_size: replayBufferSize,
        target_freq: targetFreq,
      });
      const figs = res?.figures ?? res?.charts ?? {};
      setFigures(figs as Record<string, PlotlyFigure | null>);
      if (res?.metrics) setMetrics(res.metrics);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Training failed");
    } finally {
      setLoading(false);
    }
  }, [hiddenLayers, units, replayBufferSize, targetFreq]);

  useEffect(() => {
    let cancelled = false;
    setExploreLoading(true);
    apiPost<M12Response>("/api/m12/train", {
      hidden_layers: 2,
      units: 64,
      replay_size: 2000,
      target_freq: 50,
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

  const trainingFigure = figures.primary ?? figures.training_curve ?? figures.training ?? null;

  return (
    <ModuleLayout
      chapterNumber={12}
      title="Deep Reinforcement Learning"
      subtitle="What happens when you give the agent a neural network for a brain?"
      estimatedMinutes={25}
      learningObjectives={["Experience replay", "Target networks", "Neural Q-function"]}
      currentSlug="m12-dqn"
    >
      <StorySection beat="FRAME" title="From Q-table to neural network">
        <div className="space-y-4 text-slate-500">
          <p>
            Q-Learning stores one Q-value per state-action pair. With 405 states and 5 actions,
            that&apos;s 2,025 entries for a simple grid. Real pricing has continuous states —
            infinite combinations of price, competitor price, temperature, day-of-week, and more.
            A Q-table would need millions of cells. DQN replaces the table with a neural network
            that generalizes: it learns a function from state to Q-values, so it can handle states
            it has never seen.
          </p>
          <p>
            DQN is the same technology behind game-playing AIs. Applied to pricing: the neural
            network learns which market conditions favor aggressive pricing vs holding margins.
            Each &quot;game&quot; is a pricing episode where the agent interacts with the simulated
            market, observes rewards, and updates its weights. Over thousands of episodes, it
            converges to a policy that works across the full state space.
          </p>
          <p>
            Two key innovations make DQN stable: <strong>Experience replay</strong> stores past
            transitions (state, action, reward, next state) in a buffer and samples from it
            randomly. The agent learns from past pricing decisions multiple times — like a team
            reviewing case studies of past campaigns. <strong>Target networks</strong> provide a
            stable learning signal: the network is trained against a slowly updated copy of
            itself, reducing the feedback loops that cause Q-values to oscillate.
          </p>
        </div>
      </StorySection>

      <StorySection beat="EXPLORE" title="Training curve">
        <div className="space-y-4">
          <p className="text-slate-600">
            The training curve plots reward (or average reward) over episodes. Unlike Q-Learning,
            which updates one Q-value per step, DQN updates a neural network from batches of
            experience. The curve is typically noisier — rewards can spike or dip as the agent
            explores different parts of the state space. Convergence is slower but the payoff
            is generalization: the policy works for any state, not just those in a finite table.
          </p>
          <ChartPanel
            figure={trainingFigure}
            title="DQN training curve"
            loading={exploreLoading}
          />
          <GuidedInsight type="notice">
            The DQN training curve is noisier than Q-Learning because experience replay breaks
            temporal correlations.
          </GuidedInsight>
        </div>
      </StorySection>

      <StorySection beat="BUILD" title="Train the DQN">
        <div className="space-y-4">
          <div className="flex flex-wrap items-center gap-6">
            <label className="flex flex-col gap-1">
              <span className="text-xs text-slate-500">Hidden layers (1–4)</span>
              <input
                type="range"
                min={1}
                max={4}
                value={hiddenLayers}
                onChange={(e) => setHiddenLayers(Number(e.target.value))}
                className="w-32"
              />
              <span className="text-xs text-slate-500">{hiddenLayers}</span>
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-xs text-slate-500">Units per layer (16–128)</span>
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
              <span className="text-xs text-slate-500">Replay buffer (500–5000)</span>
              <input
                type="range"
                min={500}
                max={5000}
                step={500}
                value={replayBufferSize}
                onChange={(e) => setReplayBufferSize(Number(e.target.value))}
                className="w-32"
              />
              <span className="text-xs text-slate-500">{replayBufferSize}</span>
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-xs text-slate-500">Target update freq (10–100)</span>
              <input
                type="range"
                min={10}
                max={100}
                step={10}
                value={targetFreq}
                onChange={(e) => setTargetFreq(Number(e.target.value))}
                className="w-32"
              />
              <span className="text-xs text-slate-500">{targetFreq}</span>
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
          {metrics?.final_reward != null && (
            <div className="rounded-lg border border-slate-200 bg-slate-50 px-4 py-2">
              <span className="text-xs text-slate-500">Final reward</span>
              <p className="text-lg font-semibold text-emerald-400">
                {metrics.final_reward.toFixed(2)}
              </p>
            </div>
          )}
          <ChartPanel
            figure={trainingFigure}
            title="Training curve"
            loading={loading}
          />
        </div>
      </StorySection>

      <StorySection beat="INTERROGATE" title="Key DQN components">
        <div className="space-y-4">
          <GuidedInsight type="try">
            Reduce replay buffer to 100. The agent becomes unstable — it &quot;forgets&quot; good
            experiences.
          </GuidedInsight>
          <GuidedInsight type="warning">
            Without a target network (target_freq=1), Q-values oscillate wildly. The target network
            provides a stable learning signal.
          </GuidedInsight>
        </div>
      </StorySection>

      <StorySection beat="DECIDE" title="DQN or tabular Q-Learning?">
        <div className="space-y-4">
          <p className="text-slate-500">
            DQN can handle continuous states but requires more data and compute. For a network of 100
            stations, would you deploy DQN or stick with tabular Q-Learning?
          </p>
          <p className="text-slate-600">
            <strong>Practical considerations for deploying RL in pricing:</strong> (1) <em>Simulation
            vs reality</em> — RL agents are typically trained in a simulated environment. The
            simulation must match reality; otherwise the policy will fail. (2) <em>Safety</em> —
            constrain actions (e.g., max price change per day) to avoid extreme recommendations.
            (3) <em>Monitoring</em> — track reward, action distribution, and policy drift;
            retrain when the environment shifts. (4) <em>Interpretability</em> — DQN is harder
            to explain than bandits or tabular Q-Learning. For regulated industries or
            stakeholder buy-in, consider simpler models first.
          </p>
          <Assessment questions={ASSESSMENT_QUESTIONS} />
        </div>
      </StorySection>
    </ModuleLayout>
  );
}
