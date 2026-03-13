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
  figures?: { primary?: PlotlyFigure };
  metrics?: { train_rmse?: number; test_rmse?: number };
}

const ASSESSMENT_QUESTIONS: AssessmentQuestion[] = [
  {
    id: "q1",
    type: "conceptual",
    question: "What is bias in the context of model complexity?",
    options: [
      "Error from wrong assumptions about the true relationship",
      "Error from random fluctuations in the training data",
      "Error from having too many features",
      "Error from using the wrong algorithm",
    ],
    modelAnswer:
      "Bias is the error from wrong assumptions — a degree-1 line assumes a linear relationship; if the true relationship is curved, that assumption creates systematic error.",
  },
  {
    id: "q2",
    type: "applied",
    question: "Given this data, what degree would you recommend for production?",
    options: ["1–2", "3–4", "5–7", "8+"],
    modelAnswer:
      "Typically 3–4 balances train and test RMSE. Higher degrees overfit; lower degrees underfit. Check where test RMSE is lowest and stable.",
  },
  {
    id: "q3",
    type: "critical",
    question: "When might a simple model be preferred even if a complex one has lower test error?",
    options: [
      "When interpretability matters for stakeholders",
      "When the complex model is harder to maintain",
      "When you have very little data for retraining",
      "All of the above",
    ],
    modelAnswer:
      "All of the above. Simplicity aids interpretability, maintenance, and robustness when data shifts. A slightly higher test error may be acceptable for these benefits.",
  },
];

export default function M00FoundationsPage() {
  const [degree, setDegree] = useState(3);
  const [trainSplit, setTrainSplit] = useState(0.7);
  const [loading, setLoading] = useState(false);
  const [exploreFigure, setExploreFigure] = useState<PlotlyFigure | null>(null);
  const [buildFigure, setBuildFigure] = useState<PlotlyFigure | null>(null);
  const [trainRmse, setTrainRmse] = useState<number | null>(null);
  const [testRmse, setTestRmse] = useState<number | null>(null);

  const train = useCallback(async () => {
    setLoading(true);
    setBuildFigure(null);
    setTrainRmse(null);
    setTestRmse(null);
    try {
      const res = await apiPost<TrainResponse>("/api/m00/train", {
        degree,
        train_split: trainSplit,
      });
      const fig = res?.figures?.primary;
      if (fig) setBuildFigure(fig as PlotlyFigure);
      if (res?.metrics?.train_rmse != null) setTrainRmse(res.metrics.train_rmse);
      if (res?.metrics?.test_rmse != null) setTestRmse(res.metrics.test_rmse);
    } catch {
      // Error handled by UI
    } finally {
      setLoading(false);
    }
  }, [degree, trainSplit]);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const res = await apiPost<TrainResponse>("/api/m00/train", {
          degree: 1,
          train_split: 0.8,
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
      chapterNumber={0}
      title="The Bias-Variance Tradeoff"
      subtitle="Why does adding more complexity sometimes make predictions worse?"
      estimatedMinutes={15}
      learningObjectives={[
        "Bias vs variance",
        "Overfitting detection",
        "Train/test methodology",
      ]}
      currentSlug="m00-foundations"
    >
      <StorySection beat="FRAME" title="The pricing analyst's dilemma">
        <div className="space-y-4 text-slate-500">
          <p>
            Imagine you manage pricing for a 200-location retail network. Your
            analyst builds a demand model to predict volume from price. You try
            a simple line — it misses the curve. A polynomial of degree 10 fits
            the training data perfectly… but fails on new stations. Your
            manager asks: &quot;Can we predict demand from price?&quot; The
            answer depends on how you balance simplicity and complexity.
          </p>
          <p>
            Whether you price fuel at motorway stations, set promotional prices
            for convenience goods, or negotiate supply chain contracts, the
            same tension appears: too simple and you underfit; too complex and
            you overfit. The bias–variance tradeoff is the lens that explains
            why.
          </p>
          <p>
            Bias is like always assuming demand is linear — you systematically
            miss the curve. Variance is like memorizing last week&apos;s sales
            exactly — your predictions jump wildly when next week looks
            different. A good model finds the sweet spot between these two
            sources of error.
          </p>
          <p>
            This module answers: <em>Why does adding more complexity sometimes
            make predictions worse?</em> By the end, you&apos;ll know how to
            diagnose overfitting, interpret train vs test error, and choose a
            model that generalizes to new data.
          </p>
        </div>
      </StorySection>

      <StorySection beat="EXPLORE" title="Price vs volume: a degree-1 fit">
        <div className="space-y-4">
          <p className="text-slate-500">
            The chart below shows price on the horizontal axis and volume on the
            vertical axis, with a linear (degree-1) fit drawn through the data.
            The visualization reveals how well — or poorly — a straight line
            captures the true demand relationship. In fuel retail, for example,
            volume often drops sharply at high prices but flattens at low prices;
            a linear model cannot capture that curvature. The same pattern
            appears in many domains: promotional elasticity in FMCG, occupancy
            vs room rate in hotels, or throughput vs capacity in logistics.
          </p>
          <ChartPanel
            figure={exploreFigure}
            title="Price vs volume with linear fit"
            loading={!exploreFigure && !loading}
          />
          <GuidedInsight type="notice">
            Notice the curve in the residuals — a straight line is too simple for
            this relationship. The model assumes a linear effect of price on
            volume, but the real demand curve bends.
          </GuidedInsight>
        </div>
      </StorySection>

      <StorySection beat="BUILD" title="Experiment with degree and train split">
        <div className="space-y-4">
          <div className="flex flex-wrap gap-6">
            <label className="flex flex-col gap-1">
              <span className="text-xs text-slate-500">Degree (1–12)</span>
              <input
                type="range"
                min={1}
                max={12}
                value={degree}
                onChange={(e) => setDegree(Number(e.target.value))}
                className="w-40"
              />
              <span className="text-xs text-slate-500">{degree}</span>
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-xs text-slate-500">Train split (60–90%)</span>
              <input
                type="range"
                min={0.6}
                max={0.9}
                step={0.05}
                value={trainSplit}
                onChange={(e) => setTrainSplit(Number(e.target.value))}
                className="w-40"
              />
              <span className="text-xs text-slate-500">
                {Math.round(trainSplit * 100)}%
              </span>
            </label>
            <button
              onClick={train}
              disabled={loading}
              className="mt-2 rounded-lg bg-amber-500 px-4 py-2 text-sm font-medium text-slate-900 hover:bg-amber-400 disabled:opacity-50"
            >
              Train
            </button>
          </div>
          <p className="text-slate-500">
            The degree slider controls model complexity — think of it as how many
            bends the demand curve is allowed to make. The train split decides
            what percentage of your historical data goes to building the model
            versus testing it. A higher train split gives the model more data to
            learn from, but less data to validate on; a lower split does the
            opposite.
          </p>
          <div className="flex gap-4">
            {trainRmse != null && (
              <div className="rounded-lg border border-slate-200 bg-slate-50 px-4 py-2">
                <span className="text-xs text-slate-500">Train RMSE</span>
                <p className="text-lg font-semibold text-emerald-400">
                  {trainRmse.toFixed(2)}
                </p>
              </div>
            )}
            {testRmse != null && (
              <div className="rounded-lg border border-slate-200 bg-slate-50 px-4 py-2">
                <span className="text-xs text-slate-500">Test RMSE</span>
                <p className="text-lg font-semibold text-sky-400">
                  {testRmse.toFixed(2)}
                </p>
              </div>
            )}
          </div>
          <ChartPanel
            figure={buildFigure}
            title="Polynomial fit"
            loading={loading}
          />
          <GuidedInsight type="try">
            Set degree to 10. Watch test RMSE explode while train RMSE drops.
            That&apos;s overfitting in action.
          </GuidedInsight>
        </div>
      </StorySection>

      <StorySection beat="INTERROGATE" title="Train vs test RMSE">
        <div className="space-y-4">
          <div className="rounded-lg border border-slate-200 bg-slate-50 p-4">
            <p className="text-sm text-slate-600">
              Compare train and test RMSE. When they diverge, the model has
              memorized training noise instead of learning the underlying
              pattern.
            </p>
            {trainRmse != null && testRmse != null && (
              <p className="mt-2 text-sm text-slate-500">
                Current: Train RMSE = {trainRmse.toFixed(2)}, Test RMSE ={" "}
                {testRmse.toFixed(2)}
              </p>
            )}
          </div>
          <p className="text-slate-500">
            Results look the way they do because the training set is used to
            fit the model, while the test set simulates unseen data. A model
            that fits training data too closely will chase random fluctuations
            that don&apos;t repeat — hence high test RMSE. A model that&apos;s too
            simple will miss real structure — hence high train RMSE too.
          </p>
          <p className="text-slate-500">
            A common mistake in pricing teams is using the most complex model
            available because it &quot;fits better.&quot; That often means it
            fits the noise. When you deploy such a model, small changes in
            market conditions cause large swings in predicted demand, and
            stakeholders lose trust.
          </p>
          <p className="text-slate-500">
            A supply chain manager forecasting warehouse demand faces the same
            tradeoff. A simple seasonal model may miss spikes; an overly complex
            one may overreact to last month&apos;s anomalies. The bias–variance
            lens applies wherever you predict from limited data.
          </p>
          <GuidedInsight type="warning">
            When test RMSE &gt; 2× train RMSE, you&apos;ve overfit. Real-world
            predictions will be unreliable.
          </GuidedInsight>
          <GuidedInsight type="think">
            In business, a model that generalizes poorly doesn&apos;t just
            produce bad numbers — it undermines decisions. If your demand
            forecast swings wildly with each retrain, pricing and inventory
            teams will revert to gut feel. Choose a model that balances
            accuracy with stability.
          </GuidedInsight>
        </div>
      </StorySection>

      <StorySection beat="DECIDE" title="What degree for production?">
        <div className="space-y-4">
          <p className="text-slate-500">
            Given this tradeoff, what degree would you recommend for production?
            Consider bias, variance, and the cost of being wrong on new
            stations.
          </p>
          <p className="text-slate-500">
            Use simpler models (degree 1–2) when interpretability matters, when
            you have little data for retraining, or when the business needs
            stable predictions that don&apos;t swing with every data refresh.
            Use more complex models (degree 3–4) when you have enough data, when
            the true relationship is clearly nonlinear, and when stakeholders
            accept some opacity for higher accuracy. In real-world deployment,
            consider maintenance: a simpler model is easier to debug, explain,
            and update when market structure shifts.
          </p>
          <Assessment questions={ASSESSMENT_QUESTIONS} />
        </div>
      </StorySection>
    </ModuleLayout>
  );
}
