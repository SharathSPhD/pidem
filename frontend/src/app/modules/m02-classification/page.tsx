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
  metrics?: {
    confusion_matrix_costs?: {
      tn?: number;
      fp?: number;
      fn?: number;
      tp?: number;
      cost_fn?: number;
      cost_fp?: number;
      total_cost?: number;
    };
    auc?: number;
    precision?: number;
    recall?: number;
  };
}

const ASSESSMENT_QUESTIONS: AssessmentQuestion[] = [
  {
    id: "q1",
    type: "conceptual",
    question: "What does the ROC AUC score represent?",
    options: [
      "The probability that the model ranks a random positive higher than a random negative",
      "The accuracy of the model",
      "The proportion of correct predictions",
      "The threshold that maximizes F1",
    ],
    modelAnswer:
      "AUC is the probability that the model ranks a random positive (threat) higher than a random negative. AUC = 0.5 is random; AUC = 1 is perfect discrimination.",
  },
  {
    id: "q2",
    type: "applied",
    question: "Given the cost asymmetry (FN = EUR 500, FP = EUR 100), which threshold minimizes total cost?",
    options: [
      "Lower threshold — catch more threats even if more false alarms",
      "Higher threshold — fewer false alarms even if we miss threats",
      "Depends on the base rate of threats",
      "Always 0.5",
    ],
    modelAnswer:
      "Depends on the base rate. With high FN cost, you typically want lower threshold (higher recall) to catch threats — but if threats are rare, too many FPs can dominate. Run the numbers for your scenario.",
  },
  {
    id: "q3",
    type: "critical",
    question: "Given the cost asymmetry, would you choose precision or recall as your primary metric?",
    options: [
      "Precision — minimize false alarms",
      "Recall — minimize missed threats",
      "F1 — balance both",
      "Total cost — optimize for business outcome",
    ],
    modelAnswer:
      "Total cost. Precision and recall are proxies; the business cares about EUR. A false negative costs EUR 500; a false positive costs EUR 100. Optimize the threshold to minimize total_cost = FN×500 + FP×100.",
  },
];

const MODEL_TYPES = [
  { value: "logistic", label: "Logistic" },
  { value: "tree", label: "Decision Tree" },
  { value: "xgboost", label: "XGBoost" },
];

export default function M02ClassificationPage() {
  const [modelType, setModelType] = useState("logistic");
  const [threshold, setThreshold] = useState(0.5);
  const [treeDepth, setTreeDepth] = useState(4);
  const [loading, setLoading] = useState(false);
  const [exploreFigure, setExploreFigure] = useState<PlotlyFigure | null>(null);
  const [buildFigure, setBuildFigure] = useState<PlotlyFigure | null>(null);
  const [confusionMatrix, setConfusionMatrix] = useState<PlotlyFigure | null>(null);
  const [rocFigure, setRocFigure] = useState<PlotlyFigure | null>(null);
  const [prFigure, setPrFigure] = useState<PlotlyFigure | null>(null);
  const [shapFigure, setShapFigure] = useState<PlotlyFigure | null>(null);
  const [costs, setCosts] = useState<{
    tn?: number;
    fp?: number;
    fn?: number;
    tp?: number;
    cost_fn?: number;
    cost_fp?: number;
    total_cost?: number;
  } | null>(null);

  const train = useCallback(async () => {
    setLoading(true);
    setBuildFigure(null);
    setConfusionMatrix(null);
    setRocFigure(null);
    setPrFigure(null);
    setShapFigure(null);
    setCosts(null);
    try {
      const res = await apiPost<TrainResponse>("/api/m02/train", {
        model_type: modelType,
        threshold,
        tree_depth: treeDepth,
      });
      const figs = res?.figures ?? {};
      if (figs.confusion_matrix) {
        setConfusionMatrix(figs.confusion_matrix as PlotlyFigure);
        setBuildFigure(figs.confusion_matrix as PlotlyFigure);
      }
      if (figs.roc) setRocFigure(figs.roc as PlotlyFigure);
      if (figs.precision_recall) setPrFigure(figs.precision_recall as PlotlyFigure);
      if (figs.shap_beeswarm) setShapFigure(figs.shap_beeswarm as PlotlyFigure);
      if (res?.metrics?.confusion_matrix_costs)
        setCosts(res.metrics.confusion_matrix_costs);
    } catch {
      // Error handled by UI
    } finally {
      setLoading(false);
    }
  }, [modelType, threshold, treeDepth]);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const res = await apiPost<TrainResponse>("/api/m02/train", {
          model_type: "logistic",
          threshold: 0.5,
          tree_depth: 4,
        });
        if (!cancelled && res?.figures?.confusion_matrix) {
          setExploreFigure(res.figures.confusion_matrix as PlotlyFigure);
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
      chapterNumber={2}
      title="Threat Detection"
      subtitle="Can you predict which competitor moves will cost you 15% of volume?"
      estimatedMinutes={20}
      learningObjectives={[
        "Binary classification",
        "Confusion matrix business costs",
        "Threshold tuning",
      ]}
      currentSlug="m02-classification"
    >
      <StorySection beat="FRAME" title="Predicting volume loss">
        <div className="space-y-4 text-slate-500">
          <p>
            A competitor drops price by 5ct at 3 locations near yours. Within 48
            hours, your volume drops 18%. Could you have predicted this — and
            acted before the damage was done?
          </p>
          <p>
            Threat detection isn&apos;t just about competitors undercutting your
            fuel price. A convenience retailer needs to spot when a new coffee
            chain is cannibalizing morning traffic. A supply chain manager needs
            to flag when a key supplier&apos;s quality metrics signal a coming
            disruption. In each case, you&apos;re classifying events as
            &quot;threat&quot; or &quot;not threat&quot; — and the cost of
            getting it wrong is asymmetric.
          </p>
          <p>
            Binary classification assigns each event to one of two classes. The
            confusion matrix — true positives, false positives, true negatives,
            false negatives — tells you where the model succeeds and where it
            fails. In business terms, a false negative (missing a real threat)
            often costs far more than a false positive (a false alarm). That cost
            asymmetry drives your threshold choice: lower threshold catches more
            threats but triggers more false alarms; higher threshold reduces
            noise but lets threats slip through.
          </p>
          <p>
            A classifier trained on historical competitive moves can flag
            threats before they become losses. The red dots mark days where
            volume dropped &gt;15% vs rolling baseline — these are the threats
            we want to predict.
          </p>
        </div>
      </StorySection>

      <StorySection beat="EXPLORE" title="Confusion matrix">
        <div className="space-y-4">
          <p className="text-slate-500">
            Before tuning thresholds or costs, you need to see how the model
            partitions predictions. The confusion matrix shows four quadrants:
            correct predictions (true positives and true negatives) and errors
            (false positives and false negatives). Each quadrant maps to a
            business outcome — and each error type has a different cost.
          </p>
          <ChartPanel
            figure={exploreFigure}
            title="Confusion matrix (default model)"
            loading={!exploreFigure && !loading}
          />
          <GuidedInsight type="notice">
            The red dots mark days where volume dropped &gt;15% vs rolling
            baseline — these are the threats we want to predict. The confusion
            matrix shows how well the model separates them.
          </GuidedInsight>
        </div>
      </StorySection>

      <StorySection beat="BUILD" title="Train the threat classifier">
        <div className="space-y-4">
          <p className="text-slate-500">
            Model type trades off interpretability (logistic) vs accuracy
            (XGBoost). Threshold controls the trade-off between catching threats
            and avoiding false alarms — lower threshold means more alerts, higher
            recall. Tree depth affects how complex the decision boundary is;
            deeper trees can overfit to noise.
          </p>
          <div className="flex flex-wrap gap-6">
            <label className="flex flex-col gap-1">
              <span className="text-xs text-slate-500">Model type</span>
              <select
                value={modelType}
                onChange={(e) => setModelType(e.target.value)}
                className="rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm text-slate-600"
              >
                {MODEL_TYPES.map((t) => (
                  <option key={t.value} value={t.value}>
                    {t.label}
                  </option>
                ))}
              </select>
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-xs text-slate-500">Threshold (0.05–0.5)</span>
              <input
                type="range"
                min={0.05}
                max={0.5}
                step={0.05}
                value={threshold}
                onChange={(e) => setThreshold(Number(e.target.value))}
                className="w-40"
              />
              <span className="text-xs text-slate-500">{threshold}</span>
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-xs text-slate-500">Tree depth (2–8)</span>
              <input
                type="range"
                min={2}
                max={8}
                value={treeDepth}
                onChange={(e) => setTreeDepth(Number(e.target.value))}
                className="w-40"
              />
              <span className="text-xs text-slate-500">{treeDepth}</span>
            </label>
            <button
              onClick={train}
              disabled={loading}
              className="mt-2 rounded-lg bg-amber-500 px-4 py-2 text-sm font-medium text-slate-900 hover:bg-amber-400 disabled:opacity-50"
            >
              Train
            </button>
          </div>
          <ChartPanel figure={buildFigure} title="Confusion matrix" loading={loading} />
          <div className="grid gap-4 sm:grid-cols-2">
            <ChartPanel figure={rocFigure} title="ROC curve" />
            <ChartPanel figure={prFigure} title="Precision–Recall" />
          </div>
          {shapFigure && (
            <ChartPanel figure={shapFigure} title="SHAP beeswarm" />
          )}
        </div>
      </StorySection>

      <StorySection beat="INTERROGATE" title="Business costs">
        <div className="space-y-4">
          <p className="text-slate-500">
            Here we assign explicit costs: a false negative (missed threat) costs
            EUR 500 in lost volume; a false positive (false alarm) costs EUR 100
            in wasted analyst time. The same logic applies across domains — a
            retailer missing a competitor&apos;s promo might lose margin; a
            supply chain missing a supplier risk might face stockouts. Optimizing
            for total cost, not accuracy, aligns the model with business
            outcomes.
          </p>
          <ChartPanel figure={confusionMatrix} title="Confusion matrix" />
          {costs && (
            <div className="rounded-lg border border-slate-200 bg-slate-50 p-4">
              <p className="text-xs text-slate-500">Cost breakdown</p>
              <div className="mt-2 flex flex-wrap gap-4">
                <span className="text-sm text-slate-500">
                  TN: {costs.tn ?? 0} | FP: {costs.fp ?? 0} (×{costs.cost_fp ?? 100} = {(costs.fp ?? 0) * (costs.cost_fp ?? 100)} EUR)
                </span>
                <span className="text-sm text-slate-500">
                  FN: {costs.fn ?? 0} (×{costs.cost_fn ?? 500} = {(costs.fn ?? 0) * (costs.cost_fn ?? 500)} EUR) | TP: {costs.tp ?? 0}
                </span>
                <span className="font-semibold text-amber-600">
                  Total cost: {costs.total_cost ?? 0} EUR
                </span>
              </div>
            </div>
          )}
          <GuidedInsight type="think">
            A false negative (missing a threat) costs EUR 500 in lost volume. A
            false positive (false alarm) costs EUR 100 in wasted analyst time.
            Which threshold minimizes total cost?
          </GuidedInsight>
          <GuidedInsight type="notice">
            In domains where threats are rare, even a low false-positive rate can
            swamp you with alerts. In domains where threats are common, missing
            one is costly. Run the numbers for your base rate and cost structure.
          </GuidedInsight>
        </div>
      </StorySection>

      <StorySection beat="DECIDE" title="Precision or recall?">
        <div className="space-y-4">
          <p className="text-slate-500">
            Given the cost asymmetry, would you choose precision or recall as
            your primary metric? Or would you optimize for total cost?
          </p>
          <p className="text-slate-500">
            The decision framework: (1) Quantify costs for FN and FP in your
            domain. (2) Estimate the base rate of threats. (3) Sweep the
            threshold and compute total cost for each. (4) Choose the threshold
            that minimizes total cost. Precision and recall are proxies; the
            business cares about EUR. When stakes are high and costs are
            asymmetric, defaulting to 0.5 or F1-maximizing thresholds is
            suboptimal.
          </p>
          <Assessment questions={ASSESSMENT_QUESTIONS} />
        </div>
      </StorySection>
    </ModuleLayout>
  );
}
