"use client";

import { useState, useCallback, useEffect } from "react";
import { ModuleLayout } from "@/components/module-layout";
import { StorySection } from "@/components/story-section";
import { GuidedInsight } from "@/components/guided-insight";
import { ChartPanel } from "@/components/chart-panel";
import { Assessment, type AssessmentQuestion } from "@/components/assessment";
import { apiPost } from "@/lib/api-client";
import type { PlotlyFigure } from "@/components/chart-panel";

interface M04Response {
  figures?: Record<string, PlotlyFigure>;
  metrics?: Record<string, number | string | null | undefined>;
  data?: { profiles?: Record<string, Record<string, number>> };
}

const ASSESSMENT_QUESTIONS: AssessmentQuestion[] = [
  {
    id: "q1",
    type: "conceptual",
    question: "How does the elbow method help choose K?",
    modelAnswer:
      "The elbow method plots inertia (within-cluster sum of squares) against K. The 'elbow' point—where the curve bends sharply—suggests a natural number of clusters. Beyond that point, adding more clusters yields diminishing returns.",
  },
  {
    id: "q2",
    type: "applied",
    question: "What does silhouette score measure?",
    modelAnswer:
      "Silhouette score measures how well each point fits its assigned cluster versus the nearest neighboring cluster. Values range from -1 to 1; higher values indicate better-defined, well-separated clusters.",
  },
  {
    id: "q3",
    type: "critical",
    question: "If clusters reveal that 12 'urban' stations have motorway-like characteristics, what operational implications might reclassification have?",
    options: [
      "Pricing strategy only",
      "Supply chain and pricing",
      "No impact",
      "Marketing only",
    ],
    modelAnswer:
      "Reclassification affects both pricing (motorway stations often command premium prices) and supply chain (delivery schedules, product categories). It may also impact staffing and promotional strategies.",
  },
];

export default function M04ClusteringPage() {
  const [k, setK] = useState(4);
  const [figures, setFigures] = useState<Record<string, PlotlyFigure | null>>({});
  const [metrics, setMetrics] = useState<Record<string, number | string | null | undefined>>({});
  const [profiles, setProfiles] = useState<Record<string, Record<string, number>>>({});
  const [loading, setLoading] = useState(false);
  const [exploreLoading, setExploreLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const train = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await apiPost<M04Response>("/api/m04/train", { k });
      if (res?.figures) setFigures(res.figures as Record<string, PlotlyFigure | null>);
      if (res?.metrics) setMetrics(res.metrics);
      if (res?.data?.profiles) setProfiles(res.data.profiles);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Training failed");
    } finally {
      setLoading(false);
    }
  }, [k]);

  useEffect(() => {
    let cancelled = false;
    setExploreLoading(true);
    apiPost<M04Response>("/api/m04/train", { k: 4 })
      .then((res) => {
        if (!cancelled && res?.figures) {
          setFigures(res.figures as Record<string, PlotlyFigure | null>);
          if (res?.metrics) setMetrics(res.metrics);
          if (res?.data?.profiles) setProfiles(res.data.profiles ?? {});
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

  const silhouetteScore = metrics?.silhouette_score ?? metrics?.silhouette;

  return (
    <ModuleLayout
      chapterNumber={4}
      title="Market Segmentation"
      subtitle="Are your 'motorway' and 'urban' labels really the best grouping?"
      estimatedMinutes={20}
      learningObjectives={["K-Means clustering", "Elbow method", "PCA visualization"]}
      currentSlug="m04-clustering"
    >
      <StorySection beat="FRAME" title="The problem with hand-assigned labels">
        <div className="space-y-4 text-slate-500">
          <p>
            Your network has 100 stations labeled &quot;motorway,&quot; &quot;urban,&quot; and
            &quot;suburban.&quot; But pricing strategy uses these labels, and they were assigned years
            ago. Should we reprice the stations that cluster together differently than our current
            labels suggest? K-Means discovers natural groupings in your data — like finding that some
            &quot;urban&quot; stations behave like motorway stations because of traffic and captive
            demand. Those stations may deserve motorway-style pricing.
          </p>
          <p>
            Whether you segment fuel stations, group convenience store SKUs by purchasing patterns, or
            classify suppliers by reliability metrics, the core question is the same: do your current
            categories reflect reality, or are they legacy labels that no longer serve you?
          </p>
          <p>
            K-Means is like asking an objective intern to sort your locations into groups based purely
            on behavior data — ignoring the labels you&apos;ve always used. The algorithm finds natural
            groupings that might surprise you: some &quot;urban&quot; locations may behave like
            &quot;motorway&quot; sites because of traffic patterns and captive demand.
          </p>
          <p>
            In this module you&apos;ll learn: PCA for visualization (compressing many features into two
            dimensions), the elbow method for choosing K, and silhouette scores for validating cluster
            quality.
          </p>
        </div>
      </StorySection>

      <StorySection beat="EXPLORE" title="PCA scatter — do labels match clusters?">
        <div className="space-y-4">
          <p className="text-slate-500">
            The PCA scatter plot compresses many features into two dimensions so you can see clusters
            visually. Each dot is a location, colored by cluster assignment. If you see dots from
            different official labels mixing within the same cluster, your current segmentation
            doesn&apos;t match the data.
          </p>
          <GuidedInsight type="notice">
            Stations naturally form clusters that don&apos;t perfectly align with the official
            labels.
          </GuidedInsight>
          <ChartPanel
            figure={figures.primary ?? figures.pca ?? null}
            title="Station clusters (PCA)"
            loading={exploreLoading}
          />
        </div>
      </StorySection>

      <StorySection beat="BUILD" title="Train K-Means with your chosen K">
        <div className="space-y-4">
          <div className="flex flex-wrap items-center gap-4">
            <label className="flex flex-col gap-1">
              <span className="text-xs text-slate-500">K (2–10)</span>
              <input
                type="range"
                min={2}
                max={10}
                value={k}
                onChange={(e) => setK(Number(e.target.value))}
                className="w-32"
              />
              <span className="text-xs text-slate-500">{k}</span>
            </label>
            <button
              onClick={train}
              disabled={loading}
              className="rounded-lg bg-amber-500 px-4 py-2 text-sm font-medium text-slate-900 hover:bg-amber-400 disabled:opacity-50"
            >
              {loading ? "Training…" : "Train"}
            </button>
          </div>
          <p className="text-slate-500">
            K controls how many segments the algorithm creates. Too few and you miss important
            differences between locations; too many and each segment is too small for a meaningful
            pricing strategy. The sweet spot is where you have enough segments to capture real
            behavioral differences but not so many that operational complexity explodes.
          </p>
          {error && (
            <p className="text-sm text-rose-400">{error}</p>
          )}
          <div className="grid gap-4 sm:grid-cols-2">
            <ChartPanel
              figure={figures.primary ?? figures.pca ?? null}
              title="PCA scatter"
              loading={loading}
            />
            <ChartPanel figure={figures.elbow ?? null} title="Elbow chart" loading={loading} />
          </div>
          {typeof silhouetteScore === "number" && (
            <p className="text-sm text-slate-500">
              Silhouette score: <span className="font-mono text-amber-600">{silhouetteScore.toFixed(3)}</span>
            </p>
          )}
        </div>
      </StorySection>

      <StorySection beat="INTERROGATE" title="Interpreting the elbow and silhouette">
        <div className="space-y-4">
          <p className="text-slate-500">
            The elbow method plots how much variance remains within clusters as K increases. In
            business terms: imagine you&apos;re splitting your network into more and more segments.
            At first, each new segment captures a meaningful difference (e.g., motorway vs. urban).
            Eventually you hit a point where adding another segment barely improves fit — that&apos;s the
            elbow. Beyond it, you&apos;re over-segmenting.
          </p>
          <p className="text-slate-500">
            Think of silhouette as a measure of how confident the algorithm is about each assignment.
            A high silhouette means each location clearly belongs to its cluster; a low score means
            it sits on the boundary between two clusters and could arguably go either way.
          </p>
          <p className="text-slate-500">
            A category manager running the same analysis on SKU sales data might find that
            &quot;premium snacks&quot; and &quot;impulse beverages&quot; cluster together — both are
            high-margin, low-frequency items driven by the same customer behavior.
          </p>
          <GuidedInsight type="try">
            Set K=2, then K=8. The silhouette score peaks around K=4, suggesting 4 natural segments.
          </GuidedInsight>
          <GuidedInsight type="warning">
            K=8 fragments your network into too-small groups — pricing becomes unpredictable.
          </GuidedInsight>
          <GuidedInsight type="think">
            If 12 locations reclassify from &quot;urban&quot; to &quot;motorway-like,&quot; the pricing
            impact could be 3–5% margin improvement per location — but only if operations (delivery,
            staffing) can support the change.
          </GuidedInsight>
          <ChartPanel figure={figures.elbow ?? null} title="Elbow chart" />
        </div>
      </StorySection>

      <StorySection beat="DECIDE" title="Reclassification decision">
        <div className="space-y-4 text-slate-500">
          <p>
            If clusters reveal that 12 &quot;urban&quot; stations have motorway-like characteristics,
            should you reclassify them? What are the operational implications?
          </p>
          <p>
            A decision framework: consider whether operations (delivery routes, staffing, product mix)
            can support the change. Reclassification may improve pricing accuracy, but it also demands
            that supply chain and store teams treat those locations differently. When to override the
            algorithm: if a location clusters with motorway sites but has structural constraints (e.g.,
            no space for premium forecourt services), business judgment may keep it in the urban tier.
            Use clustering as evidence, not as the final word.
          </p>
        </div>
      </StorySection>

      <Assessment questions={ASSESSMENT_QUESTIONS} />
    </ModuleLayout>
  );
}
