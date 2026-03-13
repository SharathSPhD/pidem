"use client";

import { useState, useCallback, useEffect } from "react";
import { ModuleLayout } from "@/components/module-layout";
import { StorySection } from "@/components/story-section";
import { GuidedInsight } from "@/components/guided-insight";
import { ChartPanel } from "@/components/chart-panel";
import { Assessment, type AssessmentQuestion } from "@/components/assessment";
import { TrainingStatus } from "@/components/training-status";
import { apiPost } from "@/lib/api-client";
import { useTrainingRunner } from "@/hooks/use-training-runner";
import type { PlotlyFigure } from "@/components/chart-panel";

interface M17TrainResponse {
  figures?: Record<string, PlotlyFigure>;
  data?: { chunk_size?: number; top_k?: number; pipeline_status?: string };
}

interface M17QueryResponse {
  answer?: string;
  sources?: Array<{ chunk: string; score: number }>;
}

const ASSESSMENT_QUESTIONS: AssessmentQuestion[] = [
  {
    id: "q1",
    type: "conceptual",
    question: "How does chunk size affect retrieval quality?",
    options: [
      "Larger chunks = more precise retrieval",
      "Smaller chunks = more precise but may miss context",
      "Chunk size has no effect",
      "Chunk size only affects speed",
    ],
    modelAnswer:
      "Smaller chunks (e.g., 200) give more precise retrieval but may miss broader context. Larger chunks (1000+) provide context but dilute relevance — the LLM gets confused by irrelevant information within the chunk.",
  },
  {
    id: "q2",
    type: "applied",
    question: "What happens when no relevant chunks exist in RAG?",
    options: [
      "The LLM always says 'I don't have data on this'",
      "The LLM often hallucinates instead",
      "The query fails with an error",
      "RAG automatically expands the search",
    ],
    modelAnswer:
      "The LLM should say 'I don't have data on this' but often hallucinates instead. This is a critical failure mode — you need validation before RAG answers reach business users.",
  },
  {
    id: "q3",
    type: "critical",
    question: "RAG reduces hallucination but doesn't eliminate it. How would you validate RAG answers before they reach business users?",
    modelAnswer:
      "Options include: source attribution (show which chunks supported the answer), confidence thresholds, human review for high-stakes decisions, automated fact-checking against known data, and A/B testing with human evaluation.",
  },
];

export default function M17RagPage() {
  const [chunkSize, setChunkSize] = useState(500);
  const [topK, setTopK] = useState(5);
  const [query, setQuery] = useState("");
  const [relevanceFigure, setRelevanceFigure] = useState<PlotlyFigure | null>(null);
  const [answer, setAnswer] = useState<string | null>(null);
  const [sources, setSources] = useState<Array<{ chunk: string; score: number }>>([]);

  const { status, error, run } = useTrainingRunner<M17TrainResponse | M17QueryResponse>();
  const ragOffline = (answer ?? "").toLowerCase().includes("offline");

  const train = useCallback(async () => {
    const res = await run(() =>
      apiPost<M17TrainResponse>("/api/m17/train", {
        chunk_size: chunkSize,
        top_k: topK,
      })
    ) as M17TrainResponse | undefined;
    if (res?.figures?.primary) setRelevanceFigure(res.figures.primary as PlotlyFigure);
  }, [chunkSize, topK, run]);

  const ragQuery = useCallback(async () => {
    const res = await run(() =>
      apiPost<M17QueryResponse>("/api/m17/query", { query })
    ) as M17QueryResponse | undefined;
    if (res?.answer != null) setAnswer(res.answer);
    if (res?.sources) setSources(res.sources);
  }, [query, run]);

  useEffect(() => {
    apiPost<M17TrainResponse>("/api/m17/train", {
      chunk_size: 500,
      top_k: 5,
    })
      .then((res) => {
        if (res?.figures?.primary) setRelevanceFigure(res.figures.primary as PlotlyFigure);
      })
      .catch(() => {});
  }, []);

  return (
    <ModuleLayout
      chapterNumber={17}
      title="RAG Pipeline"
      subtitle="How do you ground an LLM's answers in your actual pricing data?"
      estimatedMinutes={25}
      learningObjectives={[
        "Retrieval-Augmented Generation",
        "Chunk size tuning",
        "Source attribution",
      ]}
      currentSlug="m17-rag"
    >
      <StorySection beat="FRAME" title="Context for the LLM">
        <div className="space-y-4 text-slate-500">
          <p>
            An LLM without context is like a pricing analyst without data — articulate but uninformed. RAG (Retrieval-Augmented Generation) connects the LLM to your actual pricing documents, reports, and data. When you ask &quot;What drove last week&apos;s volume drop?&quot;, RAG retrieves relevant chunks from your knowledge base and passes them to the LLM.
          </p>
          <p>
            RAG applies wherever domain-specific knowledge matters. A retailer grounds its AI assistant in store operations manuals and pricing policies. A supply chain system retrieves recent supplier performance data before answering procurement questions. Without RAG, the LLM is smart but uninformed — with RAG, it becomes a genuine expert system.
          </p>
          <p>
            The RAG pipeline has five steps. <strong className="text-slate-700">Chunk</strong> documents into overlapping segments (e.g., 200–1000 tokens). <strong className="text-slate-700">Embed</strong> each chunk into a vector using a sentence encoder. <strong className="text-slate-700">Store</strong> vectors in a vector database (e.g., Milvus, Pinecone). <strong className="text-slate-700">Retrieve</strong> the top-K most similar chunks for each query via cosine similarity. <strong className="text-slate-700">Inject</strong> those chunks into the LLM prompt as context, so the model answers from your data rather than from training memory.
          </p>
        </div>
      </StorySection>

      <StorySection beat="EXPLORE" title="Chunk relevance scores">
        <div className="space-y-4">
          <p className="text-slate-500">
            The chart below shows how well each document chunk matches your query. Chunks are ranked by cosine similarity — a measure of semantic closeness. Higher scores mean the chunk is more likely to contain the information the LLM needs to answer accurately. Poor retrieval (low scores across the board) means the LLM will either guess or refuse; strong retrieval (high scores) gives it a solid foundation.
          </p>
          <ChartPanel
            figure={relevanceFigure}
            title="Retrieved Chunk Relevance Scores"
            loading={!relevanceFigure}
          />
          <GuidedInsight type="notice">
            Each bar shows cosine similarity between your query and a document chunk. Higher similarity means more relevant context.
          </GuidedInsight>
        </div>
      </StorySection>

      <StorySection beat="BUILD" title="RAG pipeline controls">
        <div className="space-y-4">
          <p className="text-slate-500">
            <strong className="text-slate-700">Chunk size</strong> trades precision vs context. Smaller chunks (200–300) give more precise retrieval — the right sentence lands in the prompt — but may fragment ideas that span paragraphs. Larger chunks (800–1000) preserve context but dilute relevance; the LLM may get distracted by tangential content. <strong className="text-slate-700">Top-K</strong> controls how many chunks are injected: too few (e.g., 2) may miss critical information; too many (e.g., 10) can overwhelm the prompt and confuse the model. For pricing Q&amp;A, 5–7 chunks is often a good starting point.
          </p>
          {ragOffline && (
            <GuidedInsight type="notice">
              RAG is currently in graceful offline mode in this environment. The UI remains usable and clearly indicates that retrieval dependencies are unavailable.
            </GuidedInsight>
          )}
          <div className="flex flex-wrap gap-6">
            <label className="flex flex-col gap-1">
              <span className="text-xs text-slate-500">Chunk size (200–1000)</span>
              <input
                type="range"
                min={200}
                max={1000}
                step={100}
                value={chunkSize}
                onChange={(e) => setChunkSize(Number(e.target.value))}
                className="w-40"
              />
              <span className="text-xs text-slate-500">{chunkSize}</span>
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-xs text-slate-500">Top-K (3–10)</span>
              <input
                type="range"
                min={3}
                max={10}
                step={1}
                value={topK}
                onChange={(e) => setTopK(Number(e.target.value))}
                className="w-40"
              />
              <span className="text-xs text-slate-500">{topK}</span>
            </label>
            <button
              onClick={train}
              disabled={status === "running"}
              className="mt-2 rounded-lg bg-amber-500 px-4 py-2 text-sm font-medium text-slate-900 hover:bg-amber-400 disabled:opacity-50"
            >
              Update pipeline
            </button>
          </div>
          <ChartPanel
            figure={relevanceFigure}
            title="Chunk relevance"
            loading={status === "running"}
          />
          <div className="flex gap-2">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="RAG query (e.g., What drove last week's volume drop?)"
              className="flex-1 rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm text-slate-600 placeholder-slate-400"
            />
            <button
              onClick={ragQuery}
              disabled={status === "running"}
              className="rounded-lg bg-amber-500 px-4 py-2 text-sm font-medium text-slate-900 hover:bg-amber-400 disabled:opacity-50"
            >
              Query
            </button>
          </div>
          <TrainingStatus status={status} error={error} />
          {answer && (
            <div className="rounded-xl border border-slate-200 bg-slate-50 p-4">
              <h3 className="mb-2 text-sm font-semibold uppercase tracking-wider text-slate-500">
                Answer
              </h3>
              <p className="whitespace-pre-wrap text-sm text-slate-600">{answer}</p>
              {sources.length > 0 && (
                <div className="mt-4 border-t border-slate-200 pt-4">
                  <h4 className="mb-2 text-xs font-semibold uppercase tracking-wider text-slate-500">
                    Sources
                  </h4>
                  <ul className="space-y-2">
                    {sources.map((s, i) => (
                      <li key={i} className="text-xs text-slate-500">
                        <span className="text-amber-600">({(s.score * 100).toFixed(0)}%)</span>{" "}
                        {s.chunk}...
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}
        </div>
      </StorySection>

      <StorySection beat="INTERROGATE" title="Chunk size and hallucination">
        <div className="space-y-4">
          <GuidedInsight type="try">
            Set chunk_size to 200. Retrieval is more precise but may miss broader context.
          </GuidedInsight>
          <GuidedInsight type="warning">
            Large chunks (1000+) provide context but dilute relevance — the LLM gets confused by irrelevant information within the chunk.
          </GuidedInsight>
          <GuidedInsight type="think">
            What happens when no relevant chunks exist? The LLM should say &quot;I don&apos;t have data on this&quot; but often hallucinates instead.
          </GuidedInsight>
        </div>
      </StorySection>

      <StorySection beat="DECIDE" title="Validating RAG answers">
        <div className="space-y-4">
          <p className="text-slate-500">
            Deciding what to include in the knowledge base shapes RAG quality. Include: pricing policies, competitor playbooks, historical reports, product catalogs, and any document analysts routinely consult. Exclude: stale data, confidential material that shouldn&apos;t be surfaced, or formats the chunker handles poorly (e.g., dense tables). Maintenance matters: as policies change, competitors shift, and new products launch, the knowledge base must be updated. A RAG system built on last year&apos;s data will give last year&apos;s answers.
          </p>
          <p className="text-slate-500">
            RAG reduces hallucination but doesn&apos;t eliminate it. How would you validate RAG answers before they reach business users?
          </p>
          <Assessment questions={ASSESSMENT_QUESTIONS} />
        </div>
      </StorySection>
    </ModuleLayout>
  );
}
