"use client";

import { useState, useEffect } from "react";
import { ModuleLayout } from "@/components/module-layout";
import { StorySection } from "@/components/story-section";
import { GuidedInsight } from "@/components/guided-insight";
import { Assessment, type AssessmentQuestion } from "@/components/assessment";
import { apiGet } from "@/lib/api-client";

interface Architecture {
  name: string;
  type: string;
  key_innovation: string;
  pricing_use: string;
}

interface M15ArchitecturesResponse {
  architectures: Architecture[];
}

const ASSESSMENT_QUESTIONS: AssessmentQuestion[] = [
  {
    id: "q1",
    type: "conceptual",
    question: "What is the main difference between self-attention and cross-attention?",
    options: [
      "Self-attention is faster; cross-attention is more accurate",
      "Self-attention attends within one sequence; cross-attention attends from decoder to encoder",
      "Self-attention is bidirectional; cross-attention is unidirectional",
      "Both are identical",
    ],
    modelAnswer:
      "Self-attention attends within a single sequence (e.g., encoder tokens to all encoder tokens). Cross-attention is used in encoder-decoder models: the decoder attends to the encoder output to incorporate context.",
  },
  {
    id: "q2",
    type: "applied",
    question: "If you need to classify pricing emails as 'urgent' vs 'routine,' which architecture would you choose?",
    options: ["BERT (encoder)", "GPT (decoder)", "T5 (encoder-decoder)", "Any of the above"],
    modelAnswer:
      "BERT (encoder). Classification is a discriminative task — you need to understand the full context of the email. BERT's bidirectional attention reads the whole text holistically. GPT's decoder-only design is optimized for generation, not classification.",
  },
  {
    id: "q3",
    type: "critical",
    question: "Your company wants to deploy one model for all NLP tasks. Would you choose an encoder (BERT), decoder (GPT), or encoder-decoder (T5)?",
    options: [
      "BERT — best for understanding",
      "GPT — simpler to scale, emergent capabilities",
      "T5 — most flexible",
      "Depends on primary use case",
    ],
    modelAnswer:
      "Decoder (GPT) has become dominant because it's simpler to scale and train. A single decoder can do classification (via prompting), generation, summarization, and more. BERT excels at understanding but can't generate. T5 is flexible but heavier. Most production systems today use decoder-only models.",
  },
];

export default function M15TransformerZooPage() {
  const [architectures, setArchitectures] = useState<Architecture[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedArch, setSelectedArch] = useState<Architecture | null>(null);

  useEffect(() => {
    apiGet<M15ArchitecturesResponse>("/api/m15/architectures")
      .then((res) => {
        setArchitectures(res.architectures ?? []);
        if (res.architectures?.length) setSelectedArch(res.architectures[0]);
      })
      .catch(() => setArchitectures([]))
      .finally(() => setLoading(false));
  }, []);

  return (
    <ModuleLayout
      chapterNumber={15}
      title="Transformer Architectures"
      subtitle="Encoder, decoder, encoder-decoder: what's the difference for pricing?"
      estimatedMinutes={15}
      learningObjectives={[
        "Transformer taxonomy",
        "Self-attention vs cross-attention",
        "Architecture selection",
      ]}
      currentSlug="m15-transformer-zoo"
    >
      <StorySection beat="FRAME" title="The transformer family">
        <div className="space-y-4 text-slate-500">
          <p>
            Before diving into LLMs (M16), you need to understand the transformer family tree. These
            architectures process text differently and suit different pricing tasks. BERT (encoder)
            reads text bidirectionally — good for understanding pricing reports. GPT (decoder)
            generates text left-to-right — good for writing pricing narratives. T5 (encoder-decoder)
            translates input to output — good for summarization.
          </p>
          <p>
            In pricing operations, you encounter all three: encoding (understanding market reports,
            classifying customer feedback), decoding (generating pricing memos, drafting
            recommendations), and seq2seq (translating competitor filings into structured data,
            summarizing long contracts). Choosing the right architecture saves compute and improves
            results.
          </p>
          <p>
            Think of it through pricing analogies. An encoder is like a reader who absorbs a full
            market report and extracts key facts — no generation, just understanding. A decoder is
            like a writer who composes a pricing memo from scratch, one word at a time. An
            encoder-decoder is like a translator who reads a competitor filing and produces a
            structured summary — input and output are different formats.
          </p>
        </div>
      </StorySection>

      <StorySection beat="EXPLORE" title="Architecture comparison">
        <div className="space-y-4">
          <p className="text-slate-500">
            The table below summarizes each architecture&apos;s type, key innovation, and pricing
            use case. Encoders use self-attention within a single sequence; decoders use
            masked self-attention (each token sees only prior tokens); encoder-decoder models add
            cross-attention so the decoder attends to the encoder output. The choice depends on
            whether your task is understanding, generation, or transformation.
          </p>
          {loading ? (
            <div className="flex min-h-[200px] items-center justify-center rounded-lg border border-slate-200 bg-slate-50">
              <span className="text-slate-500">Loading architectures...</span>
            </div>
          ) : (
            <div className="overflow-x-auto rounded-lg border border-slate-200">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-slate-200 bg-slate-50">
                    <th className="px-4 py-3 text-left font-medium text-slate-600">Architecture</th>
                    <th className="px-4 py-3 text-left font-medium text-slate-600">Type</th>
                    <th className="px-4 py-3 text-left font-medium text-slate-600">Key Innovation</th>
                    <th className="px-4 py-3 text-left font-medium text-slate-600">Pricing Use</th>
                  </tr>
                </thead>
                <tbody>
                  {architectures.map((a) => (
                    <tr
                      key={a.name}
                      className="border-b border-slate-200 hover:bg-slate-50"
                    >
                      <td className="px-4 py-3 font-medium text-slate-700">{a.name}</td>
                      <td className="px-4 py-3 text-slate-500">{a.type}</td>
                      <td className="px-4 py-3 text-slate-500">{a.key_innovation}</td>
                      <td className="px-4 py-3 text-slate-500">{a.pricing_use}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          <GuidedInsight type="notice">
            Each architecture has a different inductive bias. BERT can&apos;t generate; GPT can&apos;t encode context holistically; T5 can do both but is heavier.
          </GuidedInsight>
        </div>
      </StorySection>

      <StorySection beat="BUILD" title="Interactive architecture explorer">
        <div className="space-y-4">
          <p className="text-slate-500">
            Select an architecture to see its key properties, use cases, and pricing applications.
            BERT suits classification and extraction — e.g., sentiment on pricing feedback, entity
            extraction from contracts. GPT suits generation — pricing narratives, Q&amp;A, few-shot
            prompts. T5 suits transformation — summarization, translation, structured output from
            unstructured text. Match the architecture to the task.
          </p>
          <div className="grid gap-4 sm:grid-cols-3">
            {architectures.map((a) => (
              <button
                key={a.name}
                onClick={() => setSelectedArch(a)}
                className={`rounded-xl border p-4 text-left transition-all ${
                  selectedArch?.name === a.name
                    ? "border-amber-500 bg-amber-500/10 ring-2 ring-amber-500/50"
                    : "border-slate-200 bg-slate-50 hover:border-slate-200"
                }`}
              >
                <h3 className="font-semibold text-slate-700">{a.name}</h3>
                <span className="mt-1 inline-block rounded bg-slate-100 px-2 py-0.5 text-xs text-slate-500">
                  {a.type}
                </span>
              </button>
            ))}
          </div>
          {selectedArch && (
            <div className="rounded-xl border border-slate-200 bg-slate-50 p-6">
              <h3 className="mb-2 text-lg font-semibold text-slate-700">
                {selectedArch.name} — {selectedArch.type}
              </h3>
              <dl className="space-y-2 text-sm">
                <div>
                  <dt className="text-slate-500">Key innovation</dt>
                  <dd className="text-slate-600">{selectedArch.key_innovation}</dd>
                </div>
                <div>
                  <dt className="text-slate-500">Pricing application</dt>
                  <dd className="text-slate-600">{selectedArch.pricing_use}</dd>
                </div>
              </dl>
            </div>
          )}
        </div>
      </StorySection>

      <StorySection beat="INTERROGATE" title="Architecture selection">
        <div className="space-y-4">
          <p className="text-slate-500">
            Business examples clarify the choice. Classifying pricing emails as &quot;urgent&quot; vs
            &quot;routine&quot; requires understanding the full context — use an encoder. Generating
            a pricing memo from bullet points requires left-to-right generation — use a decoder.
            Converting a competitor filing into a structured table requires reading input and
            producing different output — use encoder-decoder. For many tasks, a single decoder can
            do classification via prompting, but it&apos;s less efficient than a dedicated encoder.
          </p>
          <GuidedInsight type="think">
            If you need to classify pricing emails as &quot;urgent&quot; vs &quot;routine,&quot; which architecture would you choose? Hint: it&apos;s a classification task, not generation.
          </GuidedInsight>
          <GuidedInsight type="notice">
            The decoder-only architecture (GPT) has become dominant because it&apos;s simpler to scale and train.
          </GuidedInsight>
        </div>
      </StorySection>

      <StorySection beat="DECIDE" title="One model for all NLP tasks?">
        <div className="space-y-4">
          <p className="text-slate-500">
            Architecture selection guidance: if you have one primary task (e.g., classification),
            a specialized encoder may be more efficient. If you need flexibility (classification,
            generation, summarization, Q&amp;A) and prefer a single model, decoder-only is the
            pragmatic choice — it scales well and exhibits emergent capabilities. Encoder-decoder
            models are best when input and output are structurally different (e.g., long document
            in, short summary out). Your company wants to deploy one model for all NLP tasks.
            Would you choose an encoder (BERT), decoder (GPT), or encoder-decoder (T5)?
          </p>
          <Assessment questions={ASSESSMENT_QUESTIONS} />
        </div>
      </StorySection>
    </ModuleLayout>
  );
}
