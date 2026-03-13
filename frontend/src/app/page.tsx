"use client";

import Link from "next/link";
import { useStore } from "@/lib/store";
import { cn } from "@/lib/utils";
import {
  BookOpen, Target, BarChart3, Layers, Search, Clock,
  TrendingUp, Brain, Zap, Gauge, Dices, Gamepad2, Bot,
  Network, Cpu, Sparkles, MessageSquare, GitMerge, Flag,
} from "lucide-react";

const PARTS = [
  {
    part: "I",
    label: "Foundation",
    description: "Build your data intuition before any modeling",
    modules: [
      {
        slug: "m00-foundations",
        id: "M00",
        title: "The Bias-Variance Tradeoff",
        hook: "Why does adding more complexity sometimes make predictions worse?",
        minutes: 15,
        icon: BookOpen,
      },
    ],
  },
  {
    part: "II",
    label: "Supervised Learning",
    description: "Teach machines to predict with labeled pricing data",
    modules: [
      {
        slug: "m01-regression",
        id: "M01",
        title: "Price Elasticity",
        hook: "When does cutting your price actually pay off in volume?",
        minutes: 25,
        icon: Target,
        prereqs: ["M00"],
      },
      {
        slug: "m02-classification",
        id: "M02",
        title: "Threat Detection",
        hook: "Can you predict which competitor moves will cost you 15% of volume?",
        minutes: 20,
        icon: BarChart3,
        prereqs: ["M01"],
      },
      {
        slug: "m03-ensembles",
        id: "M03",
        title: "Ensemble Intelligence",
        hook: "What if combining weak models creates a strong one?",
        minutes: 25,
        icon: Layers,
        prereqs: ["M01", "M02"],
      },
    ],
  },
  {
    part: "III",
    label: "Unsupervised Learning",
    description: "Discover hidden structure in your locations, products, and customers",
    modules: [
      {
        slug: "m04-clustering",
        id: "M04",
        title: "Market Segmentation",
        hook: "Are your existing product or location labels really the best grouping?",
        minutes: 20,
        icon: Search,
        prereqs: ["M00"],
      },
      {
        slug: "m05-anomaly",
        id: "M05",
        title: "Anomaly Detection",
        hook: "Which sales drops are natural and which signal a problem you need to act on?",
        minutes: 15,
        icon: Zap,
        prereqs: ["M04"],
      },
    ],
  },
  {
    part: "IV",
    label: "Time Series",
    description: "Forecast demand across hours, days, and seasons",
    modules: [
      {
        slug: "m06-timeseries",
        id: "M06",
        title: "Classical Forecasting",
        hook: "How far ahead can ARIMA see, and when does it go blind?",
        minutes: 25,
        icon: Clock,
        prereqs: ["M01"],
      },
      {
        slug: "m07-sequence",
        id: "M07",
        title: "Sequence Models",
        hook: "Can yesterday's lag features outperform a full ARIMA model?",
        minutes: 20,
        icon: TrendingUp,
        prereqs: ["M06"],
      },
      {
        slug: "m08-tft",
        id: "M08",
        title: "Temporal Fusion Transformer",
        hook: "What if a model could tell you which inputs it's paying attention to?",
        minutes: 25,
        icon: Brain,
        prereqs: ["M07"],
      },
    ],
  },
  {
    part: "V",
    label: "Optimization & RL",
    description: "From finding the best price to learning pricing strategies",
    modules: [
      {
        slug: "m09-optimization",
        id: "M09",
        title: "Price Optimization",
        hook: "Given constraints, what's the mathematically optimal price for each product or location?",
        minutes: 20,
        icon: Gauge,
        prereqs: ["M01"],
      },
      {
        slug: "m10-bandits",
        id: "M10",
        title: "Price Experimentation",
        hook: "Should you exploit what works or explore what might work better?",
        minutes: 20,
        icon: Dices,
        prereqs: ["M09"],
      },
      {
        slug: "m11-qlearning",
        id: "M11",
        title: "Dynamic Pricing",
        hook: "Can an agent learn a pricing policy by trial and error?",
        minutes: 25,
        icon: Gamepad2,
        prereqs: ["M10"],
      },
      {
        slug: "m12-dqn",
        id: "M12",
        title: "Deep Reinforcement Learning",
        hook: "What happens when you give the agent a neural network for a brain?",
        minutes: 25,
        icon: Bot,
        prereqs: ["M11"],
      },
    ],
  },
  {
    part: "VI",
    label: "Neural & Transformers",
    description: "Deep learning from MLPs to attention mechanisms",
    modules: [
      {
        slug: "m13-neural",
        id: "M13",
        title: "Neural Networks",
        hook: "How does a neural network decide whether to cut price or hold?",
        minutes: 25,
        icon: Network,
        prereqs: ["M03"],
      },
      {
        slug: "m14-ft-transformer",
        id: "M14",
        title: "FT-Transformer",
        hook: "Can attention on tabular data beat gradient boosting?",
        minutes: 20,
        icon: Cpu,
        prereqs: ["M13"],
      },
      {
        slug: "m15-transformer-zoo",
        id: "M15",
        title: "Transformer Architectures",
        hook: "Encoder, decoder, encoder-decoder: what's the difference for pricing?",
        minutes: 15,
        icon: Sparkles,
        prereqs: ["M14"],
      },
    ],
  },
  {
    part: "VII",
    label: "LLM & RAG",
    description: "Large language models applied to pricing intelligence",
    modules: [
      {
        slug: "m16-llm",
        id: "M16",
        title: "LLM Capabilities",
        hook: "What can a 9-billion-parameter model understand about pricing?",
        minutes: 20,
        icon: MessageSquare,
        prereqs: ["M15"],
      },
      {
        slug: "m17-rag",
        id: "M17",
        title: "RAG Pipeline",
        hook: "How do you ground an LLM's answers in your actual pricing data?",
        minutes: 25,
        icon: GitMerge,
        prereqs: ["M16"],
      },
    ],
  },
  {
    part: "VIII",
    label: "Synthesis",
    description: "Putting it all together into a production system",
    modules: [
      {
        slug: "m18-synthesis",
        id: "M18",
        title: "System Design",
        hook: "How do you ship 18 models into a system a VP would approve?",
        minutes: 30,
        icon: Flag,
        prereqs: ["M17"],
      },
    ],
  },
];

const STATUS_LABELS = {
  not_started: { label: "Not started", color: "bg-slate-200" },
  visited: { label: "Visited", color: "bg-amber-500" },
  trained: { label: "Model trained", color: "bg-blue-500" },
  assessed: { label: "Assessed", color: "bg-violet-500" },
  passed: { label: "Completed", color: "bg-emerald-500" },
} as const;

export default function LearningJourney() {
  const getProgress = useStore((s) => s.getProgress);

  const completedCount = PARTS.flatMap((p) => p.modules).filter(
    (m) => getProgress(m.slug) === "passed"
  ).length;
  const totalModules = PARTS.flatMap((p) => p.modules).length;

  return (
    <div className="min-h-screen bg-white text-slate-900 pt-14">
      {/* Hero */}
      <header className="border-b border-slate-200 bg-gradient-to-b from-slate-50 to-white px-6 py-14">
        <div className="mx-auto max-w-4xl">
          <h1 className="text-4xl font-bold tracking-tight md:text-5xl">
            Pricing Intelligence Lab
          </h1>
          <p className="mt-4 max-w-2xl text-lg text-slate-500 leading-relaxed">
            A guided journey through machine learning, optimization, and AI — told through
            the lens of real-world pricing decisions across fuel, convenience retail, and
            supply chain. No prior ML experience required; deep business experience assumed.
          </p>

          {/* Who this is for */}
          <div className="mt-6 rounded-xl border border-slate-200 bg-white p-5">
            <h2 className="text-sm font-semibold text-slate-700 uppercase tracking-wider mb-2">
              Who this is for
            </h2>
            <p className="text-sm text-slate-600 leading-relaxed">
              For pricing analysts, category managers, supply chain leaders, and revenue
              strategists who want to understand machine learning without becoming data
              scientists. Each module connects a core ML concept to a pricing decision you
              already make — so you know exactly when (and when not) to deploy these techniques.
            </p>
          </div>
          <div className="mt-6 flex items-center gap-4">
            <div className="flex-1 h-2 rounded-full bg-white overflow-hidden">
              <div
                className="h-full rounded-full bg-gradient-to-r from-amber-500 to-emerald-500 transition-all duration-500"
                style={{ width: `${(completedCount / totalModules) * 100}%` }}
              />
            </div>
            <span className="text-sm text-slate-500 tabular-nums">
              {completedCount}/{totalModules} chapters
            </span>
          </div>

          <div className="mt-6 flex flex-wrap gap-3">
            {Object.entries(STATUS_LABELS).map(([key, val]) => (
              <div key={key} className="flex items-center gap-2 text-xs text-slate-500">
                <span className={cn("h-2 w-2 rounded-full", val.color)} />
                {val.label}
              </div>
            ))}
          </div>
        </div>
      </header>

      {/* Journey Map */}
      <main className="mx-auto max-w-4xl px-6 py-10">
        {PARTS.map((part, partIdx) => (
          <section key={part.part} className="relative mb-12">
            {/* Vertical connector line */}
            {partIdx < PARTS.length - 1 && (
              <div className="absolute left-5 top-12 bottom-0 w-px bg-slate-200" aria-hidden />
            )}

            {/* Part header */}
            <div className="mb-5 flex items-baseline gap-3">
              <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-white text-sm font-bold text-amber-500 ring-2 ring-slate-200">
                {part.part}
              </span>
              <div>
                <h2 className="text-xl font-bold text-slate-900">{part.label}</h2>
                <p className="text-sm text-slate-500">{part.description}</p>
              </div>
            </div>

            {/* Module cards */}
            <div className="ml-5 border-l border-slate-200 pl-8 space-y-3">
              {part.modules.map((mod) => {
                const progress = getProgress(mod.slug);
                const Icon = mod.icon;
                const statusInfo = STATUS_LABELS[progress] ?? STATUS_LABELS.not_started;
                return (
                  <Link
                    key={mod.slug}
                    href={`/modules/${mod.slug}`}
                    className="group block rounded-xl border border-slate-200 bg-slate-50/50 p-5 transition-all hover:border-slate-300 hover:bg-slate-50"
                  >
                    <div className="flex items-start gap-4">
                      <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-white text-slate-500 group-hover:text-amber-600 transition-colors">
                        <Icon className="h-5 w-5" />
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-3 mb-1">
                          <span className="font-mono text-xs text-slate-500">{mod.id}</span>
                          <h3 className="font-semibold text-slate-900 group-hover:text-amber-600 transition-colors">
                            {mod.title}
                          </h3>
                          <span className={cn("ml-auto h-2 w-2 rounded-full shrink-0", statusInfo.color)} />
                        </div>
                        <p className="text-sm text-slate-500 italic">"{mod.hook}"</p>
                        <div className="mt-2 flex items-center gap-3 text-xs text-slate-500">
                          <span>~{mod.minutes} min</span>
                          {"prereqs" in mod && mod.prereqs && (
                            <span>
                              Prereqs: {mod.prereqs.join(", ")}
                            </span>
                          )}
                        </div>
                      </div>
                    </div>
                  </Link>
                );
              })}
            </div>
          </section>
        ))}
      </main>
    </div>
  );
}
