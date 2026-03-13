"use client";

import dynamic from "next/dynamic";
import { cn } from "@/lib/utils";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

export interface PlotlyFigure {
  data: Plotly.Data[] | unknown[];
  layout?: Partial<Plotly.Layout> | Record<string, unknown>;
  frames?: Plotly.Frame[];
}

interface ChartPanelProps {
  figure: PlotlyFigure | null;
  title?: string;
  loading?: boolean;
  className?: string;
}

export function ChartPanel({ figure, title, loading, className }: ChartPanelProps) {
  if (loading) {
    return (
      <div
        className={cn(
          "flex min-h-[320px] flex-col rounded-xl border border-slate-200 bg-slate-50 p-4",
          className
        )}
      >
        {title && (
          <h3 className="mb-4 text-sm font-semibold uppercase tracking-wider text-slate-500">
            {title}
          </h3>
        )}
        <div className="flex flex-1 animate-pulse items-center justify-center rounded-lg bg-slate-50">
          <div className="h-8 w-48 rounded bg-slate-100" />
        </div>
      </div>
    );
  }

  if (!figure) {
    return (
      <div
        className={cn(
          "flex min-h-[320px] flex-col rounded-xl border border-slate-200 bg-slate-50 p-4",
          className
        )}
      >
        {title && (
          <h3 className="mb-4 text-sm font-semibold uppercase tracking-wider text-slate-500">
            {title}
          </h3>
        )}
        <div className="flex flex-1 items-center justify-center text-slate-500">
          No chart data
        </div>
      </div>
    );
  }

  const mergedLayout: Partial<Plotly.Layout> = {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(248, 250, 252, 0.5)",
    font: { color: "#64748b", family: "inherit" },
    margin: { t: 40, r: 20, b: 40, l: 50 },
    xaxis: { gridcolor: "rgba(203, 213, 225, 0.5)" },
    yaxis: { gridcolor: "rgba(203, 213, 225, 0.5)" },
    ...figure.layout,
  };

  return (
    <div
      className={cn(
        "flex flex-col rounded-xl border border-slate-200 bg-slate-50 p-4",
        className
      )}
    >
      {title && (
        <h3 className="mb-4 text-sm font-semibold uppercase tracking-wider text-slate-500">
          {title}
        </h3>
      )}
      <div className="relative w-full overflow-hidden" style={{ minHeight: 320 }}>
        <Plot
          data={figure.data}
          layout={mergedLayout}
          frames={figure.frames}
          config={{
            responsive: true,
            displayModeBar: true,
            displaylogo: false,
            modeBarButtonsToRemove: ["lasso2d", "select2d"],
            toImageButtonOptions: {
              format: "png",
              filename: "pidem-chart",
              scale: 2,
            },
          }}
          useResizeHandler
          className="w-full"
          style={{ width: "100%", minHeight: 320 }}
        />
      </div>
    </div>
  );
}
