declare module "react-plotly.js" {
  import type { ComponentType } from "react";
  interface PlotParams {
    data: unknown[];
    layout?: Record<string, unknown>;
    frames?: unknown[];
    config?: Record<string, unknown>;
    useResizeHandler?: boolean;
    style?: React.CSSProperties;
    className?: string;
  }
  const Plot: ComponentType<PlotParams>;
  export default Plot;
}
