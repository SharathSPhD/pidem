const BASE =
  (typeof process !== "undefined" && process.env?.NEXT_PUBLIC_API_URL) ||
  (typeof import.meta !== "undefined" && (import.meta as unknown as { env?: { VITE_API_URL?: string } }).env?.VITE_API_URL) ||
  (typeof window !== "undefined" && (window as unknown as { __API_BASE__?: string }).__API_BASE__) ||
  "http://localhost:8000";

export async function apiGet<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) throw new Error(`GET ${path}: ${res.statusText}`);
  return res.json();
}

export async function apiPost<T = unknown>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) throw new Error(`POST ${path}: ${res.statusText}`);
  return res.json();
}

export type ChartFigure = { data: unknown[]; layout?: Record<string, unknown> };
export type ChartsResponse = { charts?: Record<string, ChartFigure> };
export type FigureResponse = { figure?: ChartFigure };
