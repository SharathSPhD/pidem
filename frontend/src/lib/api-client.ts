const BASE = "http://localhost:8000";

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
