const BASE = (import.meta.env.VITE_API_URL ?? "http://localhost:8000").replace(/\/$/, "");
const TOKEN_KEY = "fa_token";

export const auth = {
  get: () => { try { return localStorage.getItem(TOKEN_KEY); } catch { return null; } },
  set: (t: string) => { try { localStorage.setItem(TOKEN_KEY, t); } catch { /* private mode */ } },
  clear: () => { try { localStorage.removeItem(TOKEN_KEY); } catch { /* ignore */ } },
};

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) { super(message); this.status = status; }
}

export async function api<T>(path: string, init: { method?: string; body?: unknown } = {}): Promise<T> {
  const token = auth.get();
  let res: Response;
  try {
    res = await fetch(`${BASE}/api${path}`, {
      method: init.method ?? (init.body ? "POST" : "GET"),
      headers: { "Content-Type": "application/json", ...(token ? { Authorization: `Bearer ${token}` } : {}) },
      body: init.body ? JSON.stringify(init.body) : undefined,
    });
  } catch {
    throw new ApiError(0, "Can't reach the server. The free server sleeps when idle; wait a minute and try again.");
  }
  if (res.status === 204) return undefined as T;
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const d = data?.detail;
    throw new ApiError(res.status, typeof d === "string" ? d : "Something went wrong. Please check your input.");
  }
  return data as T;
}

export type Product = {
  asin: string; type: string; title: string; brand?: string; image?: string | null;
  colour_words?: string; rgb: number[]; harmony: number;
};
export type PlanLine = {
  item: string; tier: string; estimated_price_inr: number; typical_range_inr: number[]; products: Product[];
};
export type Plan = {
  room_type: string; missing: string[]; plan: PlanLine[]; not_in_budget: string[];
  estimated_total_inr: number; price_note: string;
};
export type Pick = {
  id: number; item_type: string; asin: string; title: string; image: string | null; rgb: number[] | null;
  price_inr: number | null; added_by: number; my_vote: number | null; votes_in: number;
  mean: number | null; consensus: number | null; vetoes: number;
};
export type BoardSummary = {
  id: number; name: string; room_type: string; wall_rgb: number[]; budget_inr: number;
  member_count: number; pick_count: number;
};
export type Board = BoardSummary & {
  invite_code: string | null; members: { user_id: number; name: string; role: string }[]; picks: Pick[];
};
export type Decision = {
  items: { item_type: string; pick: Pick; contested: boolean; estimated_price_inr: number; price_is_estimate: boolean }[];
  estimated_total_inr: number; budget_inr: number; within_budget: boolean; over_by_inr: number;
};
