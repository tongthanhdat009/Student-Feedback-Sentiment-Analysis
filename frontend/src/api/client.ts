const API_BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000";
const API_KEY = import.meta.env.VITE_ADMIN_API_KEY ?? "change-me";
export async function api<T>(path: string, init: RequestInit = {}): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      "X-API-Key": API_KEY,
      ...init.headers,
    },
  });
  if (!res.ok) {
    const text = await res.text();
    let message = text || `${res.status} ${res.statusText}`;
    try {
      const parsed = JSON.parse(text) as { detail?: unknown; error?: unknown };
      message = String(parsed.detail ?? parsed.error ?? message);
    } catch {
      // Keep raw text fallback.
    }
    throw new Error(message);
  }
  return res.json();
}
