import { api } from "./client";
import type { Account, Job, Notebook, PageResult } from "../types/kaggle";
export const kaggleApi = {
  health: () => api<{ ok: boolean }>("/api/kaggle/health", { headers: {} }),
  accounts: (page = 1, pageSize = 20) => api<PageResult<Account>>(`/api/kaggle/accounts?page=${page}&page_size=${pageSize}`),
  createAccount: (body: {
    name: string;
    kaggle_username: string;
    kaggle_key: string;
  }) =>
    api<Account>("/api/kaggle/accounts", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  updateAccount: (name: string, body: { name?: string; kaggle_username?: string; kaggle_key?: string; is_active?: boolean }) =>
    api<Account>(`/api/kaggle/accounts/${encodeURIComponent(name)}`, { method: "PATCH", body: JSON.stringify(body) }),
  deleteAccount: (name: string) =>
    api(`/api/kaggle/accounts/${encodeURIComponent(name)}`, { method: "DELETE" }),
  testAccount: (name: string) =>
    api(`/api/kaggle/accounts/${encodeURIComponent(name)}/test`, { method: "POST" }),
  inventory: (page = 1, pageSize = 20) => api<PageResult<Notebook>>(`/api/kaggle/notebooks/inventory?page=${page}&page_size=${pageSize}`),
  validateNotebook: (slug: string) =>
    api<Notebook>(`/api/kaggle/notebooks/${slug}/validate`, { method: "POST" }),
  trigger: (body: { account?: string; accounts?: string[]; notebook_id: string }) =>
    api<Job | Job[]>("/api/kaggle/notebooks/trigger", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  jobs: (page = 1, pageSize = 20) => api<PageResult<Job>>(`/api/kaggle/jobs?page=${page}&page_size=${pageSize}`),
  download: (id: string) =>
    api<Job>(`/api/kaggle/jobs/${id}/download-output`, { method: "POST" }),
  artifactUrl: (id: string) =>
    api<{ url: string; expires_at: string }>(
      `/api/kaggle/jobs/${id}/artifact-url`,
    ),
};
