import { api } from "./client";
import type { Account, Dataset, Job, Notebook, PageResult } from "../types/kaggle";
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
  datasets: (page = 1, pageSize = 100, activeOnly = false) => api<PageResult<Dataset>>(`/api/kaggle/datasets?page=${page}&page_size=${pageSize}&active_only=${activeOnly}`),
  createDataset: (body: { slug: string; dataset_ref: string; title?: string; description?: string; local_path?: string; status?: string }) =>
    api<Dataset>("/api/kaggle/datasets", { method: "POST", body: JSON.stringify(body) }),
  updateDataset: (id: string, body: { slug?: string; dataset_ref?: string; title?: string; description?: string; local_path?: string; status?: string; is_active?: boolean }) =>
    api<Dataset>(`/api/kaggle/datasets/id/${encodeURIComponent(id)}`, { method: "PATCH", body: JSON.stringify(body) }),
  deleteDataset: (id: string) =>
    api(`/api/kaggle/datasets/id/${encodeURIComponent(id)}`, { method: "DELETE" }),
  validateDataset: (id: string) =>
    api<Dataset>(`/api/kaggle/datasets/id/${encodeURIComponent(id)}/validate-local`, { method: "POST" }),
  inventory: (page = 1, pageSize = 20) => api<PageResult<Notebook>>(`/api/kaggle/notebooks/inventory?page=${page}&page_size=${pageSize}`),
  validateNotebook: (slug: string) =>
    api<Notebook>(`/api/kaggle/notebooks/${slug}/validate`, { method: "POST" }),
  trigger: (body: { account?: string; accounts?: string[]; notebook_id: string; dataset_source: string }) =>
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
