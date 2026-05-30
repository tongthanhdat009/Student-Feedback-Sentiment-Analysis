import { api } from "./client";
import type { Account, Job, Notebook } from "../types/kaggle";
export const kaggleApi = {
  health: () => api<{ ok: boolean }>("/api/kaggle/health", { headers: {} }),
  accounts: () => api<Account[]>("/api/kaggle/accounts"),
  createAccount: (body: {
    name: string;
    kaggle_username: string;
    kaggle_key: string;
  }) =>
    api<Account>("/api/kaggle/accounts", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  deleteAccount: (name: string) =>
    api(`/api/kaggle/accounts/${name}`, { method: "DELETE" }),
  testAccount: (name: string) =>
    api(`/api/kaggle/accounts/${name}/test`, { method: "POST" }),
  inventory: () => api<Notebook[]>("/api/kaggle/notebooks/inventory"),
  trigger: (body: { account: string; notebook_id: string }) =>
    api<Job>("/api/kaggle/notebooks/trigger", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  jobs: () => api<Job[]>("/api/kaggle/jobs"),
  download: (id: string) =>
    api<Job>(`/api/kaggle/jobs/${id}/download-output`, { method: "POST" }),
  artifactUrl: (id: string) =>
    api<{ url: string; expires_at: string }>(
      `/api/kaggle/jobs/${id}/artifact-url`,
    ),
};
