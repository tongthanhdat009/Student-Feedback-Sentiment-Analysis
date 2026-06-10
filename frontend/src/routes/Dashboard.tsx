import { useCallback, useEffect, useState } from "react";
import { Play, Download, ExternalLink } from "lucide-react";
import { kaggleApi } from "../api/kaggleApi";
import type { Job } from "../types/kaggle";
import { Pager } from "../components/kaggle/Pager";
import { StatusBadge } from "../components/kaggle/StatusBadge";

import { usePolling } from "../hooks/usePolling";

const activeStatuses = new Set(["pending", "staging", "pushed", "running"]);

export function Dashboard() {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(10);
  const [loading, setLoading] = useState(true);

  const load = useCallback((nextPage = page) => {
    setLoading(true);
    kaggleApi.jobs(nextPage, pageSize).then((res) => { setJobs(res.items); setTotal(res.total); setPage(res.page); }).catch(console.error).finally(() => setLoading(false));
  }, [page, pageSize]);
  useEffect(() => { void load(1); }, [load]);
  const hasActiveJobs = jobs.some((j) => activeStatuses.has(j.status));
  usePolling(hasActiveJobs, () => load(page), 5000);

  const stats = {
    total: jobs.length,
    completed: jobs.filter(j => j.status === "completed").length,
    running: jobs.filter(j => activeStatuses.has(j.status)).length,
    failed: jobs.filter(j => j.status === "failed").length,
  };

  return (
    <div className="space-y-8">
      <div>
        <p className="text-[11px] font-mono uppercase tracking-[0.12em] mb-1" style={{ color: "hsl(var(--text-muted))" }}>Overview</p>
        <h1 className="text-2xl font-semibold tracking-tight" style={{ color: "hsl(var(--text-primary))" }}>Kaggle Jobs</h1>
      </div>

      <div className="grid grid-cols-4 gap-4">
        {[
          { label: "Total Jobs", value: stats.total, color: "hsl(var(--accent))" },
          { label: "Completed", value: stats.completed, color: "hsl(var(--success))" },
          { label: "Active", value: stats.running, color: "hsl(var(--warning))" },
          { label: "Failed", value: stats.failed, color: "hsl(var(--danger))" },
        ].map(s => (
          <div key={s.label} className="card flex flex-col gap-1">
            <span className="stat stat-value" style={{ color: s.color }}>{s.value}</span>
            <span className="stat-label">{s.label}</span>
          </div>
        ))}
      </div>

      <div className="card">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-sm font-semibold tracking-tight" style={{ color: "hsl(var(--text-primary))" }}>Recent Jobs</h2>
          <button className="btn btn-ghost text-xs" onClick={() => load(page)}>Refresh</button>
        </div>
        {loading ? (
          <div style={{ color: "hsl(var(--text-muted))" }} className="text-xs py-8 text-center">Loading jobs&hellip;</div>
        ) : jobs.length === 0 ? (
          <div className="py-12 text-center space-y-2">
            <Play size={28} style={{ color: "hsl(var(--text-muted))", margin: "0 auto" }} />
            <p style={{ color: "hsl(var(--text-muted))" }} className="text-sm">No jobs yet. Trigger a notebook to get started.</p>
          </div>
        ) : (
          <table>
            <thead>
              <tr>
                <th>Job ID</th><th>Type</th><th>Target</th><th>Kaggle Ref</th><th>Status</th><th>Message</th><th>Artifact</th><th></th>
              </tr>
            </thead>
            <tbody>
              {jobs.map(j => (
                <tr key={j.id}>
                  <td className="font-mono text-xs">{j.id.slice(0, 12)}&hellip;</td>
                  <td className="font-mono text-[11px]">{j.job_type}</td>
                  <td className="text-xs">{j.target_ref}</td>
                  <td className="font-mono text-[11px]">{j.kaggle_ref ?? <span style={{ color: "hsl(var(--text-muted))" }}>&mdash;</span>}</td>
                  <td><StatusBadge status={j.status} /></td>
                  <td className="text-[11px] max-w-md truncate" title={j.message ?? ""}>{j.message ?? <span style={{ color: "hsl(var(--text-muted))" }}>&mdash;</span>}</td>
                  <td className="font-mono text-[11px]">{j.s3_object_key ? "stored" : <span style={{ color: "hsl(var(--text-muted))" }}>&mdash;</span>}</td>
                  <td><div className="flex gap-2">
                    {j.status === "completed" && j.job_type !== "notebook_output_download" && !j.s3_object_key && (
                      <button className="btn btn-ghost text-xs" onClick={() => { kaggleApi.download(j.id).then(() => load(page)); }}><Download size={13} /> Output</button>
                    )}
                    {j.s3_presigned_url || j.s3_object_key ? (
                      <button className="btn btn-ghost text-xs" onClick={() => { kaggleApi.artifactUrl(j.id).then(r => window.open(r.url)); }}><ExternalLink size={13} /> Open</button>
                    ) : null}
                  </div></td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
        {!loading && jobs.length > 0 && <Pager page={page} pageSize={pageSize} total={total} onPage={(next) => load(next)} onPageSize={(size) => setPageSize(size)} />}
      </div>
    </div>
  );
}
