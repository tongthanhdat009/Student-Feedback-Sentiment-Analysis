import { useEffect, useState } from "react";
import { Play, Download, CheckCircle2, XCircle, Clock, ExternalLink } from "lucide-react";
import { kaggleApi } from "../api/kaggleApi";
import type { Job } from "../types/kaggle";

function jobStatusBadge(status: string) {
  const cls =
    status === "completed" ? "badge badge-success" :
    status === "failed" ? "badge badge-danger" :
    status === "running" ? "badge badge-running" :
    "badge badge-muted";
  const icon =
    status === "completed" ? <CheckCircle2 size={11} /> :
    status === "failed" ? <XCircle size={11} /> :
    status === "running" ? <Clock size={11} /> :
    null;
  return <span className={cls}>{icon}<span className="ml-1">{status}</span></span>;
}

export function Dashboard() {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [loading, setLoading] = useState(true);

  const load = () => {
    setLoading(true);
    kaggleApi.jobs().then(setJobs).catch(console.error).finally(() => setLoading(false));
  };
  useEffect(() => { void load(); }, []);

  const stats = {
    total: jobs.length,
    completed: jobs.filter(j => j.status === "completed").length,
    running: jobs.filter(j => j.status === "running").length,
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
          { label: "Running", value: stats.running, color: "hsl(var(--warning))" },
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
          <button className="btn btn-ghost text-xs" onClick={load}>Refresh</button>
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
                <th>Job ID</th>
                <th>Type</th>
                <th>Target</th>
                <th>Status</th>
                <th>S3 Artifact</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {jobs.map(j => (
                <tr key={j.id}>
                  <td className="font-mono text-xs">{j.id.slice(0, 12)}&hellip;</td>
                  <td className="font-mono text-[11px]">{j.job_type}</td>
                  <td className="text-xs">{j.target_ref}</td>
                  <td>{jobStatusBadge(j.status)}</td>
                  <td className="font-mono text-[11px]">{j.s3_object_key ? "stored" : <span style={{ color: "hsl(var(--text-muted))" }}>&mdash;</span>}</td>
                  <td>
                    <div className="flex gap-2">
                      {j.status === "completed" && j.job_type !== "notebook_output_download" && (
                        <button className="btn btn-ghost text-xs" onClick={() => { kaggleApi.download(j.id).then(load); }}>
                          <Download size={13} /> Output
                        </button>
                      )}
                      {j.s3_object_key && (
                        <button className="btn btn-ghost text-xs" onClick={() => { kaggleApi.artifactUrl(j.id).then(r => window.open(r.url)); }}>
                          <ExternalLink size={13} /> Open
                        </button>
                      )}
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
