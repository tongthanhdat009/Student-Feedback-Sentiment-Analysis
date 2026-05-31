import { useEffect, useMemo, useState } from "react";
import { Download, ExternalLink, RefreshCw, Search } from "lucide-react";
import { kaggleApi } from "../api/kaggleApi";
import { StatusBadge } from "../components/kaggle/StatusBadge";
import type { Job } from "../types/kaggle";
import { Pager } from "../components/kaggle/Pager";

const activeStatuses = new Set(["pending", "staging", "pushed", "running"]);

function formatTime(value?: string | null) {
  if (!value) return "—";
  return new Date(value).toLocaleString();
}

export function AuditPage() {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(10);
  const [loading, setLoading] = useState(true);
  const [query, setQuery] = useState("");
  const [status, setStatus] = useState("all");

  const load = (nextPage = page) => {
    setLoading(true);
    kaggleApi.jobs(nextPage, pageSize).then((res) => { setJobs(res.items); setTotal(res.total); setPage(res.page); }).catch(console.error).finally(() => setLoading(false));
  };

  useEffect(() => { void load(1); }, []);
  useEffect(() => {
    if (!jobs.some((job) => activeStatuses.has(job.status))) return;
    const id = window.setInterval(() => load(page), 5000);
    return () => window.clearInterval(id);
  }, [jobs]);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    return jobs.filter((job) => {
      const matchesStatus = status === "all" || job.status === status;
      const haystack = [job.id, job.job_type, job.target_ref, job.kaggle_ref, job.message, job.s3_object_key]
        .filter(Boolean)
        .join(" ")
        .toLowerCase();
      return matchesStatus && (!q || haystack.includes(q));
    });
  }, [jobs, query, status]);

  const counts = {
    total: jobs.length,
    active: jobs.filter((job) => activeStatuses.has(job.status)).length,
    completed: jobs.filter((job) => job.status === "completed").length,
    failed: jobs.filter((job) => job.status === "failed").length,
  };

  return (
    <div className="space-y-8">
      <div className="flex items-center justify-between gap-4">
        <div>
          <p className="text-[11px] font-mono uppercase tracking-[0.12em] mb-1" style={{ color: "hsl(var(--text-muted))" }}>Audit</p>
          <h1 className="text-2xl font-semibold tracking-tight" style={{ color: "hsl(var(--text-primary))" }}>Job Activity Log</h1>
          <p className="text-sm mt-2" style={{ color: "hsl(var(--text-secondary))" }}>
            Trigger, Kaggle polling, output, and failure history in one place.
          </p>
        </div>
        <button className="btn btn-ghost" onClick={() => load(page)}>
          <RefreshCw size={14} /> Refresh
        </button>
      </div>

      <div className="grid grid-cols-4 gap-4">
        {[
          { label: "Total", value: counts.total, color: "hsl(var(--accent))" },
          { label: "Active", value: counts.active, color: "hsl(var(--warning))" },
          { label: "Completed", value: counts.completed, color: "hsl(var(--success))" },
          { label: "Failed", value: counts.failed, color: "hsl(var(--danger))" },
        ].map((item) => (
          <div key={item.label} className="card flex flex-col gap-1">
            <span className="stat stat-value" style={{ color: item.color }}>{item.value}</span>
            <span className="stat-label">{item.label}</span>
          </div>
        ))}
      </div>

      <div className="card space-y-4">
        <div className="flex items-center justify-between gap-3">
          <div className="relative flex-1">
            <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2" style={{ color: "hsl(var(--text-muted))" }} />
            <input
              className="input pl-9"
              placeholder="Search job id, notebook, Kaggle ref, message..."
              value={query}
              onChange={(event) => setQuery(event.target.value)}
            />
          </div>
          <select className="input max-w-xs" value={status} onChange={(event) => setStatus(event.target.value)}>
            <option value="all">All statuses</option>
            <option value="pending">pending</option>
            <option value="staging">staging</option>
            <option value="pushed">pushed</option>
            <option value="running">running</option>
            <option value="completed">completed</option>
            <option value="failed">failed</option>
          </select>
        </div>

        {loading ? (
          <div className="text-xs py-8 text-center" style={{ color: "hsl(var(--text-muted))" }}>Loading audit log&hellip;</div>
        ) : filtered.length === 0 ? (
          <div className="text-sm py-12 text-center" style={{ color: "hsl(var(--text-muted))" }}>No audit entries match the filters.</div>
        ) : (
          <div className="overflow-x-auto">
            <table>
              <thead>
                <tr>
                  <th>Created</th>
                  <th>Job</th>
                  <th>Target</th>
                  <th>Kaggle Ref</th>
                  <th>Status</th>
                  <th>Message</th>
                  <th>Runtime</th>
                  <th>Artifact</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {filtered.map((job) => (
                  <tr key={job.id}>
                    <td className="font-mono text-[11px] whitespace-nowrap">{formatTime(job.created_at)}</td>
                    <td>
                      <div className="font-mono text-xs" style={{ color: "hsl(var(--text-primary))" }}>{job.id.slice(0, 12)}&hellip;</div>
                      <div className="font-mono text-[11px]" style={{ color: "hsl(var(--text-muted))" }}>{job.job_type}</div>
                    </td>
                    <td className="font-mono text-[11px]">{job.target_ref}</td>
                    <td className="font-mono text-[11px]">{job.kaggle_ref ?? "—"}</td>
                    <td><StatusBadge status={job.status} /></td>
                    <td className="text-[11px] min-w-80 max-w-xl" title={job.message ?? ""}>{job.message ?? "—"}</td>
                    <td className="font-mono text-[11px] whitespace-nowrap">
                      <div>start: {formatTime(job.started_at)}</div>
                      <div>poll: {formatTime(job.last_polled_at)}</div>
                      <div>end: {formatTime(job.finished_at)}</div>
                    </td>
                    <td className="font-mono text-[11px]">{job.s3_object_key ? "stored" : "—"}</td>
                    <td>
                      <div className="flex gap-2">
                        {job.status === "completed" && job.job_type !== "notebook_output_download" && !job.s3_object_key && (
                          <button className="btn btn-ghost text-xs" onClick={() => { kaggleApi.download(job.id).then(() => load(page)); }}>
                            <Download size={13} /> Output
                          </button>
                        )}
                        {job.s3_presigned_url || job.s3_object_key ? (
                          <button className="btn btn-ghost text-xs" onClick={() => { kaggleApi.artifactUrl(job.id).then((res) => window.open(res.url)); }}>
                            <ExternalLink size={13} /> Open
                          </button>
                        ) : null}
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        {!loading && filtered.length > 0 && <Pager page={page} pageSize={pageSize} total={total} onPage={(next) => load(next)} onPageSize={(size) => setPageSize(size)} />}
      </div>
    </div>
  );
}
