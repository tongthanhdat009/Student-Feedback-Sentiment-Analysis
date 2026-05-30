import { useEffect, useState } from "react";
import { Play, FileText, RefreshCw } from "lucide-react";
import { kaggleApi } from "../api/kaggleApi";
import type { Notebook } from "../types/kaggle";

export function NotebooksPage() {
  const [items, setItems] = useState<Notebook[]>([]);
  const [loading, setLoading] = useState(true);

  const load = () => {
    setLoading(true);
    kaggleApi.inventory().then(setItems).catch(console.error).finally(() => setLoading(false));
  };
  useEffect(() => { void load(); }, []);

  return (
    <div className="space-y-8">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-[11px] font-mono uppercase tracking-[0.12em] mb-1" style={{ color: "hsl(var(--text-muted))" }}>Catalog</p>
          <h1 className="text-2xl font-semibold tracking-tight" style={{ color: "hsl(var(--text-primary))" }}>Notebook Inventory</h1>
        </div>
        <button className="btn btn-ghost" onClick={load}>
          <RefreshCw size={14} /> Refresh
        </button>
      </div>

      <div className="card">
        <h2 className="text-sm font-semibold tracking-tight mb-4" style={{ color: "hsl(var(--text-primary))" }}>
          Available Notebooks
          <span className="font-mono text-xs ml-2" style={{ color: "hsl(var(--text-muted))" }}>({items.length})</span>
        </h2>
        {loading ? (
          <div className="text-xs py-8 text-center" style={{ color: "hsl(var(--text-muted))" }}>Scanning inventory&hellip;</div>
        ) : items.length === 0 ? (
          <div className="py-12 text-center space-y-2">
            <FileText size={28} style={{ color: "hsl(var(--text-muted))", margin: "0 auto" }} />
            <p className="text-sm" style={{ color: "hsl(var(--text-muted))" }}>No notebooks found in notebook/kaggle/.</p>
          </div>
        ) : (
          <table>
            <thead>
              <tr>
                <th>Notebook ID</th>
                <th>Title</th>
                <th>GPU</th>
                <th>Internet</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {items.map(n => (
                <tr key={n.notebook_id}>
                  <td className="font-mono text-xs" style={{ color: "hsl(var(--text-primary))" }}>{n.notebook_id}</td>
                  <td className="text-xs">{String(n.metadata?.title ?? "-")}</td>
                  <td>
                    <span className={n.metadata?.enable_gpu ? "badge badge-success" : "badge badge-muted"}>
                      {n.metadata?.enable_gpu ? "on" : "off"}
                    </span>
                  </td>
                  <td>
                    <span className={n.metadata?.enable_internet ? "badge badge-success" : "badge badge-muted"}>
                      {n.metadata?.enable_internet ? "on" : "off"}
                    </span>
                  </td>
                  <td>
                    <button
                      className="btn text-xs"
                      onClick={() => {
                        kaggleApi.trigger({ account: "main", notebook_id: n.notebook_id });
                        alert("Job queued!");
                      }}
                    >
                      <Play size={13} /> Trigger
                    </button>
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
