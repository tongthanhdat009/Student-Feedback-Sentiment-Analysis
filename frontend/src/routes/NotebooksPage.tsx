import { useEffect, useState } from "react";
import { Edit3, FileText, Plus, Play, RefreshCw } from "lucide-react";
import { kaggleApi } from "../api/kaggleApi";
import type { Account, Notebook } from "../types/kaggle";
import { SideDrawer } from "../components/kaggle/SideDrawer";
import { Pager } from "../components/kaggle/Pager";
import { toast } from "../components/kaggle/Toast";

export function NotebooksPage() {
  const [items, setItems] = useState<Notebook[]>([]);
  const [accounts, setAccounts] = useState<Account[]>([]);
  const [selectedAccounts, setSelectedAccounts] = useState<string[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [loading, setLoading] = useState(true);
  const [drawer, setDrawer] = useState<{ type: "trigger" | "new" | "edit"; notebook?: Notebook } | null>(null);
  const [pageSize, setPageSize] = useState(10);

  const load = (nextPage = page) => {
    setLoading(true);
    Promise.all([kaggleApi.inventory(nextPage, pageSize), kaggleApi.accounts(1, 100)])
      .then(([notebooks, accountPage]) => {
        setItems(notebooks.items); setTotal(notebooks.total); setPage(notebooks.page); setAccounts(accountPage.items);
        setSelectedAccounts((current) => current.length ? current : (accountPage.items[0]?.name ? [accountPage.items[0].name] : []));
      })
      .catch(console.error).finally(() => setLoading(false));
  };
  useEffect(() => { void load(1); }, []);

  const toggleAccount = (name: string) => setSelectedAccounts((current) => current.includes(name) ? current.filter((item) => item !== name) : [...current, name]);
  const trigger = () => {
    const notebook = drawer?.notebook;
    if (!notebook || !selectedAccounts.length || !notebook.valid) return;
    kaggleApi.trigger({ accounts: selectedAccounts, notebook_id: notebook.slug }).then((result) => {
      const count = Array.isArray(result) ? result.length : 1;
      toast.success(`${count} job(s) queued`); setDrawer(null);
    }).catch((err) => toast.error(String(err)));
  };

  return (
    <div className="space-y-8">
      <div className="flex items-center justify-between">
        <div><p className="text-[11px] font-mono uppercase tracking-[0.12em] mb-1" style={{ color: "hsl(var(--text-muted))" }}>Catalog</p><h1 className="text-2xl font-semibold tracking-tight" style={{ color: "hsl(var(--text-primary))" }}>Notebook Inventory</h1></div>
        <div className="flex gap-2"><button className="btn btn-ghost" onClick={() => load(page)}><RefreshCw size={14} /> Refresh</button><button className="btn" onClick={() => setDrawer({ type: "new" })}><Plus size={14} /> Add Notebook</button></div>
      </div>

      <div className="card space-y-4">
        <h2 className="text-sm font-semibold tracking-tight" style={{ color: "hsl(var(--text-primary))" }}>Available Notebooks <span className="font-mono text-xs ml-2" style={{ color: "hsl(var(--text-muted))" }}>({total})</span></h2>
        {loading ? <div className="text-xs py-8 text-center" style={{ color: "hsl(var(--text-muted))" }}>Scanning inventory&hellip;</div> : items.length === 0 ? (
          <div className="py-12 text-center space-y-2"><FileText size={28} style={{ color: "hsl(var(--text-muted))", margin: "0 auto" }} /><p className="text-sm" style={{ color: "hsl(var(--text-muted))" }}>No notebooks found in notebook/kaggle/.</p></div>
        ) : <>
          <table><thead><tr><th>Notebook</th><th>Manifest</th><th>Status</th><th>Accelerator</th><th>Tags</th><th></th></tr></thead><tbody>
            {items.map(n => <tr key={n.slug}>
              <td className="font-mono text-xs" style={{ color: "hsl(var(--text-primary))" }}>{n.slug}</td>
              <td className="text-xs"><div style={{ color: "hsl(var(--text-primary))" }}>{n.manifest.title ?? n.title ?? "-"}</div><div>{n.manifest.description ?? ""}</div><div className="font-mono text-[11px]">entry: {n.manifest.entry_file ?? "-"}</div></td>
              <td><span className={n.valid ? "badge badge-success" : "badge badge-danger"}>{n.valid ? "valid" : "invalid"}</span>{!n.valid && <ul className="mt-2 text-[11px]" style={{ color: "hsl(var(--danger))" }}>{n.errors.map((e) => <li key={e}>• {e}</li>)}</ul>}</td>
              <td className="font-mono text-[11px]">{n.manifest.default_accelerator ?? "-"}</td>
              <td>{(n.manifest.tags ?? []).map((tag) => <span key={tag} className="badge badge-muted mr-1">{tag}</span>)}</td>
              <td><div className="flex justify-end gap-2"><button className="btn btn-ghost text-xs" onClick={() => setDrawer({ type: "edit", notebook: n })}><Edit3 size={13} /> Edit</button><button className="btn text-xs" disabled={!n.valid} onClick={() => setDrawer({ type: "trigger", notebook: n })}><Play size={13} /> Trigger</button></div></td>
            </tr>)}
          </tbody></table>
          <Pager page={page} pageSize={pageSize} total={total} onPage={(next) => load(next)} onPageSize={(size) => setPageSize(size)} />
        </>}
      </div>

      <SideDrawer open={!!drawer} title={drawer?.type === "trigger" ? `Run ${drawer.notebook?.slug}` : drawer?.type === "edit" ? "Edit Notebook" : "Add Notebook"} onClose={() => setDrawer(null)}>
        {drawer?.type === "trigger" ? <div className="space-y-4">
          <p className="text-sm" style={{ color: "hsl(var(--text-secondary))" }}>Select one or more accounts. A separate job and unique Kaggle ref will be created for each account.</p>
          <div className="space-y-2">{accounts.map((a) => <label key={a.id} className="flex items-center justify-between rounded-lg border p-3 text-sm" style={{ borderColor: "hsl(var(--border))" }}><span>{a.name} <span className="font-mono text-[11px]" style={{ color: "hsl(var(--text-muted))" }}>({a.kaggle_username})</span></span><input type="checkbox" checked={selectedAccounts.includes(a.name)} onChange={() => toggleAccount(a.name)} /></label>)}</div>
          <button className="btn w-full" disabled={!selectedAccounts.length} onClick={trigger}><Play size={14} /> Queue {selectedAccounts.length || 0} job(s)</button>
        </div> : <div className="space-y-4"><p className="text-sm" style={{ color: "hsl(var(--text-secondary))" }}>Notebook files live in <code>notebook/kaggle/&lt;slug&gt;</code>. Add/edit currently requires creating or updating <code>notebook.yaml</code>, <code>notebook.ipynb</code>, and <code>kernel-metadata.json</code> on disk.</p><div className="rounded-lg border p-4 text-xs" style={{ borderColor: "hsl(var(--border))", color: "hsl(var(--text-muted))" }}>{drawer?.notebook ? JSON.stringify(drawer.notebook.manifest, null, 2) : "Backend file-management endpoint not implemented yet."}</div></div>}
      </SideDrawer>
    </div>
  );
}
