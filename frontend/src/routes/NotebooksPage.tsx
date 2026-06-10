import { useCallback, useEffect, useState } from "react";
import { Play, RefreshCw, UploadCloud } from "lucide-react";
import { kaggleApi } from "../api/kaggleApi";
import type { Account, Dataset, Notebook, NotebookDeployment } from "../types/kaggle";
import { SideDrawer } from "../components/kaggle/SideDrawer";
import { Pager } from "../components/kaggle/Pager";
import { toast } from "../components/kaggle/Toast";

import { usePolling } from "../hooks/usePolling";

export function NotebooksPage() {
  const [items, setItems] = useState<Notebook[]>([]);
  const [accounts, setAccounts] = useState<Account[]>([]);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [deployments, setDeployments] = useState<NotebookDeployment[]>([]);
  const [selectedAccounts, setSelectedAccounts] = useState<string[]>([]);
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(10);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [drawer, setDrawer] = useState<{ type: "sync" | "trigger"; notebook: Notebook } | null>(null);
  const [remoteSlug, setRemoteSlug] = useState("");
  const [title, setTitle] = useState("");
  const [datasetSource, setDatasetSource] = useState("jadt145/sfsa-026");

  const deploymentsFor = (slug: string) => deployments.filter((d) => d.notebook_id === slug && d.is_active);
  const accountName = (id: string) => accounts.find((a) => a.id === id)?.name ?? id;

  const loadInventory = useCallback((nextPage = page) => {
    setLoading(true);
    Promise.all([kaggleApi.inventory(nextPage, pageSize), kaggleApi.deployments()])
      .then(([notebooks, deploymentRows]) => {
        setItems(notebooks.items); setPage(notebooks.page); setTotal(notebooks.total);
        setDeployments(deploymentRows);
      }).catch((err) => toast.error(String(err))).finally(() => setLoading(false));
  }, [page, pageSize]);
  const loadStaticMetadata = useCallback(() => {
    Promise.all([kaggleApi.accounts(1, 100), kaggleApi.datasets(1, 100, true)])
      .then(([accountPage, datasetPage]) => {
        setAccounts(accountPage.items); setDatasets(datasetPage.items);
        setSelectedAccounts((cur) => cur.length ? cur : (accountPage.items[0]?.name ? [accountPage.items[0].name] : []));
      }).catch((err) => toast.error(String(err)));
  }, []);
  const loadDeployments = useCallback(() => {
    return kaggleApi.deployments().then(setDeployments).catch((err) => toast.error(String(err)));
  }, []);
  useEffect(() => { loadStaticMetadata(); loadInventory(1); }, [loadStaticMetadata, loadInventory]);
  const activeDeployment = deployments.some((d) => ["syncing", "pushed", "running"].includes(d.last_status ?? ""));
  usePolling(activeDeployment, loadDeployments, 10000);

  const openSync = (n: Notebook) => { setRemoteSlug(n.slug); setTitle(n.manifest.title ?? n.title ?? n.slug); setDatasetSource(datasets[0]?.dataset_ref ?? "jadt145/sfsa-026"); setDrawer({ type: "sync", notebook: n }); };
  const syncNotebook = () => {
    if (!drawer || !selectedAccounts[0]) return;
    toast.info("Sync started; polling deployment status…");
    kaggleApi.syncNotebook(drawer.notebook.slug, { account: selectedAccounts[0], remote_slug: remoteSlug, title, dataset_sources: datasetSource ? [datasetSource] : undefined, enable_gpu: true })
      .then((d) => { toast.success(`Sync queued: ${d.kaggle_ref}`); setDrawer(null); loadDeployments(); })
      .catch((err) => toast.error(String(err)));
  };
  const trigger = () => {
    if (!drawer) return;
    kaggleApi.trigger({ accounts: selectedAccounts, notebook_id: drawer.notebook.slug })
      .then((r) => { toast.success(`${Array.isArray(r) ? r.length : 1} job(s) queued`); setDrawer(null); })
      .catch((err) => toast.error(String(err)));
  };

  return <div className="space-y-8">
    <div className="flex items-center justify-between">
      <div><p className="text-[11px] font-mono uppercase">Catalog</p><h1 className="text-2xl font-semibold">Notebook Inventory</h1></div>
      <button className="btn btn-ghost" onClick={() => { loadStaticMetadata(); loadInventory(page); }}><RefreshCw size={14} /> Refresh</button>
    </div>
    <div className="card space-y-4">
      <h2 className="text-sm font-semibold">Available Notebooks ({total})</h2>
      {loading ? <div className="text-xs py-8 text-center">Scanning inventory…</div> : <>
        <table><thead><tr><th>Notebook</th><th>Manifest</th><th>Status</th><th>Deployments</th><th></th></tr></thead><tbody>
          {items.map((n) => {
            const deps = deploymentsFor(n.slug);
            return <tr key={n.slug}>
              <td className="font-mono text-xs">{n.slug}</td>
              <td className="text-xs"><div>{n.manifest.title ?? n.title ?? "-"}</div><div>{n.manifest.description ?? ""}</div></td>
              <td><span className={n.valid ? "badge badge-success" : "badge badge-danger"}>{n.valid ? "valid" : "invalid"}</span></td>
              <td className="text-[11px]">{deps.length ? deps.map((d) => <div key={d.id} className="font-mono"><a className="underline" target="_blank" rel="noreferrer" href={`https://www.kaggle.com/code/${d.kaggle_ref}`}>{accountName(d.account_id)}: {d.kaggle_ref}</a><div>{d.last_status ?? "synced"}</div></div>) : <span className="badge badge-muted">not synced</span>}</td>
              <td><div className="flex justify-end gap-2"><button className="btn btn-ghost text-xs" disabled={!n.valid} onClick={() => openSync(n)}><UploadCloud size={13} /> Sync</button><button className="btn text-xs" disabled={!n.valid || !deps.length} onClick={() => setDrawer({ type: "trigger", notebook: n })}><Play size={13} /> Trigger</button></div></td>
            </tr>;
          })}
        </tbody></table>
        <Pager page={page} pageSize={pageSize} total={total} onPage={(next) => loadInventory(next)} onPageSize={setPageSize} />
      </>}
    </div>
    <SideDrawer open={!!drawer} title={drawer?.type === "sync" ? `Sync ${drawer.notebook.slug}` : `Trigger ${drawer?.notebook.slug}`} onClose={() => setDrawer(null)}>
      {drawer?.type === "sync" && <div className="space-y-4">
        <p className="text-sm">Sync updates the same Kaggle notebook ref with latest code. Trigger runs that synced notebook, polls it, then downloads outputs.</p>
        <select className="input w-full" value={selectedAccounts[0] ?? ""} onChange={(e) => setSelectedAccounts(e.target.value ? [e.target.value] : [])}>{accounts.map((a) => <option key={a.id} value={a.name}>{a.name} ({a.kaggle_username})</option>)}</select>
        <input className="input w-full font-mono" value={remoteSlug} onChange={(e) => setRemoteSlug(e.target.value)} placeholder="remote-slug" />
        <input className="input w-full" value={title} onChange={(e) => setTitle(e.target.value)} placeholder="Title" />
        <select className="input w-full font-mono text-xs" value={datasetSource} onChange={(e) => setDatasetSource(e.target.value)}><option value="">Manual Kaggle config</option><option value="jadt145/sfsa-026">jadt145/sfsa-026</option>{datasets.map((d) => <option key={d.id} value={d.dataset_ref}>{d.slug} — {d.dataset_ref}</option>)}</select>
        <button className="btn w-full" onClick={syncNotebook}><UploadCloud size={14} /> Sync to Kaggle</button>
      </div>}
      {drawer?.type === "trigger" && <div className="space-y-4">
        <p className="text-sm">Trigger synced Kaggle notebook refs; outputs auto-upload to S3.</p>
        {accounts.map((a) => { const ok = deploymentsFor(drawer.notebook.slug).some((d) => d.account_id === a.id); return <label key={a.id} className="flex items-center justify-between rounded-lg border p-3 text-sm"><span>{a.name} {!ok && <span className="badge badge-muted ml-2">sync first</span>}</span><input type="checkbox" disabled={!ok} checked={selectedAccounts.includes(a.name)} onChange={() => setSelectedAccounts((cur) => cur.includes(a.name) ? cur.filter((x) => x !== a.name) : [...cur, a.name])} /></label>; })}
        <button className="btn w-full" disabled={!selectedAccounts.length} onClick={trigger}><Play size={14} /> Queue job(s)</button>
      </div>}
    </SideDrawer>
  </div>;
}
