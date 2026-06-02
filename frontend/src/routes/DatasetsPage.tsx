import { useEffect, useState } from "react";
import { CheckCircle2, Database, Edit3, Plus, RefreshCw, Trash2 } from "lucide-react";
import { kaggleApi } from "../api/kaggleApi";
import type { Dataset } from "../types/kaggle";
import { SideDrawer } from "../components/kaggle/SideDrawer";
import { Pager } from "../components/kaggle/Pager";
import { toast } from "../components/kaggle/Toast";

type DatasetForm = { slug: string; dataset_ref: string; title: string; description: string; local_path: string; status: string };
const blankForm: DatasetForm = { slug: "", dataset_ref: "", title: "", description: "", local_path: "data/processed", status: "active" };

export function DatasetsPage() {
  const [items, setItems] = useState<Dataset[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(10);
  const [loading, setLoading] = useState(true);
  const [drawer, setDrawer] = useState<{ mode: "create" | "edit"; original?: Dataset } | null>(null);
  const [form, setForm] = useState<DatasetForm>(blankForm);
  const [validating, setValidating] = useState<Record<string, boolean>>({});

  const load = (nextPage = page, nextPageSize = pageSize) => {
    setLoading(true);
    kaggleApi.datasets(nextPage, nextPageSize).then((res) => {
      setItems(res.items); setTotal(res.total); setPage(res.page);
    }).catch((err) => toast.error(String(err))).finally(() => setLoading(false));
  };
  useEffect(() => { void load(1, pageSize); }, [pageSize]);

  const openCreate = () => { setForm(blankForm); setDrawer({ mode: "create" }); };
  const openEdit = (dataset: Dataset) => {
    setForm({ slug: dataset.slug, dataset_ref: dataset.dataset_ref, title: dataset.title ?? "", description: dataset.description ?? "", local_path: dataset.local_path ?? "", status: dataset.status });
    setDrawer({ mode: "edit", original: dataset });
  };
  const save = async () => {
    const body = { slug: form.slug.trim(), dataset_ref: form.dataset_ref.trim(), title: form.title || undefined, description: form.description || undefined, local_path: form.local_path || undefined, status: form.status || "active" };
    try {
      if (drawer?.mode === "edit" && drawer.original) await kaggleApi.updateDataset(drawer.original.id, body);
      else await kaggleApi.createDataset(body);
      toast.success(drawer?.mode === "edit" ? "Dataset updated" : "Dataset created"); setDrawer(null); load(1);
    } catch (err) {
      toast.error(String(err));
    }
  };
  const validate = async (dataset: Dataset) => {
    setValidating((p) => ({ ...p, [dataset.id]: true }));
    try { await kaggleApi.validateDataset(dataset.id); toast.success("Dataset validated"); load(page); }
    catch (err) { toast.error(String(err)); }
    finally { setValidating((p) => ({ ...p, [dataset.id]: false })); }
  };
  const remove = async (dataset: Dataset) => {
    try { await kaggleApi.deleteDataset(dataset.id); toast.success("Dataset deactivated"); load(page); }
    catch (err) { toast.error(String(err)); }
  };

  return (
    <div className="space-y-8">
      <div className="flex items-center justify-between">
        <div><p className="text-[11px] font-mono uppercase tracking-[0.12em] mb-1" style={{ color: "hsl(var(--text-muted))" }}>Registry</p><h1 className="text-2xl font-semibold tracking-tight" style={{ color: "hsl(var(--text-primary))" }}>Datasets</h1></div>
        <div className="flex gap-2"><button className="btn btn-ghost" onClick={() => load(page)}><RefreshCw size={14} /> Refresh</button><button className="btn" onClick={openCreate}><Plus size={14} /> New Dataset</button></div>
      </div>

      <div className="card space-y-4">
        <h2 className="text-sm font-semibold tracking-tight" style={{ color: "hsl(var(--text-primary))" }}>Managed Datasets <span className="font-mono text-xs ml-2" style={{ color: "hsl(var(--text-muted))" }}>({total})</span></h2>
        {loading ? <div className="text-xs py-8 text-center" style={{ color: "hsl(var(--text-muted))" }}>Loading datasets&hellip;</div> : items.length === 0 ? (
          <div className="py-12 text-center space-y-2"><Database size={28} style={{ color: "hsl(var(--text-muted))", margin: "0 auto" }} /><p className="text-sm" style={{ color: "hsl(var(--text-muted))" }}>No datasets. Create one before triggering notebooks.</p></div>
        ) : <>
          <table><thead><tr><th>Slug</th><th>Kaggle Ref</th><th>Status</th><th>Local Path</th><th>Validated</th><th></th></tr></thead><tbody>
            {items.map((d) => <tr key={d.id}>
              <td><div className="font-mono text-xs" style={{ color: "hsl(var(--text-primary))" }}>{d.slug}</div><div className="text-[11px]" style={{ color: "hsl(var(--text-muted))" }}>{d.title ?? ""}</div></td>
              <td className="font-mono text-xs">{d.dataset_ref}</td>
              <td><span className={d.is_active && d.status !== "failed" ? "badge badge-success" : d.status === "failed" ? "badge badge-danger" : "badge badge-muted"}>{d.status}</span></td>
              <td className="font-mono text-[11px]">{d.local_path ?? "-"}</td>
              <td className="font-mono text-[11px]">{d.last_validated_at ? new Date(d.last_validated_at).toLocaleString() : "Never"}</td>
              <td><div className="flex justify-end gap-2"><button className="btn btn-ghost text-xs" onClick={() => validate(d)} disabled={validating[d.id]}><CheckCircle2 size={13} /> {validating[d.id] ? "Validating" : "Validate"}</button><button className="btn btn-ghost text-xs" onClick={() => openEdit(d)}><Edit3 size={13} /> Edit</button><button className="btn btn-ghost btn-danger text-xs" onClick={() => remove(d)}><Trash2 size={13} /></button></div></td>
            </tr>)}
          </tbody></table>
          <Pager page={page} pageSize={pageSize} total={total} onPage={(next) => load(next, pageSize)} onPageSize={(size) => { setPageSize(size); load(1, size); }} />
        </>}
      </div>

      <SideDrawer open={!!drawer} title={drawer?.mode === "edit" ? "Edit Dataset" : "New Dataset"} onClose={() => setDrawer(null)}>
        <div className="space-y-4">
          <label className="block text-xs">Slug<input className="input mt-1 font-mono" value={form.slug} onChange={(e) => setForm({ ...form, slug: e.target.value })} placeholder="uit-vsfc-processed" /></label>
          <label className="block text-xs">Kaggle Ref<input className="input mt-1 font-mono" value={form.dataset_ref} onChange={(e) => setForm({ ...form, dataset_ref: e.target.value })} placeholder="your-user/uit-vsfc-processed" /></label>
          <label className="block text-xs">Title<input className="input mt-1" value={form.title} onChange={(e) => setForm({ ...form, title: e.target.value })} /></label>
          <label className="block text-xs">Local Path<input className="input mt-1 font-mono" value={form.local_path} onChange={(e) => setForm({ ...form, local_path: e.target.value })} placeholder="data/processed" /></label>
          <label className="block text-xs">Description<textarea className="input mt-1 min-h-24" value={form.description} onChange={(e) => setForm({ ...form, description: e.target.value })} /></label>
          <div className="flex gap-2 pt-2"><button className="btn" disabled={!form.slug.trim() || !form.dataset_ref.trim()} onClick={save}>{drawer?.mode === "edit" ? "Save changes" : "Create"}</button><button className="btn btn-ghost" onClick={() => setDrawer(null)}>Cancel</button></div>
        </div>
      </SideDrawer>
    </div>
  );
}
