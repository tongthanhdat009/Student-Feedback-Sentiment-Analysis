import { useEffect, useState } from "react";
import { Edit3, Eye, Plus, Trash2, Terminal, ShieldCheck } from "lucide-react";
import { kaggleApi } from "../api/kaggleApi";
import type { Account } from "../types/kaggle";
import { SideDrawer } from "../components/kaggle/SideDrawer";
import { toast } from "../components/kaggle/Toast";
import { Pager } from "../components/kaggle/Pager";

type AccountForm = { name: string; kaggle_username: string; kaggle_key: string; is_active: boolean };
const blankForm: AccountForm = { name: "", kaggle_username: "", kaggle_key: "", is_active: true };

export function AccountsPage() {
  const [accounts, setAccounts] = useState<Account[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [loading, setLoading] = useState(true);
  const [drawer, setDrawer] = useState<{ mode: "create" | "edit"; original?: Account } | null>(null);
  const [detailAccount, setDetailAccount] = useState<Account | null>(null);
  const [form, setForm] = useState<AccountForm>(blankForm);
  const [testing, setTesting] = useState<Record<string, boolean>>({});
  const [pageSize, setPageSize] = useState(10);

  const load = (nextPage = page, nextPageSize = pageSize) => {
    setLoading(true);
    kaggleApi.accounts(nextPage, nextPageSize).then((res) => {
      setAccounts(res.items); setTotal(res.total); setPage(res.page);
    }).catch(console.error).finally(() => setLoading(false));
  };
  useEffect(() => { void load(1, pageSize); }, [pageSize]);



  const openCreate = () => { setForm(blankForm); setDrawer({ mode: "create" }); };
  const openEdit = (account: Account) => {
    setForm({ name: account.name, kaggle_username: account.kaggle_username, kaggle_key: "", is_active: account.is_active });
    setDrawer({ mode: "edit", original: account });
  };

  const save = async () => {
    if (drawer?.mode === "edit" && drawer.original) {
      await kaggleApi.updateAccount(drawer.original.name, {
        name: form.name,
        kaggle_username: form.kaggle_username,
        kaggle_key: form.kaggle_key || undefined,
        is_active: form.is_active,
      });
    } else {
      await kaggleApi.createAccount({ name: form.name, kaggle_username: form.kaggle_username, kaggle_key: form.kaggle_key });
    }
    setDrawer(null); load(1);
  };

  const handleTest = async (name: string) => {
    setTesting(p => ({ ...p, [name]: true }));
    try { await kaggleApi.testAccount(name); toast.success("Auth OK"); }
    catch (e) { toast.error(`Failed: ${e}`); }
    finally { setTesting(p => ({ ...p, [name]: false })); }
  };

  return (
    <div className="space-y-8">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-[11px] font-mono uppercase tracking-[0.12em] mb-1" style={{ color: "hsl(var(--text-muted))" }}>Configuration</p>
          <h1 className="text-2xl font-semibold tracking-tight" style={{ color: "hsl(var(--text-primary))" }}>Kaggle Accounts</h1>
        </div>
        <button className="btn" onClick={openCreate}><Plus size={15} /> New Account</button>
      </div>



      <div className="card">
        <h2 className="text-sm font-semibold tracking-tight mb-4" style={{ color: "hsl(var(--text-primary))" }}>All Accounts <span className="font-mono text-xs ml-2" style={{ color: "hsl(var(--text-muted))" }}>({total})</span></h2>
        {loading ? <div className="text-xs py-8 text-center" style={{ color: "hsl(var(--text-muted))" }}>Loading accounts&hellip;</div> : accounts.length === 0 ? (
          <div className="py-12 text-center space-y-2"><ShieldCheck size={28} style={{ color: "hsl(var(--text-muted))", margin: "0 auto" }} /><p className="text-sm" style={{ color: "hsl(var(--text-muted))" }}>No accounts. Create one to connect Kaggle.</p></div>
        ) : <>
          <table><thead><tr><th>Name</th><th>Kaggle User</th><th>Active</th><th>Created</th><th></th></tr></thead><tbody>
            {accounts.map(a => <tr key={a.id}>
              <td className="font-mono text-xs" style={{ color: "hsl(var(--text-primary))" }}>{a.name}</td>
              <td className="text-xs">{a.kaggle_username}</td>
              <td><span className={a.is_active ? "badge badge-success" : "badge badge-muted"}>{a.is_active ? "active" : "disabled"}</span></td>
              <td className="font-mono text-[11px]">{new Date(a.created_at).toLocaleDateString()}</td>
              <td><div className="flex gap-2 justify-end">
                <button className="btn btn-ghost text-xs" onClick={() => setDetailAccount(a)}><Eye size={13} /> Detail</button>
                <button className="btn btn-ghost text-xs" onClick={() => openEdit(a)}><Edit3 size={13} /> Edit</button>
                <button className="btn btn-ghost text-xs" onClick={() => handleTest(a.name)} disabled={testing[a.name]}><Terminal size={13} /> {testing[a.name] ? "Testing..." : "Test"}</button>
                <button className="btn btn-ghost btn-danger text-xs" onClick={() => kaggleApi.deleteAccount(a.name).then(() => load(page))}><Trash2 size={13} /></button>
              </div></td>
            </tr>)}
          </tbody></table>
          <Pager page={page} pageSize={pageSize} total={total} onPage={(next) => load(next, pageSize)} onPageSize={(size) => { setPageSize(size); load(1, size); }} />
        </>}
      </div>

      <SideDrawer open={!!detailAccount} title="Account Detail" onClose={() => setDetailAccount(null)}>
        {detailAccount && (
          <div className="space-y-5">
            <div className="rounded-xl border p-4" style={{ borderColor: "hsl(var(--border))", background: "hsl(var(--surface-muted))" }}>
              <div className="flex items-center justify-between gap-3">
                <div>
                  <p className="text-[11px] font-mono uppercase tracking-[0.12em]" style={{ color: "hsl(var(--text-muted))" }}>Account</p>
                  <h3 className="text-lg font-semibold" style={{ color: "hsl(var(--text-primary))" }}>{detailAccount.name}</h3>
                </div>
                <span className={detailAccount.is_active ? "badge badge-success" : "badge badge-muted"}>{detailAccount.is_active ? "active" : "disabled"}</span>
              </div>
            </div>

            <div className="space-y-3 text-sm">
              <div className="grid grid-cols-[130px_1fr] gap-3">
                <span style={{ color: "hsl(var(--text-muted))" }}>Kaggle user</span>
                <span className="font-mono text-xs" style={{ color: "hsl(var(--text-primary))" }}>{detailAccount.kaggle_username}</span>
              </div>
              <div className="grid grid-cols-[130px_1fr] gap-3">
                <span style={{ color: "hsl(var(--text-muted))" }}>Token/API key</span>
                <span className="font-mono text-xs" style={{ color: "hsl(var(--text-muted))" }}>Hidden for security</span>
              </div>
              <div className="grid grid-cols-[130px_1fr] gap-3">
                <span style={{ color: "hsl(var(--text-muted))" }}>Created</span>
                <span className="font-mono text-xs" style={{ color: "hsl(var(--text-primary))" }}>{new Date(detailAccount.created_at).toLocaleString()}</span>
              </div>
              <div className="grid grid-cols-[130px_1fr] gap-3">
                <span style={{ color: "hsl(var(--text-muted))" }}>Updated</span>
                <span className="font-mono text-xs" style={{ color: "hsl(var(--text-primary))" }}>{new Date(detailAccount.updated_at).toLocaleString()}</span>
              </div>
              <div className="grid grid-cols-[130px_1fr] gap-3">
                <span style={{ color: "hsl(var(--text-muted))" }}>Last used</span>
                <span className="font-mono text-xs" style={{ color: "hsl(var(--text-primary))" }}>{detailAccount.last_used_at ? new Date(detailAccount.last_used_at).toLocaleString() : "Never"}</span>
              </div>
            </div>

            <div className="rounded-xl border p-4 text-xs" style={{ borderColor: "hsl(var(--border))", color: "hsl(var(--text-muted))" }}>
              Kaggle API token is never displayed. Use Edit to rotate/update credentials.
            </div>

            <div className="flex gap-2 pt-2">
              <button className="btn" onClick={() => { openEdit(detailAccount); setDetailAccount(null); }}><Edit3 size={13} /> Edit account</button>
              <button className="btn btn-ghost" onClick={() => setDetailAccount(null)}>Close</button>
            </div>
          </div>
        )}
      </SideDrawer>

      <SideDrawer open={!!drawer} title={drawer?.mode === "edit" ? "Edit Account" : "New Account"} onClose={() => setDrawer(null)}>
        <div className="space-y-4">
          <label className="block text-xs">Name<input className="input mt-1" value={form.name} onChange={e => setForm({ ...form, name: e.target.value })} /></label>
          <label className="block text-xs">Kaggle Username<input className="input mt-1" value={form.kaggle_username} onChange={e => setForm({ ...form, kaggle_username: e.target.value })} /></label>
          <label className="block text-xs">Kaggle Key / API Token<input className="input mt-1" type="password" value={form.kaggle_key} placeholder={drawer?.mode === "edit" ? "Leave blank to keep existing" : ""} onChange={e => setForm({ ...form, kaggle_key: e.target.value })} /></label>
          <label className="flex items-center gap-2 text-xs"><input type="checkbox" checked={form.is_active} onChange={e => setForm({ ...form, is_active: e.target.checked })} /> Active</label>
          <div className="flex gap-2 pt-2"><button className="btn" onClick={save}>{drawer?.mode === "edit" ? "Save changes" : "Create"}</button><button className="btn btn-ghost" onClick={() => setDrawer(null)}>Cancel</button></div>
        </div>
      </SideDrawer>
    </div>
  );
}
