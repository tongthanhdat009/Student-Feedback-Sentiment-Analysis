import { useEffect, useState } from "react";
import { Plus, Trash2, Terminal, ShieldCheck } from "lucide-react";
import { kaggleApi } from "../api/kaggleApi";
import type { Account } from "../types/kaggle";

export function AccountsPage() {
  const [accounts, setAccounts] = useState<Account[]>([]);
  const [loading, setLoading] = useState(true);
  const [showCreate, setShowCreate] = useState(false);
  const [form, setForm] = useState({ name: "main", kaggle_username: "", kaggle_key: "" });
  const [testing, setTesting] = useState<Record<string, boolean>>({});

  const load = () => {
    setLoading(true);
    kaggleApi.accounts().then(setAccounts).catch(console.error).finally(() => setLoading(false));
  };
  useEffect(() => { void load(); }, []);

  const handleCreate = async () => {
    await kaggleApi.createAccount(form);
    setShowCreate(false);
    setForm({ name: "main", kaggle_username: "", kaggle_key: "" });
    load();
  };

  const handleTest = async (name: string) => {
    setTesting(p => ({ ...p, [name]: true }));
    try { await kaggleApi.testAccount(name); alert("Auth OK"); }
    catch (e) { alert(`Failed: ${e}`); }
    finally { setTesting(p => ({ ...p, [name]: false })); }
  };

  return (
    <div className="space-y-8">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-[11px] font-mono uppercase tracking-[0.12em] mb-1" style={{ color: "hsl(var(--text-muted))" }}>Configuration</p>
          <h1 className="text-2xl font-semibold tracking-tight" style={{ color: "hsl(var(--text-primary))" }}>Kaggle Accounts</h1>
        </div>
        <button className="btn" onClick={() => setShowCreate(!showCreate)}>
          <Plus size={15} /> New Account
        </button>
      </div>

      {showCreate && (
        <div className="card space-y-4">
          <h3 className="text-sm font-semibold" style={{ color: "hsl(var(--text-primary))" }}>Create Account</h3>
          <div className="grid grid-cols-3 gap-4">
            <div>
              <label className="text-[11px] font-mono uppercase tracking-[0.06em] mb-1 block" style={{ color: "hsl(var(--text-muted))" }}>Name</label>
              <input className="input" value={form.name} onChange={e => setForm({ ...form, name: e.target.value })} />
            </div>
            <div>
              <label className="text-[11px] font-mono uppercase tracking-[0.06em] mb-1 block" style={{ color: "hsl(var(--text-muted))" }}>Kaggle Username</label>
              <input className="input" value={form.kaggle_username} onChange={e => setForm({ ...form, kaggle_username: e.target.value })} />
            </div>
            <div>
              <label className="text-[11px] font-mono uppercase tracking-[0.06em] mb-1 block" style={{ color: "hsl(var(--text-muted))" }}>Kaggle Key</label>
              <input className="input" type="password" value={form.kaggle_key} onChange={e => setForm({ ...form, kaggle_key: e.target.value })} />
            </div>
          </div>
          <div className="flex gap-2">
            <button className="btn" onClick={handleCreate}>Save</button>
            <button className="btn btn-ghost" onClick={() => setShowCreate(false)}>Cancel</button>
          </div>
        </div>
      )}

      <div className="card">
        <h2 className="text-sm font-semibold tracking-tight mb-4" style={{ color: "hsl(var(--text-primary))" }}>
          All Accounts
          <span className="font-mono text-xs ml-2" style={{ color: "hsl(var(--text-muted))" }}>({accounts.length})</span>
        </h2>
        {loading ? (
          <div style={{ color: "hsl(var(--text-muted))" }} className="text-xs py-8 text-center">Loading accounts&hellip;</div>
        ) : accounts.length === 0 ? (
          <div className="py-12 text-center space-y-2">
            <ShieldCheck size={28} style={{ color: "hsl(var(--text-muted))", margin: "0 auto" }} />
            <p style={{ color: "hsl(var(--text-muted))" }} className="text-sm">No accounts. Create one to connect Kaggle.</p>
          </div>
        ) : (
          <table>
            <thead>
              <tr>
                <th>Name</th>
                <th>Kaggle User</th>
                <th>Active</th>
                <th>Created</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {accounts.map(a => (
                <tr key={a.id}>
                  <td className="font-mono text-xs" style={{ color: "hsl(var(--text-primary))" }}>{a.name}</td>
                  <td className="text-xs">{a.kaggle_username}</td>
                  <td>
                    <span className={a.is_active ? "badge badge-success" : "badge badge-muted"}>
                      {a.is_active ? "active" : "disabled"}
                    </span>
                  </td>
                  <td className="font-mono text-[11px]">{new Date(a.created_at).toLocaleDateString()}</td>
                  <td>
                    <div className="flex gap-2">
                      <button className="btn btn-ghost text-xs" onClick={() => handleTest(a.name)} disabled={testing[a.name]}>
                        <Terminal size={13} /> {testing[a.name] ? "Testing..." : "Test"}
                      </button>
                      <button className="btn btn-ghost btn-danger text-xs" onClick={() => kaggleApi.deleteAccount(a.name).then(load)}>
                        <Trash2 size={13} />
                      </button>
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
