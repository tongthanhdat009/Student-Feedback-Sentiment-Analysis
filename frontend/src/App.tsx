import { useEffect, useState } from "react";
import { Activity, ClipboardList, MonitorPlay, Users, Server, ChevronRight, PanelLeftClose, PanelLeftOpen } from "lucide-react";
import { Dashboard } from "./routes/Dashboard";
import { AccountsPage } from "./routes/AccountsPage";
import { NotebooksPage } from "./routes/NotebooksPage";
import { AuditPage } from "./routes/AuditPage";
import { Toaster } from "./components/kaggle/Toast";

type Page = "dashboard" | "accounts" | "notebooks" | "audit";

const NAV: { id: Page; path: string; label: string; icon: typeof Activity }[] = [
  { id: "dashboard", path: "/", label: "Overview", icon: Activity },
  { id: "notebooks", path: "/notebooks", label: "Notebooks", icon: MonitorPlay },
  { id: "audit", path: "/audit", label: "Audit", icon: ClipboardList },
  { id: "accounts", path: "/accounts", label: "Accounts", icon: Users },
];

function pageFromPath(path: string): Page {
  const match = NAV.find((item) => item.path === path);
  return match?.id ?? "dashboard";
}

export default function App() {
  const [page, setPage] = useState<Page>(() => pageFromPath(window.location.pathname));
  const [collapsed, setCollapsed] = useState(false);

  useEffect(() => {
    const onPop = () => setPage(pageFromPath(window.location.pathname));
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
  }, []);

  const navigate = (next: Page) => {
    const item = NAV.find((nav) => nav.id === next)!;
    setPage(next);
    window.history.pushState(null, "", item.path);
  };

  const content =
    page === "accounts" ? <AccountsPage /> :
    page === "notebooks" ? <NotebooksPage /> :
    page === "audit" ? <AuditPage /> :
    <Dashboard />;

  return (
    <div className="flex min-h-screen">
      <aside className={`sidebar transition-all duration-200 ${collapsed ? "w-20" : "w-60"}`}>
        <div className="flex items-center gap-2.5 px-4 py-5 border-b" style={{ borderColor: "hsl(var(--border))" }}>
          <div className="w-7 h-7 rounded-md flex items-center justify-center shrink-0" style={{ background: "hsl(var(--accent))" }}>
            <Server size={15} className="text-white" />
          </div>
          {!collapsed && <div>
            <div className="text-sm font-semibold tracking-tight" style={{ color: "hsl(var(--text-primary))" }}>Kaggle Manager</div>
            <div className="text-[10px] font-mono uppercase tracking-[0.08em]" style={{ color: "hsl(var(--text-muted))" }}>Notebook MVP</div>
          </div>}
        </div>

        <nav className="flex-1 px-3 py-4 space-y-1 overflow-y-auto scrollbar-thin">
          <button className="sidebar-link w-full text-left mb-2" onClick={() => setCollapsed(!collapsed)}>
            {collapsed ? <PanelLeftOpen size={17} /> : <PanelLeftClose size={17} />}
            {!collapsed && <span className="flex-1">Collapse</span>}
          </button>
          {!collapsed && <div className="text-[10px] font-mono uppercase tracking-[0.1em] px-3 mb-2" style={{ color: "hsl(var(--text-muted))" }}>Navigation</div>}
          {NAV.map((item) => (
            <button key={item.id} onClick={() => navigate(item.id)} className={`sidebar-link w-full text-left ${page === item.id ? "active" : ""}`} title={item.label}>
              <item.icon size={17} />
              {!collapsed && <span className="flex-1">{item.label}</span>}
              {!collapsed && page === item.id && <ChevronRight size={14} style={{ color: "hsl(var(--accent))" }} />}
            </button>
          ))}
        </nav>

        {!collapsed && <div className="px-4 py-4 border-t" style={{ borderColor: "hsl(var(--border))" }}>
          <div className="text-[10px] font-mono uppercase tracking-[0.08em]" style={{ color: "hsl(var(--text-muted))" }}>API Status</div>
          <div className="flex items-center gap-2 mt-1.5">
            <span className="w-2 h-2 rounded-full" style={{ background: "hsl(var(--success))" }} />
            <span className="text-xs" style={{ color: "hsl(var(--text-secondary))" }}>Connected</span>
          </div>
        </div>}
      </aside>

      <main className={`flex-1 p-8 min-h-screen transition-all duration-200 ${collapsed ? "ml-20" : "ml-60"}`} style={{ background: "hsl(var(--surface))" }}>
        {content}
      </main>
      <Toaster />
    </div>
  );
}
