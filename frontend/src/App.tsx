import { useState } from "react";
import { Activity, MonitorPlay, Users, Server, ChevronRight } from "lucide-react";
import { Dashboard } from "./routes/Dashboard";
import { AccountsPage } from "./routes/AccountsPage";
import { NotebooksPage } from "./routes/NotebooksPage";

type Page = "dashboard" | "accounts" | "notebooks";

const NAV: { id: Page; label: string; icon: typeof Activity }[] = [
  { id: "dashboard", label: "Overview", icon: Activity },
  { id: "notebooks", label: "Notebooks", icon: MonitorPlay },
  { id: "accounts", label: "Accounts", icon: Users },
];

export default function App() {
  const [page, setPage] = useState<Page>("dashboard");

  const content =
    page === "accounts" ? <AccountsPage /> :
    page === "notebooks" ? <NotebooksPage /> :
    <Dashboard />;

  return (
    <div className="flex min-h-screen">
      <aside className="sidebar">
        <div className="flex items-center gap-2.5 px-4 py-5 border-b" style={{ borderColor: "hsl(var(--border))" }}>
          <div className="w-7 h-7 rounded-md flex items-center justify-center" style={{ background: "hsl(var(--accent))" }}>
            <Server size={15} className="text-white" />
          </div>
          <div>
            <div className="text-sm font-semibold tracking-tight" style={{ color: "hsl(var(--text-primary))" }}>Kaggle Manager</div>
            <div className="text-[10px] font-mono uppercase tracking-[0.08em]" style={{ color: "hsl(var(--text-muted))" }}>Notebook MVP</div>
          </div>
        </div>

        <nav className="flex-1 px-3 py-4 space-y-1 overflow-y-auto scrollbar-thin">
          <div className="text-[10px] font-mono uppercase tracking-[0.1em] px-3 mb-2" style={{ color: "hsl(var(--text-muted))" }}>
            Navigation
          </div>
          {NAV.map((item) => (
            <button
              key={item.id}
              onClick={() => setPage(item.id)}
              className={`sidebar-link w-full text-left ${page === item.id ? "active" : ""}`}
            >
              <item.icon size={17} />
              <span className="flex-1">{item.label}</span>
              {page === item.id && <ChevronRight size={14} style={{ color: "hsl(var(--accent))" }} />}
            </button>
          ))}
        </nav>

        <div className="px-4 py-4 border-t" style={{ borderColor: "hsl(var(--border))" }}>
          <div className="text-[10px] font-mono uppercase tracking-[0.08em]" style={{ color: "hsl(var(--text-muted))" }}>
            API Status
          </div>
          <div className="flex items-center gap-2 mt-1.5">
            <span className="w-2 h-2 rounded-full" style={{ background: "hsl(var(--success))" }} />
            <span className="text-xs" style={{ color: "hsl(var(--text-secondary))" }}>Connected</span>
          </div>
        </div>
      </aside>

      <main className="flex-1 ml-60 p-8 min-h-screen" style={{ background: "hsl(var(--surface))" }}>
        {content}
      </main>
    </div>
  );
}
