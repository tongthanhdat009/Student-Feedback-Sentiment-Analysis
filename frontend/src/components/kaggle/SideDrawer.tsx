import type { ReactNode } from "react";
import { X } from "lucide-react";

export function SideDrawer({ open, title, children, onClose }: { open: boolean; title: string; children: ReactNode; onClose: () => void }) {
  if (!open) return null;
  return (
    <div className="fixed inset-0 z-50 flex justify-end">
      <button className="absolute inset-0 bg-black/40" aria-label="Close drawer" onClick={onClose} />
      <aside className="relative h-full w-full max-w-md border-l p-6 overflow-y-auto" style={{ background: "hsl(var(--surface-elevated))", borderColor: "hsl(var(--border))" }}>
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-lg font-semibold" style={{ color: "hsl(var(--text-primary))" }}>{title}</h2>
          <button className="btn btn-ghost" onClick={onClose}><X size={14} /></button>
        </div>
        {children}
      </aside>
    </div>
  );
}
