import { useEffect, useState } from "react";
import { CheckCircle2, Info, XCircle } from "lucide-react";

type ToastType = "success" | "error" | "info";
type ToastItem = { id: number; type: ToastType; message: string };
type ToastEvent = { type?: ToastType; message: string };

let nextId = 1;
const EVENT_NAME = "kaggle-manager-toast";

export const toast = {
  success: (message: string) => emitToast({ type: "success", message }),
  error: (message: string) => emitToast({ type: "error", message }),
  info: (message: string) => emitToast({ type: "info", message }),
};

function emitToast(detail: ToastEvent) {
  window.dispatchEvent(new CustomEvent<ToastEvent>(EVENT_NAME, { detail }));
}

export function Toaster() {
  const [items, setItems] = useState<ToastItem[]>([]);

  useEffect(() => {
    const onToast = (event: Event) => {
      const detail = (event as CustomEvent<ToastEvent>).detail;
      if (!detail?.message) return;
      const item = { id: nextId++, type: detail.type ?? "info", message: detail.message };
      setItems((current) => [...current, item]);
      window.setTimeout(() => setItems((current) => current.filter((toast) => toast.id !== item.id)), 4200);
    };
    window.addEventListener(EVENT_NAME, onToast);
    return () => window.removeEventListener(EVENT_NAME, onToast);
  }, []);

  if (!items.length) return null;

  return (
    <div className="toast-stack">
      {items.map((item) => {
        const Icon = item.type === "success" ? CheckCircle2 : item.type === "error" ? XCircle : Info;
        return (
          <div key={item.id} className={`toast toast-${item.type}`}>
            <Icon size={16} />
            <span>{item.message}</span>
          </div>
        );
      })}
    </div>
  );
}
