import { useEffect, useRef } from "react";

export function usePolling(enabled: boolean, callback: () => void | Promise<void>, intervalMs: number) {
  const callbackRef = useRef(callback);
  const inFlightRef = useRef(false);
  callbackRef.current = callback;

  useEffect(() => {
    if (!enabled) return;
    const tick = () => {
      if (inFlightRef.current) return;
      inFlightRef.current = true;
      Promise.resolve(callbackRef.current()).finally(() => {
        inFlightRef.current = false;
      });
    };
    const id = window.setInterval(tick, intervalMs);
    return () => window.clearInterval(id);
  }, [enabled, intervalMs]);
}
