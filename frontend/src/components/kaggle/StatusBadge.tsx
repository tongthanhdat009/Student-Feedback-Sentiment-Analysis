export function StatusBadge({ status }: { status: string }) {
  const c =
    status === "completed"
      ? "#16a34a"
      : status === "failed"
        ? "#dc2626"
        : status === "running"
          ? "#f59e0b"
          : "#64748b";
  return (
    <span
      style={{
        background: c,
        padding: "4px 8px",
        borderRadius: 999,
        fontSize: 12,
      }}
    >
      {status}
    </span>
  );
}
