export function StatusBadge({ status }: { status: string }) {
  const cls =
    status === "completed"
      ? "badge badge-success"
      : status === "failed"
        ? "badge badge-danger"
        : ["running", "staging", "pushed"].includes(status)
          ? "badge badge-running"
          : "badge badge-muted";
  return <span className={cls}>{status}</span>;
}
