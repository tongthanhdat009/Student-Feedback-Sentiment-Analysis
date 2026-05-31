export function Pager({
  page,
  pageSize,
  total,
  onPage,
  onPageSize,
}: {
  page: number;
  pageSize: number;
  total: number;
  onPage: (page: number) => void;
  onPageSize?: (pageSize: number) => void;
}) {
  const pages = Math.max(1, Math.ceil(total / pageSize));
  return (
    <div className="flex flex-wrap items-center justify-end gap-2 pt-4 text-xs" style={{ color: "hsl(var(--text-muted))" }}>
      {onPageSize && <label className="flex items-center gap-2 font-mono">Rows
        <select className="input w-24" value={pageSize} onChange={(e) => onPageSize(Number(e.target.value))}>
          {[5, 10, 20, 50].map((size) => <option key={size} value={size}>{size}</option>)}
        </select>
      </label>}
      <span className="font-mono">Page {page} / {pages} · {total} total</span>
      <button className="btn btn-ghost text-xs" disabled={page <= 1} onClick={() => onPage(page - 1)}>Prev</button>
      <button className="btn btn-ghost text-xs" disabled={page >= pages} onClick={() => onPage(page + 1)}>Next</button>
    </div>
  );
}
