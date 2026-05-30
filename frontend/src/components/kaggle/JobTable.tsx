import type { Job } from "../../types/kaggle";
import { kaggleApi } from "../../api/kaggleApi";
import { StatusBadge } from "./StatusBadge";
export function JobTable({
  jobs,
  onChange,
}: {
  jobs: Job[];
  onChange: () => void;
}) {
  return (
    <table>
      <thead>
        <tr>
          <th>ID</th>
          <th>Type</th>
          <th>Target</th>
          <th>Status</th>
          <th>S3</th>
          <th>Actions</th>
        </tr>
      </thead>
      <tbody>
        {jobs.map((j) => (
          <tr key={j.id}>
            <td>{j.id.slice(0, 8)}</td>
            <td>{j.job_type}</td>
            <td>{j.target_ref}</td>
            <td>
              <StatusBadge status={j.status} />
            </td>
            <td>{j.s3_object_key ?? "-"}</td>
            <td>
              <button
                className="btn"
                onClick={() => kaggleApi.download(j.id).then(onChange)}
              >
                Download
              </button>{" "}
              {j.s3_object_key && (
                <button
                  className="btn"
                  onClick={() =>
                    kaggleApi
                      .artifactUrl(j.id)
                      .then((r) => (location.href = r.url))
                  }
                >
                  Open
                </button>
              )}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
