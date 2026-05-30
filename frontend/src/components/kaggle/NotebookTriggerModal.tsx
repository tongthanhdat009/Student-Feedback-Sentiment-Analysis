import type { Notebook } from "../../types/kaggle";
import { kaggleApi } from "../../api/kaggleApi";
export function NotebookTriggerModal({
  notebooks,
  onTriggered,
}: {
  notebooks: Notebook[];
  onTriggered: () => void;
}) {
  return (
    <div className="card">
      <h3>Notebooks</h3>
      {notebooks.map((n) => (
        <p key={n.notebook_id}>
          {n.notebook_id}{" "}
          <button
            className="btn"
            onClick={() =>
              kaggleApi
                .trigger({ account: "main", notebook_id: n.notebook_id })
                .then(onTriggered)
            }
          >
            Trigger
          </button>
        </p>
      ))}
    </div>
  );
}
