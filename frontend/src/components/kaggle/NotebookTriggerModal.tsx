import type { Account, Notebook } from "../../types/kaggle";
import { kaggleApi } from "../../api/kaggleApi";

export function NotebookTriggerModal({
  notebooks,
  accounts = [],
  onTriggered,
}: {
  notebooks: Notebook[];
  accounts?: Account[];
  onTriggered: () => void;
}) {
  const account = accounts[0]?.name;
  return (
    <div className="card">
      <h3>Notebooks</h3>
      {notebooks.map((n) => (
        <p key={n.notebook_id}>
          {n.notebook_id}{" "}
          <button
            className="btn"
            disabled={!n.valid || !account}
            onClick={() =>
              account && kaggleApi
                .trigger({ account, notebook_id: n.notebook_id })
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
