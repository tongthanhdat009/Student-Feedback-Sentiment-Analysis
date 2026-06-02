import { useState } from "react";
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
  const [datasetSource, setDatasetSource] = useState("");
  return (
    <div className="card">
      <h3>Notebooks</h3>
      <input className="input" value={datasetSource} onChange={(event) => setDatasetSource(event.target.value)} placeholder="your-kaggle-user/uit-vsfc-processed" />
      {notebooks.map((n) => (
        <p key={n.notebook_id}>
          {n.notebook_id}{" "}
          <button
            className="btn"
            disabled={!n.valid || !account || !datasetSource.trim()}
            onClick={() =>
              account && kaggleApi
                .trigger({ account, notebook_id: n.notebook_id, dataset_source: datasetSource.trim() })
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
