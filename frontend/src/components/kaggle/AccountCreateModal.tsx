import { useState } from "react";
import { kaggleApi } from "../../api/kaggleApi";
export function AccountCreateModal({ onCreated }: { onCreated: () => void }) {
  const [f, setF] = useState({
    name: "main",
    kaggle_username: "",
    kaggle_key: "",
  });
  return (
    <div className="card">
      <h3>Create account</h3>
      {(["name", "kaggle_username", "kaggle_key"] as const).map((k) => (
        <p key={k}>
          <input
            className="input"
            placeholder={k}
            type={k === "kaggle_key" ? "password" : "text"}
            value={f[k]}
            onChange={(e) => setF({ ...f, [k]: e.target.value })}
          />
        </p>
      ))}
      <button
        className="btn"
        onClick={async () => {
          await kaggleApi.createAccount(f);
          onCreated();
        }}
      >
        Save
      </button>
    </div>
  );
}
