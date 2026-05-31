import type { Account } from "../../types/kaggle";
import { kaggleApi } from "../../api/kaggleApi";
import { toast } from "./Toast";
export function AccountTable({
  accounts,
  onChange,
}: {
  accounts: Account[];
  onChange: () => void;
}) {
  return (
    <table>
      <thead>
        <tr>
          <th>Name</th>
          <th>User</th>
          <th>Active</th>
          <th>Actions</th>
        </tr>
      </thead>
      <tbody>
        {accounts.map((a) => (
          <tr key={a.id}>
            <td>{a.name}</td>
            <td>{a.kaggle_username}</td>
            <td>{String(a.is_active)}</td>
            <td>
              <button
                className="btn"
                onClick={() =>
                  kaggleApi.testAccount(a.name).then(() => toast.success("Auth OK")).catch((err) => toast.error(String(err)))
                }
              >
                Test
              </button>{" "}
              <button
                className="btn"
                onClick={() => kaggleApi.deleteAccount(a.name).then(onChange)}
              >
                Delete
              </button>
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
