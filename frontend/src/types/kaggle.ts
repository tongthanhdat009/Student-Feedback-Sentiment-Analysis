export type Account = {
  id: string;
  name: string;
  kaggle_username: string;
  is_active: boolean;
  created_at: string;
  updated_at: string;
  last_used_at?: string | null;
};
export type Job = {
  id: string;
  account_id?: string | null;
  job_type: string;
  target_ref: string;
  status: string;
  message?: string | null;
  output_path?: string | null;
  s3_object_key?: string | null;
  s3_presigned_url?: string | null;
  created_at: string;
};
export type Notebook = {
  notebook_id: string;
  path: string;
  metadata: Record<string, unknown>;
};
