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
  status: "pending" | "staging" | "pushed" | "running" | "completed" | "failed" | string;
  message?: string | null;
  output_path?: string | null;
  kaggle_ref?: string | null;
  staging_path?: string | null;
  last_polled_at?: string | null;
  timeout_seconds?: number | null;
  s3_object_key?: string | null;
  s3_presigned_url?: string | null;
  s3_presigned_url_expires_at?: string | null;
  staging_s3_prefix?: string | null;
  output_s3_prefix?: string | null;
  result_metadata?: Record<string, unknown> | null;
  created_at: string;
  started_at?: string | null;
  finished_at?: string | null;
};

export type NotebookManifest = {
  slug?: string;
  title?: string;
  description?: string;
  entry_file?: string;
  default_accelerator?: string;
  default_timeout_seconds?: number;
  tags?: string[];
  params?: Record<string, unknown>;
  artifacts?: string[];
};

export type Notebook = {
  slug: string;
  notebook_id: string;
  path: string;
  title?: string | null;
  valid: boolean;
  errors: string[];
  manifest: NotebookManifest;
  metadata: Record<string, unknown>;
};

export type PageResult<T> = { items: T[]; total: number; page: number; page_size: number };


export type Dataset = {
  id: string;
  slug: string;
  dataset_ref: string;
  title?: string | null;
  description?: string | null;
  local_path?: string | null;
  status: string;
  last_synced_at?: string | null;
  last_validated_at?: string | null;
  validation_result?: Record<string, unknown> | null;
  is_active: boolean;
  created_at: string;
  updated_at: string;
};


export type NotebookDeployment = {
  id: string;
  account_id: string;
  notebook_id: string;
  kaggle_ref: string;
  remote_slug: string;
  remote_title?: string | null;
  source_path?: string | null;
  is_active: boolean;
  last_synced_at?: string | null;
  last_triggered_at?: string | null;
  last_status?: string | null;
  deployment_metadata?: Record<string, unknown> | null;
  created_at: string;
  updated_at: string;
};

export type NotebookSyncRequest = {
  account: string;
  remote_slug?: string;
  title?: string;
  dataset_sources?: string[];
  enable_gpu?: boolean;
};
