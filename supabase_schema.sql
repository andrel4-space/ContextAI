-- Run once in Supabase: SQL Editor → New query → Paste → Run

create table if not exists cognitive_logs (
  id bigint generated always as identity primary key,
  client_id text not null,
  timestamp text not null,
  energy_score integer not null,
  pressure_threshold text not null,
  raw_prompt text not null,
  optimized_output text not null,
  created_at timestamptz default now()
);

create index if not exists idx_cognitive_logs_client_id on cognitive_logs (client_id);
create index if not exists idx_cognitive_logs_id_desc on cognitive_logs (id desc);
