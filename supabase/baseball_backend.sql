create extension if not exists pgcrypto;

insert into storage.buckets (id, name, public)
values ('baseball-artifacts', 'baseball-artifacts', false)
on conflict (id) do update set public = false;

create table if not exists public.artifact_manifest (
    id uuid primary key default gen_random_uuid(),
    app_name text not null check (app_name in ('nrfi', 'hr')),
    artifact_type text not null,
    storage_path text not null,
    content_type text,
    size_bytes bigint,
    sha256 text,
    is_latest boolean not null default false,
    metadata jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now()
);

create index if not exists artifact_manifest_latest_idx
    on public.artifact_manifest (app_name, artifact_type, is_latest, created_at desc);

create table if not exists public.refresh_state (
    app_name text primary key check (app_name in ('nrfi', 'hr')),
    last_historical_fetch_at timestamptz,
    cooldown_hours integer not null default 24,
    latest_statcast_path text,
    latest_training_path text,
    latest_model_path text,
    latest_predictions_path text,
    metadata jsonb not null default '{}'::jsonb,
    updated_at timestamptz not null default now()
);

create table if not exists public.pipeline_runs (
    id uuid primary key default gen_random_uuid(),
    app_name text not null check (app_name in ('nrfi', 'hr')),
    job_name text not null,
    status text not null check (status in ('ok', 'error', 'skipped')),
    started_at timestamptz not null default now(),
    finished_at timestamptz,
    detail text,
    error text,
    metadata jsonb not null default '{}'::jsonb
);

create index if not exists pipeline_runs_app_started_idx
    on public.pipeline_runs (app_name, started_at desc);

create table if not exists public.manual_odds (
    id uuid primary key default gen_random_uuid(),
    app_name text not null check (app_name in ('nrfi', 'hr')),
    game_date date not null,
    game_pk bigint,
    market text not null,
    player_name text,
    american_odds integer not null,
    book text not null default 'manual',
    metadata jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    constraint manual_odds_market_check check (
        (app_name = 'nrfi' and market in ('NRFI', 'YRFI') and game_pk is not null)
        or
        (app_name = 'hr' and market = 'HR' and player_name is not null)
    )
);

create index if not exists manual_odds_lookup_idx
    on public.manual_odds (app_name, game_date, market, game_pk, player_name);

alter table public.artifact_manifest enable row level security;
alter table public.refresh_state enable row level security;
alter table public.pipeline_runs enable row level security;
alter table public.manual_odds enable row level security;

comment on table public.artifact_manifest is 'Metadata index for generated baseball app artifacts stored in Supabase Storage.';
comment on table public.refresh_state is 'Durable refresh/cooldown state for baseball prediction apps.';
comment on table public.pipeline_runs is 'Durable pipeline run log for dashboard and automation jobs.';
comment on table public.manual_odds is 'Optional manually entered odds for HR and NRFI/YRFI markets.';
