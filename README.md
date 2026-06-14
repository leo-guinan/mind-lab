# Mind Lab

A public research repository for observing one mind over time: Leo Guinan’s own public record, structured for analysis, annotation, and long-term study.

## What this is

This repo is a place to peer into the mind of Leo Guinan through the years.

It contains:

- `data/tweets_raw.jsonl` — exported public tweets in JSONL format
- `data/tweets_for_labeling.csv` — tabular export prepared for human labeling
- `data/tweets_for_llm_labeling.jsonl` — export formatted for LLM-assisted labeling
- `references/schema.md` — field definitions and labeling schema

The data comes from public sources. It is shared here to make the analytical record inspectable, reproducible, and improvable.

## Principles

- Public data, public repo
- No secret keys, no private credentials, no hidden snapshots
- Schema documented in code
- Changes tracked like any research artifact

## Data overview

- Source: Twitter archive / community archive exports
- Format: JSONL + CSV
- Scope: 2020-01-01 through present
- Size: ~43k tweets

## Schema

See `references/schema.md` for field definitions and labeling taxonomy.

## Safety

This repo is intentionally lean:

```text
.env
.env.*
*.private.*
.venv/
venv/
.ipynb_checkpoints/
node_modules/
*.parquet
*.csv.gz
*.tar.gz
*.zip
.DS_Store
Thumbs.db
.secrets/
secrets/
key*
id_rsa
*.pem
*.key
```

If a file could contain credentials, it stays out.

## How to use

```bash
git clone ...
cd mind-lab
```

Work in branches. Open issues for schema changes. Do not add keys.

## Citation

Leo Guinan ORCID: 0009-0007-7879-1059
