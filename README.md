# Simple ETL for ADLS/Blob (Python)

This repo contains a minimal ETL that:
1. Downloads a CSV from Azure Blob Storage / ADLS Gen2.
2. Adds a column (timestamp or a user-provided value).
3. Uploads the result back to the same container with `_enriched` suffix.

The ETL is **pure Python + pandas** (no Spark), so it’s easy to run locally and easy to migrate to Databricks as a notebook.

## Layout