from __future__ import annotations
import io
import os
from datetime import datetime, timezone
from typing import Optional, Tuple

import pandas as pd
from azure.storage.blob import BlobServiceClient

def make_blob_client(account: str, sas_token: str) -> BlobServiceClient:
    # account URL works for both Blob and ADLS Gen2 endpoints
    base_url = f"https://{account}.blob.core.windows.net"
    if not sas_token.startswith("?"):
        sas_token = "?" + sas_token
    return BlobServiceClient(account_url=base_url + sas_token)

def download_blob_to_memory(bsc: BlobServiceClient, container: str, blob_path: str) -> bytes:
    blob = bsc.get_blob_client(container=container, blob=blob_path)
    stream = blob.download_blob()
    return stream.readall()

def upload_blob_from_memory(bsc: BlobServiceClient, container: str, blob_path: str, data: bytes, overwrite: bool = True):
    blob = bsc.get_blob_client(container=container, blob=blob_path)
    blob.upload_blob(data, overwrite=overwrite)

def read_csv_bytes(content: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(content))

def write_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return buf.read()

def add_column(df: pd.DataFrame, name: str, value: Optional[str] = None) -> pd.DataFrame:
    """Add a column. If value is None, use current UTC timestamp ISO format."""
    if value is None or value == "":
        value = datetime.now(timezone.utc).isoformat()
    df[name] = value
    return df

def infer_output_blob_path(input_blob: str) -> str:
    """Change 'file.csv' to 'file_enriched.csv'."""
    base = os.path.basename(input_blob)
    stem, ext = os.path.splitext(base)
    enriched = f"{stem}_enriched{ext or '.csv'}"
    return os.path.join(os.path.dirname(input_blob) or "", enriched).replace("\\", "/")

# -- add near the top if not present --
import pandas as pd

# ------------------------------------------------------------------
# Filter rows using a pandas.query expression (e.g., "age > 30 and city == 'AMS'")
# Notes:
# - Backtick-quote column names that contain spaces or special characters: `Total Amount`
# - engine="python" is used for broad operator support
# ------------------------------------------------------------------
def filter_rows(df: pd.DataFrame, expr: str | None) -> pd.DataFrame:
    """
    Filter rows via pandas.query syntax. If expr is None/empty, returns df unchanged.
    Examples:
      expr="age >= 18 and status == 'active'"
      expr="`Total Amount` > 1000 and country in ['NL','BE']"
    """
    if not expr or not str(expr).strip():
        return df
    return df.query(expr, engine="python")


# ------------------------------------------------------------------
# GroupBy + Aggregations
# Accepts:
#   group_cols: list[str]          (e.g., ["country","city"])
#   aggs: dict[str, str|list[str]] (e.g., {"amount":["sum","mean"], "order_id":"count"})
# Returns:
#   Aggregated DataFrame with flattened column names like "sum_amount", "mean_amount", "count_order_id"
# ------------------------------------------------------------------
def groupby_agg(
    df: pd.DataFrame,
    group_cols: list[str] | None,
    aggs: dict[str, str | list[str]] | None,
) -> pd.DataFrame:
    """
    Perform group-by aggregations with deterministic, flat column names.

    Example:
      group_cols = ["country", "city"]
      aggs = {"amount": ["sum", "mean"], "id": "count"}

      Output columns: ["country","city","sum_amount","mean_amount","count_id"]
    """
    if not group_cols or not aggs:
        return df

    # Keep NA groups as groups (match Spark groupBy behavior more closely)
    g = df.groupby(group_cols, dropna=False)
    out = g.agg(aggs)

    # Flatten MultiIndex (pandas) into "func_col" (e.g., "sum_amount")
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [
            f"{func}_{col}" for (col, func) in out.columns.to_flat_index()
        ]
    else:
        out.columns = [str(c) for c in out.columns]

    out = out.reset_index()
    return out


# ------------------------------------------------------------------
# fill NA in selected columns
# ------------------------------------------------------------------
def fillna_cols(
    df: pd.DataFrame,
    value: str | int | float | bool | None,
    subset: list[str] | None = None,
) -> pd.DataFrame:
    """
    Fill NA/None in the provided subset of columns (or all if subset=None).
    Example: fillna_cols(df, 0, ["amount","quantity"])
    """
    if value is None:
        return df
    if subset:
        return df.fillna({col: value for col in subset})
    return df.fillna(value)