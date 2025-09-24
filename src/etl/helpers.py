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