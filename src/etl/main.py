from __future__ import annotations
import os
from dotenv import load_dotenv
from .logger import get_logger
from .helpers import (
    make_blob_client,
    download_blob_to_memory,
    upload_blob_from_memory,
    read_csv_bytes,
    write_csv_bytes,
    add_column,
    infer_output_blob_path,
    filter_rows, 
    groupby_agg, 
    fillna_cols,
)

log = get_logger("etl.main")

def main():
    load_dotenv()

    account = os.getenv("AZURE_STORAGE_ACCOUNT")
    container = os.getenv("AZURE_CONTAINER")
    input_blob = os.getenv("INPUT_BLOB_PATH")
    sas_token = os.getenv("AZURE_SAS_TOKEN")

    if not all([account, container, input_blob, sas_token]):
        raise SystemExit("Missing required env vars: AZURE_STORAGE_ACCOUNT, AZURE_CONTAINER, INPUT_BLOB_PATH, AZURE_SAS_TOKEN")

    output_blob = os.getenv("OUTPUT_BLOB_PATH") or infer_output_blob_path(input_blob)

    log.info(f"Reading:  wasbs://{container}@{account}.blob.core.windows.net/{input_blob}")
    bsc = make_blob_client(account, sas_token)
    raw = download_blob_to_memory(bsc, container, input_blob)

    df = read_csv_bytes(raw)
    df = add_column(df, "Column_01", "New_Value_01")
    df = add_column(df, "Time", "")
    df['passed'] = False
    df.loc[df['Score'] > 60, 'passed'] = True
    df = df.query("Score > 10")

    


    out_bytes = write_csv_bytes(df)
    upload_blob_from_memory(bsc, container, output_blob, out_bytes, overwrite=True)

    log.info(f"Wrote:   wasbs://{container}@{account}.blob.core.windows.net/{output_blob}")
    log.info("Done.")

if __name__ == "__main__":
    main()