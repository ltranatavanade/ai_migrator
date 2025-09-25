# -*- coding: utf-8 -*-
"""
AI Agents – GitHub → Databricks Migration Runner
================================================

This module implements an end‑to‑end, **guardrailed** migration pipeline that:

1) Clones a GitHub repository (HTTPS + PAT).
2) Locates a candidate ETL entry file deterministically (no LLM scanning).
3) Extracts a **deterministic IR** from pandas/PySpark using Python **AST**.
4) Calls **Analyzer** (LLM) to propose a high‑level semantic plan (strict JSON).
5) Calls **Converter** (LLM) to generate a **PySpark SOURCE** Databricks notebook
   that follows **HARD RULES** (no placeholders, correct IO, UC paths, etc.).
6) Validates the notebook and **deploys** it (imports to workspace, creates a Job,
   and optionally runs it).

Artifacts written to ``<cloned_repo>/.artifacts/``:
- ``plan.json`` – analyzer plan (strict JSON)
- ``ir.json`` – deterministic operations extracted from code
- ``generated_notebook.py`` – final notebook source for import

Usage
-----

.. code-block:: bash

    python runner_clean.py --run-now

Environment variables (required)
--------------------------------
- ``AZURE_OPENAI_ENDPOINT``
- ``AZURE_OPENAI_API_KEY``
- ``AZURE_OPENAI_API_VERSION``   (e.g., ``2024-08-01-preview``)
- ``AZURE_OPENAI_DEPLOYMENT``    (chat model deployment name)
- ``DATABRICKS_HOST``            (workspace URL)
- ``DATABRICKS_TOKEN``
- ``GIT_REPO_URL``  | ``GIT_BRANCH`` | ``GIT_TOKEN``
- ``STORAGE_ACCOUNT`` | ``STORAGE_CONTAINER``
- ``INPUT_BLOB_PATH`` | ``OUTPUT_BLOB_PATH``     (relative paths)

Notes
-----
- This script **does not execute** repo code. It only parses with AST.
- All IO uses **Unity Catalog** (`abfss://...`).
"""
from __future__ import annotations

import argparse
import ast
import base64
import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from dotenv import load_dotenv
from git import Repo  # pip install GitPython

# LangChain / LangGraph
from langchain_core.messages import HumanMessage
from langchain_openai import AzureChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command

# Databricks SDK
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import compute as compute_svc
from databricks.sdk.service import jobs as jobs_svc
from databricks.sdk.service import workspace as ws_svc

# =====================================================================
# Configuration
# =====================================================================

@dataclass(frozen=True)
class Config:
    """Holds configuration loaded from environment variables."""

    AOAI_ENDPOINT: str
    AOAI_KEY: str
    AOAI_VERSION: str
    AOAI_DEPLOYMENT: str
    DBX_HOST: str
    DBX_TOKEN: str
    GIT_URL: str
    GIT_BRANCH: str
    GIT_TOKEN: str
    ACCOUNT: str
    CONTAINER: str
    IN_BLOB: str
    OUT_BLOB: str
    JOB_NAME_PREFIX: str = "etl-migrated"
    WORKSPACE_USER_SUBDIR: str = "etl-migration"


def load_env() -> Config:
    """Load and validate configuration from environment variables.

    :raises SystemExit: if any required environment variable is missing.
    :return: ``Config`` instance with values.
    """
    load_dotenv()
    required = [
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_API_VERSION",
        "AZURE_OPENAI_DEPLOYMENT",
        "DATABRICKS_HOST",
        "DATABRICKS_TOKEN",
        "GIT_REPO_URL",
        "GIT_BRANCH",
        "GIT_TOKEN",
        "STORAGE_ACCOUNT",
        "STORAGE_CONTAINER",
        "INPUT_BLOB_PATH",
        "OUTPUT_BLOB_PATH",
    ]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        raise SystemExit(f"Missing required environment variables: {missing}")

    return Config(
        AOAI_ENDPOINT=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
        AOAI_KEY=os.getenv("AZURE_OPENAI_API_KEY", ""),
        AOAI_VERSION=os.getenv("AZURE_OPENAI_API_VERSION", ""),
        AOAI_DEPLOYMENT=os.getenv("AZURE_OPENAI_DEPLOYMENT", ""),
        DBX_HOST=os.getenv("DATABRICKS_HOST", ""),
        DBX_TOKEN=os.getenv("DATABRICKS_TOKEN", ""),
        GIT_URL=os.getenv("GIT_REPO_URL", ""),
        GIT_BRANCH=os.getenv("GIT_BRANCH", "main"),
        GIT_TOKEN=os.getenv("GIT_TOKEN", ""),
        ACCOUNT=os.getenv("STORAGE_ACCOUNT", ""),
        CONTAINER=os.getenv("STORAGE_CONTAINER", ""),
        IN_BLOB=os.getenv("INPUT_BLOB_PATH", ""),
        OUT_BLOB=os.getenv("OUTPUT_BLOB_PATH", ""),
        JOB_NAME_PREFIX=os.getenv("JOB_NAME_PREFIX", "etl-migrated"),
        WORKSPACE_USER_SUBDIR=os.getenv("WORKSPACE_USER_SUBDIR", "etl-migration"),
    )


CFG = load_env()


def build_uc_paths(cfg: Config) -> Dict[str, str]:
    """Construct Unity Catalog ``abfss://`` URIs from the config."""
    return {
        "input_path": f"abfss://{cfg.CONTAINER}@{cfg.ACCOUNT}.dfs.core.windows.net/{cfg.IN_BLOB}",
        "output_path": f"abfss://{cfg.CONTAINER}@{cfg.ACCOUNT}.dfs.core.windows.net/{cfg.OUT_BLOB}",
        "input_format": "csv",
        "output_format": "csv",
        "output_mode": "overwrite",
    }


# =====================================================================
# Git utilities
# =====================================================================

def _mask_token(url: str) -> str:
    """Mask a PAT token in a URL for logging."""
    return re.sub(r":([^@/]{4})[^@/]*@", r":\1***@", url)


def _inject_pat(url: str, token: str) -> str:
    """Insert a PAT into an HTTPS URL for Git clone operations."""
    if not url.startswith("https://"):
        raise ValueError("Use an https:// URL for token-based clone")
    return f"https://x-access-token:{token}@{url[len('https://') :]}"


def clone_repo_to_temp(cfg: Config) -> Path:
    """Clone the repository into a temporary directory at the configured branch."""
    tmp = Path(tempfile.mkdtemp(prefix="repo_"))
    url = _inject_pat(cfg.GIT_URL, cfg.GIT_TOKEN)
    print(f"[Clone] {_mask_token(url)} -> {tmp}")
    Repo.clone_from(url, tmp, branch=cfg.GIT_BRANCH, depth=1)
    return tmp


# =====================================================================
# Databricks helpers (Unity Catalog–ready)
# =====================================================================

def _dbx_client(cfg: Config) -> WorkspaceClient:
    """Create an authenticated Databricks client."""
    return WorkspaceClient(host=cfg.DBX_HOST, token=cfg.DBX_TOKEN)


def _dbx_user_home(cfg: Config) -> str:
    """Return current user's home directory path, e.g., ``/Users/you@example.com``."""
    w = _dbx_client(cfg)
    me = w.current_user.me().user_name
    return f"/Users/{me}"


def import_notebook_source(cfg: Config, workspace_tail: str, source_code: str) -> str:
    """Import SOURCE ``.py`` notebook under ``/Users/<me>/<workspace_tail>``."""
    w = _dbx_client(cfg)
    tail = workspace_tail.lstrip("/")
    nb_path = f"{_dbx_user_home(cfg).rstrip('/')}/{tail}"
    parent = os.path.dirname(nb_path)
    w.workspace.mkdirs(parent)

    b64 = base64.b64encode(source_code.encode("utf-8")).decode("ascii")
    w.workspace.import_(
        path=nb_path,
        format=ws_svc.ImportFormat.SOURCE,
        language=ws_svc.Language.PYTHON,
        content=b64,
        overwrite=True,
    )

    st = w.workspace.get_status(nb_path)
    if st.object_type != ws_svc.ObjectType.NOTEBOOK:
        raise RuntimeError("Imported object is not a notebook.")
    return nb_path


def create_job_for_notebook(
    cfg: Config, *, job_name: str, notebook_path: str, base_params: Optional[Dict[str, str]] = None
) -> int:
    """Create a small job pointing to the given notebook and return the job ID."""
    w = _dbx_client(cfg)
    cluster = compute_svc.ClusterSpec(
        spark_version=w.clusters.select_spark_version(long_term_support=True),
        node_type_id=w.clusters.select_node_type(local_disk=True),
        num_workers=1,
    )
    created = w.jobs.create(
        name=job_name,
        tasks=[
            jobs_svc.Task(
                task_key="main",
                new_cluster=cluster,
                notebook_task=jobs_svc.NotebookTask(
                    notebook_path=notebook_path,
                    base_parameters=base_params or {},
                ),
            )
        ],
    )
    return created.job_id


def run_job_and_wait(cfg: Config, job_id: int, notebook_params: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Run a job now and wait for a terminal state, returning a summary dict."""
    w = _dbx_client(cfg)
    waiter = w.jobs.run_now(job_id=job_id, notebook_params=notebook_params or {})
    final = w.jobs.wait_get_run_job_terminated_or_skipped(run_id=waiter.run_id)
    return {
        "run_id": waiter.run_id,
        "life_cycle_state": str(final.state.life_cycle_state),
        "result_state": str(final.state.result_state),
    }


# =====================================================================
# Local filesystem utilities (artifacts)
# =====================================================================

def save_artifact(repo_path: str, filename: str, content: str) -> str:
    """Save content under ``<repo_path>/.artifacts/<filename>`` and return the path."""
    out_dir = Path(repo_path) / ".artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / filename
    p.write_text(content, encoding="utf-8")
    return str(p)


# =====================================================================
# ETL entry selection & AST → IR extraction
# =====================================================================

def find_candidate_etl_file(repo_path: str) -> Optional[Path]:
    """Return a best‑effort ETL entry file (prefers ``src/etl/main.py``)."""
    base = Path(repo_path)
    preferred = base / "src" / "etl" / "main.py"
    if preferred.exists():
        return preferred
    for p in (base / "src" / "etl").rglob("*.py"):
        return p
    for p in base.rglob("main.py"):
        return p
    return None


# --- AST helpers -------------------------------------------------------

def _lit_string(node: ast.AST) -> Optional[str]:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _is_col_call(n: ast.AST) -> bool:
    """Return True if node matches ``col("x")`` or ``F.col("x")``."""
    return (
        isinstance(n, ast.Call)
        and (
            (isinstance(n.func, ast.Name) and n.func.id == "col")
            or (isinstance(n.func, ast.Attribute) and n.func.attr == "col")
        )
        and len(n.args) == 1
        and isinstance(n.args[0], ast.Constant)
        and isinstance(n.args[0].value, str)
    )


def _col_from_subscript(n: ast.AST) -> Optional[str]:
    """Return the column name if node is ``df['col']``; else ``None``."""
    if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name):
        s = n.slice if not hasattr(n.slice, "value") else n.slice.value
        if isinstance(s, ast.Constant) and isinstance(s.value, str):
            return s.value
    return None


def _emit_expr(n: ast.AST) -> str:
    """Serialize a subset of pandas/boolean AST into a PySpark‑style expression string."""
    col_name = _col_from_subscript(n)
    if col_name:
        return f"F.col('{col_name}')"
    if isinstance(n, ast.Call) and _is_col_call(n):
        return f"F.col('{n.args[0].value}')"
    if isinstance(n, ast.Constant):
        v = n.value
        if isinstance(v, str):
            return f"'{v}'"
        return str(v)
    if isinstance(n, ast.Compare) and len(n.ops) == 1 and len(n.comparators) == 1:
        left = _emit_expr(n.left)
        right = _emit_expr(n.comparators[0])
        op = n.ops[0]
        if isinstance(op, ast.Eq):
            sym = "="
        elif isinstance(op, ast.NotEq):
            sym = "!="
        elif isinstance(op, ast.Lt):
            sym = "<"
        elif isinstance(op, ast.LtE):
            sym = "<="
        elif isinstance(op, ast.Gt):
            sym = ">"
        elif isinstance(op, ast.GtE):
            sym = ">="
        else:
            sym = "??"
        return f"({left} {sym} {right})"
    if isinstance(n, ast.BinOp):  # pandas uses & / | for boolean chaining
        if isinstance(n.op, ast.BitAnd):
            return f"({_emit_expr(n.left)} AND {_emit_expr(n.right)})"
        if isinstance(n.op, ast.BitOr):
            return f"({_emit_expr(n.left)} OR {_emit_expr(n.right)})"
    if isinstance(n, ast.BoolOp):
        sep = " AND " if isinstance(n.op, ast.And) else " OR "
        return "(" + sep.join(_emit_expr(v) for v in n.values) + ")"
    if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.Invert):
        return f"(NOT {_emit_expr(n.operand)})"
    return "<unsupported_expr>"


def _is_np_where_call(n: ast.AST) -> bool:
    return (
        isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and isinstance(n.func.value, ast.Name)
        and n.func.value.id in {"np", "numpy"}
        and n.func.attr == "where"
        and len(n.args) == 3
    )


def _lit_value_expr(v: ast.AST) -> Optional[str]:
    if isinstance(v, ast.Constant):
        x = v.value
        if isinstance(x, str):
            return "F.lit('" + x.replace("'", "\\'") + "')"
        if isinstance(x, (int, float, bool)) or x is None:
            return f"F.lit({repr(x)})"
    return None


# --- Extractors --------------------------------------------------------

def _extract_filter_ops(tree: ast.AST) -> List[Dict[str, Any]]:
    ops: List[Dict[str, Any]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in {"filter", "where"} and node.args:
                arg = node.args[0]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    ops.append({"op": "filter", "expr": arg.value})
                else:
                    expr = _emit_expr(arg)
                    if expr != "<unsupported_expr>":
                        ops.append({"op": "filter", "expr": expr})
        if isinstance(node, ast.Subscript):  # df[ <mask> ]
            s = node.slice if not hasattr(node.slice, "value") else node.slice.value
            if isinstance(s, (ast.BoolOp, ast.BinOp, ast.Compare, ast.UnaryOp, ast.Call)):
                expr = _emit_expr(s)
                if expr != "<unsupported_expr>":
                    ops.append({"op": "filter", "expr": expr})
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "query" and node.args:
                q = node.args[0]
                if isinstance(q, ast.Constant) and isinstance(q.value, str):
                    ops.append({"op": "filter", "expr": q.value.replace("==", "=")})
    return ops


def _extract_assign_ops(tree: ast.AST) -> List[Dict[str, Any]]:
    ops: List[Dict[str, Any]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            tgt = node.targets[0]
            col = _col_from_subscript(tgt)
            if col:
                if _is_np_where_call(node.value):
                    cond, tval, fval = node.value.args
                    cond_expr = _emit_expr(cond)
                    t_expr = _lit_value_expr(tval)
                    f_expr = _lit_value_expr(fval)
                    if cond_expr != "<unsupported_expr>" and t_expr and f_expr:
                        expr = f"F.when({cond_expr}, {t_expr}).otherwise({f_expr})"
                        ops.append({"op": "withColumnExpr", "name": col, "expr": expr})
                    continue
                val_expr = _lit_value_expr(node.value)
                if val_expr:
                    ops.append({"op": "withColumnExpr", "name": col, "expr": val_expr})
        # df.loc[mask, 'col'] = v
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            tgt = node.targets[0]
            if isinstance(tgt, ast.Subscript) and isinstance(tgt.value, ast.Attribute) and tgt.value.attr == "loc":
                sl = tgt.slice if not hasattr(tgt.slice, "value") else tgt.slice.value
                if isinstance(sl, ast.Tuple) and len(sl.elts) == 2:
                    mask_node, col_node = sl.elts
                    if isinstance(col_node, ast.Constant) and isinstance(col_node.value, str):
                        col = col_node.value
                        mask_expr = _emit_expr(mask_node)
                        val_expr = _lit_value_expr(node.value)
                        if mask_expr != "<unsupported_expr>" and val_expr:
                            expr = f"F.when({mask_expr}, {val_expr}).otherwise(F.col('{col}'))"
                            ops.append({"op": "withColumnExpr", "name": col, "expr": expr})
    return ops


def _lit_list(args: List[ast.AST]) -> List[str]:
    vals: List[str] = []
    for a in args:
        if isinstance(a, ast.Constant) and isinstance(a.value, str):
            vals.append(a.value)
    return vals


def _collect_group_cols(args: List[ast.AST]) -> List[str]:
    if not args:
        return []
    a0 = args[0]
    if isinstance(a0, (ast.List, ast.Tuple)):
        return _lit_list(a0.elts)
    return _lit_list(args)


def _parse_alias_call(alias_call: ast.Call) -> Optional[Dict[str, str]]:
    if not (isinstance(alias_call, ast.Call) and isinstance(alias_call.func, ast.Attribute) and alias_call.func.attr == "alias"):
        return None
    fcall = alias_call.func.value
    if not isinstance(fcall, ast.Call):
        return None
    func_name = fcall.func.attr if isinstance(fcall.func, ast.Attribute) else getattr(fcall.func, "id", None)
    col = fcall.args[0].value if (fcall.args and isinstance(fcall.args[0], ast.Constant) and isinstance(fcall.args[0].value, str)) else None
    alias = alias_call.args[0].value if (alias_call.args and isinstance(alias_call.args[0], ast.Constant) and isinstance(alias_call.args[0].value, str)) else None
    if func_name and col and alias:
        return {"func": func_name, "col": col, "alias": alias}
    return None


def _extract_groupby_ops(tree: ast.AST) -> List[Dict[str, Any]]:
    ops: List[Dict[str, Any]] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "agg"):
            continue
        gb_call = node.func.value
        if not (isinstance(gb_call, ast.Call) and isinstance(gb_call.func, ast.Attribute) and gb_call.func.attr in {"groupBy", "groupby"}):
            continue
        group_cols = _collect_group_cols(gb_call.args)
        aggs: List[Dict[str, str]] = []
        if node.args and isinstance(node.args[0], ast.Dict):
            for k, v in zip(node.args[0].keys, node.args[0].values):
                if isinstance(k, ast.Constant) and isinstance(k.value, str):
                    col = k.value
                    if isinstance(v, ast.Constant) and isinstance(v.value, str):
                        func = v.value
                        alias = f"{func}_{col}"
                        aggs.append({"func": func, "col": col, "alias": alias})
        for a in node.args:
            if isinstance(a, ast.Call):
                parsed = _parse_alias_call(a)
                if parsed:
                    aggs.append(parsed)
        if node.args and isinstance(node.args[0], ast.Dict):
            for k, v in zip(node.args[0].keys, node.args[0].values):
                if isinstance(k, ast.Constant) and isinstance(k.value, str) and isinstance(v, (ast.List, ast.Tuple)):
                    col = k.value
                    for vv in v.elts:
                        if isinstance(vv, ast.Constant) and isinstance(vv.value, str):
                            func = vv.value
                            alias = f"{func}_{col}"
                            aggs.append({"func": func, "col": col, "alias": alias})
        if group_cols and aggs:
            ops.append({"op": "groupByAgg", "group_cols": group_cols, "aggs": aggs})
    return ops


def _extract_groupby_reduce_ops(tree: ast.AST) -> List[Dict[str, Any]]:
    ops: List[Dict[str, Any]] = []
    reducers = {"mean", "sum", "count", "min", "max", "avg", "median"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in reducers:
            gb_call = node.func.value
            if isinstance(gb_call, ast.Call) and isinstance(gb_call.func, ast.Attribute) and gb_call.func.attr in {"groupby", "groupBy"}:
                group_cols = _collect_group_cols(gb_call.args)
                if group_cols:
                    ops.append({"op": "groupByReduce", "group_cols": group_cols, "func": node.func.attr})
    return ops


def extract_ir_operations_from_python(code: str) -> List[Dict[str, Any]]:
    """Extract deterministic IR operations from Python code."""
    ops: List[Dict[str, Any]] = []
    try:
        tree = ast.parse(code)
    except Exception:
        return ops

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func_name = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", None)
            if func_name == "add_column" and len(node.args) >= 3:
                name = _lit_string(node.args[1])
                value = _lit_string(node.args[2])
                if name is not None:
                    ops.append({"op": "add_column", "name": name, "value": value})

    ops.extend(_extract_filter_ops(tree))
    ops.extend(_extract_assign_ops(tree))
    ops.extend(_extract_groupby_ops(tree))
    ops.extend(_extract_groupby_reduce_ops(tree))
    return ops


# =====================================================================
# Helper discovery & source loading
# =====================================================================

def find_helper_modules_in_code(code: str) -> List[str]:
    """Return helper module paths discovered in ``import`` / ``from`` statements."""
    modules: List[str] = []
    try:
        tree = ast.parse(code)
    except Exception:
        return modules
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if getattr(alias, "name", "").endswith(".helpers"):
                    modules.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.endswith(".helpers"):
                modules.append(node.module)
            for n in node.names or []:
                if n.name == "helpers":
                    modules.append(f"{node.module}.helpers")
    # de-dup
    out: List[str] = []
    for m in modules:
        if m not in out:
            out.append(m)
    return out


def module_to_file(repo_path: str, module_path: str) -> Optional[Path]:
    """Map module name (``src.etl.helpers``) to a file under the repo."""
    p = Path(repo_path) / Path(module_path.replace(".", "/") + ".py")
    return p if p.exists() else None


def read_helper_sources(
    repo_path: str,
    code: str,
    *,
    extra_candidates: Optional[List[str]] = None,
    max_per_file: int = 4000,
) -> List[Dict[str, str]]:
    """Read helper module sources referenced by the ETL code."""
    modules = find_helper_modules_in_code(code)
    if extra_candidates:
        for m in extra_candidates:
            if m not in modules:
                modules.append(m)
    files: List[Path] = []
    for m in modules:
        f = module_to_file(repo_path, m)
        if f:
            files.append(f)
    seen = set()
    payload: List[Dict[str, str]] = []
    for f in files:
        sp = str(f.resolve())
        if sp in seen:
            continue
        seen.add(sp)
        try:
            src = f.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            src = ""
        payload.append({"path": sp, "source": src[:max_per_file]})
    return payload


# =====================================================================
# Notebook validation
# =====================================================================

FORBIDDEN_TOKENS = {
    "...",
    "<INPUT_PATH>",
    "<OUTPUT_PATH>",
    "path_to_input_csv_file.csv",
    "path_to_output_csv_file.csv",
    "path_to_input_json_file.json",
    "path_to_output_json_file.json",
}


def validate_nb_source(
    nb_src: str,
    input_path: str,
    output_path: str,
    *,
    required_columns: Optional[List[str]] = None,
) -> None:
    """Fail‑fast validation for the generated Databricks notebook source."""
    if not nb_src or len(nb_src) < 100:
        raise ValueError("Notebook generation failed: source is too short.")
    s = nb_src.strip().lower()
    if not s.startswith("# databricks notebook source"):
        raise ValueError("Notebook must begin with '# Databricks notebook source'.")
    if "from pyspark.sql import functions as f" not in s:
        raise ValueError("Notebook must import 'from pyspark.sql import functions as F'.")
    if "spark.read" not in s:
        raise ValueError("Notebook must use spark.read to read input.")
    if ".write" not in s:
        raise ValueError("Notebook must write the output (df.write...).")
    for token in FORBIDDEN_TOKENS:
        if token in nb_src:
            raise ValueError(f"Notebook contains placeholder token '{token}'.")
    if input_path not in nb_src:
        raise ValueError("Notebook does not include the required INPUT_PATH.")
    if output_path not in nb_src:
        raise ValueError("Notebook does not include the required OUTPUT_PATH.")
    if required_columns:
        for col in required_columns:
            if col not in nb_src:
                raise ValueError(f"Notebook missing required column name '{col}' from IR.")


# =====================================================================
# LLM setup
# =====================================================================

AOAI = AzureChatOpenAI(
    deployment_name=CFG.AOAI_DEPLOYMENT,
    model=CFG.AOAI_DEPLOYMENT,
    api_key=CFG.AOAI_KEY,
    azure_endpoint=CFG.AOAI_ENDPOINT,
    api_version=CFG.AOAI_VERSION,
    temperature=0.2,
    max_tokens=700,
)

ANALYZER_PROMPT = """You are an ETL analyzer.
You will be given:
- ETL_SNIPPET: short Python code from the repo (pandas or PySpark, may call helper functions).
- HELPERS: the actual source code of helper functions (e.g., add_column, filter_rows, fill_missing, etc.).
Goal:
Return STRICT JSON ONLY (no prose) describing a neutral plan for a Spark/PySpark notebook:
{ "entrypoints": [ { "source_path": "<relative or null>", "inputs": [{"path": null, "format": "csv\njson\nparquet\ndelta\nnull"}], "outputs": [{"path": null, "format": "csv\njson\nparquet\ndelta\nnull", "mode": "overwrite\nappend\nnull"}], "operations": [ {"type": "withColumnExpr", "params": {"name": "ColName", "expr": "F.lit('Value')"}}, {"type": "filter", "params": {"expr": "(col('a') > 5) AND (col('b') = 'x')"}}, {"type": "select", "params": {"columns": ["a","b","c"]}}, {"type": "drop", "params": {"columns": ["c"]}}, {"type": "rename", "params": {"mapping": {"old":"new"}}}, {"type": "fillna", "params": {"value": "0", "subset": ["col1","col2"]}}, {"type": "withColumnExpr","params": {"name": "Col2", "expr": "F.col('a') + F.col('b')"}}, {"type": "groupByAgg", "params": {"group_cols": ["a","b"], "aggs": [{"func":"sum","col":"x","alias":"sum_x"}]}}, {"type": "orderBy", "params": {"cols": ["a","b"]}}, {"type": "window", "params": {"partitionBy":["a"], "orderBy":["ts"], "frame":"rowsBetween(-1, 1)"}}, {"type": "join", "params": {"target_alias":"<omit if not available>", "on":["a"], "how":"left"}} ], "helpers": [ {"module_path": "src/etl/helpers.py"} ] } ] }
Rules:
- Do not hardcode mappings; infer from HELPERS and callsites.
- If unknown, emit {"type":"unknown_helper", ...} with a reason.
- Use null (not "unknown") when path/format/mode are unknown.
- Output must be valid JSON only.
"""

CONVERTER_PROMPT = """You are a senior Spark data engineer.
You will be given:
- An INTERMEDIATE REPRESENTATION (IR) of transformations to apply (generic: add_column, filter, join, select, drop, rename, withColumnExpr, fillna, groupBy/agg, orderBy, window, limit, union, distinct, etc.).
- The INPUT_PATH, OUTPUT_PATH and formats/mode (Unity Catalog abfss:// URIs).
- A SOURCE_SNIPPET (original Python ETL) for context. If IR is incomplete, align with the SOURCE_SNIPPET semantics.
Generate a Databricks SOURCE .py notebook that:
HARD RULES:
- Begin with: "# Databricks notebook source"
- Use PySpark only (no pandas). Import: from pyspark.sql import functions as F
- Read input using INPUT_PATH and INPUT_FORMAT:
  * CSV → spark.read.option("header","true").csv(INPUT_PATH)
  * JSON → spark.read.json(INPUT_PATH)
  * Parquet/Delta → spark.read.format(fmt).load(INPUT_PATH)
- Apply ALL operations in IR in order:
  * add_column(name, value): if value is null/empty, use F.current_timestamp(); else F.lit(value).
  * filter(expr): df = df.filter(<expr>) using F.col(...)
  * select(columns): df = df.select(*[F.col(c) for c in columns])
  * drop(columns): df = df.drop(*columns)
  * rename(mapping): df = df.select([F.col(k).alias(v) if k in mapping else F.col(k) for k in df.columns])
  * withColumnExpr(name, expr): df = df.withColumn(name, expr)
  * fillna(value, subset): df = df.fillna(value, subset)
  * groupByAgg(group_cols, aggs): df = df.groupBy(*group_cols).agg(*aggs)
  * groupByReduce(group_cols, func): apply func to all numeric columns
  * orderBy(cols): df = df.orderBy(*[F.col(c) for c in cols])
  * limit(n): df = df.limit(n)
- Write output using OUTPUT_PATH, OUTPUT_FORMAT, OUTPUT_MODE.
- Do NOT use widgets, placeholders, or ellipses.
- Do NOT invent extra inputs or columns.
- INPUT_PATH and OUTPUT_PATH must appear as literal strings in the code.
- If INPUT_PATH or OUTPUT_PATH are missing, raise ValueError("Missing input/output path").
- If SOURCE_SNIPPET contains pandas idioms not in IR, implement them:
  * df['col'] = v → df = df.withColumn('col', F.lit(v))
  * df.loc[mask, 'col'] = v → df = df.withColumn('col', F.when(mask, F.lit(v)).otherwise(F.col('col')))
  * df['col'] = np.where(c, a, b) → df = df.withColumn('col', F.when(c, F.lit(a)).otherwise(F.lit(b)))
- Boolean masks: use F.col(...) and PySpark operators; support & (AND) and | (OR).
- For groupby().mean()/sum()/count() with no explicit columns: aggregate numeric columns only.
Return ONLY the notebook source; no explanations.
"""

ANALYZER = create_react_agent(AOAI, tools=[], prompt=ANALYZER_PROMPT)
CONVERTER = create_react_agent(AOAI, tools=[], prompt=CONVERTER_PROMPT)


# =====================================================================
# Graph state & nodes
# =====================================================================

class State(MessagesState):
    """LangGraph state for the migration workflow."""

    next: str
    repo_path: Optional[str]
    etl_file_path: Optional[str]
    plan: Optional[str]
    notebook_source: Optional[str]
    run_now: Optional[bool]


def bootstrap_node(state: State) -> Command[Literal["etl_analyzer"]]:
    """Select ETL entry file deterministically (no LLM)."""
    repo_path = state.get("repo_path") or "."
    etl_file = find_candidate_etl_file(repo_path)
    if not etl_file or not etl_file.exists():
        raise ValueError("No ETL entry file found. Expected 'src/etl/main.py' or any '*.py' under 'src/etl/'.")
    return Command(update={
        "etl_file_path": str(etl_file),
        "messages": [HumanMessage(content=f"ETL entry selected: {etl_file}")],
    }, goto="etl_analyzer")


def etl_analyzer_node(state: State) -> Command[Literal["converter"]]:
    """Feed ETL snippet + helper sources to the analyzer LLM and persist its plan."""
    repo_path = state.get("repo_path") or "."
    etl_path = state.get("etl_file_path")
    if not etl_path:
        raise ValueError("Missing etl_file_path; bootstrap must run first.")

    etl_code = Path(etl_path).read_text(encoding="utf-8", errors="ignore")
    etl_snippet = etl_code[:1200]

    helpers_payload = read_helper_sources(repo_path=repo_path, code=etl_code, extra_candidates=["src.etl.helpers"])

    ctx = {
        "ETL_SNIPPET": etl_snippet,
        "HELPERS": helpers_payload,
        "PREFERRED_NEUTRAL_OPS": [
            "withColumnExpr", "filter", "select", "drop", "rename",
            "fillna", "groupByAgg", "orderBy", "window", "join", "limit",
            "union", "unionByName", "distinct",
        ],
    }

    res = ANALYZER.invoke({**state, "messages": [
        HumanMessage(content="Analyze ETL_SNIPPET and HELPERS; emit STRICT JSON plan only."),
        HumanMessage(content=json.dumps(ctx, indent=2)),
    ]})

    content = res["messages"][-1].content if hasattr(res["messages"][-1], "content") else str(res["messages"][-1])
    plan_dict = _safe_json_loads(content)
    save_artifact(repo_path, "plan.json", json.dumps(plan_dict, indent=2))

    return Command(update={
        "messages": [HumanMessage(content="Plan (semantic) extracted.", name="etl_analyzer")],
        "plan": json.dumps(plan_dict),
    }, goto="converter")


def _collect_assigned_columns(code: str) -> List[str]:
    """Return columns assigned via pandas syntax (e.g., ``df['col'] = ...``)."""
    cols: List[str] = []
    try:
        tree = ast.parse(code)
    except Exception:
        return cols
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            c = _col_from_subscript(node.targets[0])
            if c:
                cols.append(c)
    out: List[str] = []
    for c in cols:
        if c not in out:
            out.append(c)
    return out


def converter_node(state: State) -> Command[Literal["deployer"]]:
    """Generate, validate, and persist the Databricks notebook source."""
    repo_path = state.get("repo_path") or "."
    etl_path = state.get("etl_file_path")
    if not etl_path:
        raise ValueError("Missing etl_file_path; bootstrap must run first.")

    uc = build_uc_paths(CFG)
    input_path = uc["input_path"]
    output_path = uc["output_path"]
    input_fmt = uc["input_format"]
    output_fmt = uc["output_format"]
    output_mode = uc["output_mode"]

    code = Path(etl_path).read_text(encoding="utf-8", errors="ignore")
    ir_ops = extract_ir_operations_from_python(code)

    required_cols = [op["name"] for op in ir_ops if op.get("op") == "add_column" and op.get("name")]
    required_cols += _collect_assigned_columns(code)

    save_artifact(repo_path, "ir.json", json.dumps(ir_ops, indent=2))

    IR = {
        "input": {"path": input_path, "format": input_fmt},
        "output": {"path": output_path, "format": output_fmt, "mode": output_mode},
        "operations": ir_ops,
    }

    context_msg = {
        "IR": IR,
        "INPUT_PATH": input_path,
        "OUTPUT_PATH": output_path,
        "INPUT_FORMAT": input_fmt,
        "OUTPUT_FORMAT": output_fmt,
        "OUTPUT_MODE": output_mode,
        "SOURCE_SNIPPET": code[:800],
    }

    tries = 0
    nb_src: Optional[str] = None
    last_err: Optional[str] = None

    while tries < 2:
        tries += 1
        res = CONVERTER.invoke({**state, "messages": [
            HumanMessage(content="Use this IR and SOURCE_SNIPPET to generate the notebook (follow HARD RULES)."),
            HumanMessage(content=json.dumps(context_msg, indent=2)),
        ]})
        nb_src = res["messages"][-1].content if hasattr(res["messages"][-1], "content") else str(res["messages"][-1])
        try:
            validate_nb_source(nb_src, input_path, output_path, required_columns=required_cols)
            break
        except Exception as e:
            last_err = str(e)
            state["messages"] = state.get("messages", []) + [
                HumanMessage(content=f"Previous notebook violated constraints: {last_err}. Regenerate strictly following HARD RULES and IR."),
            ]
            nb_src = None

    if nb_src is None:
        raise ValueError(f"Notebook generation failed after retries. Last error: {last_err}")

    save_artifact(repo_path, "generated_notebook.py", nb_src)

    return Command(update={
        "messages": [HumanMessage(content="Notebook generated by LLM (no placeholders, UC paths).", name="converter")],
        "notebook_source": nb_src,
    }, goto="deployer")


def deployer_node(state: State) -> Command[Literal["__end__"]]:
    """Import the notebook, create a job, optionally run it, and emit a summary."""
    repo_path = state.get("repo_path") or "."
    nb_src = state.get("notebook_source") or ""
    if not nb_src:
        raise ValueError("No notebook_source found; converter must run first.")

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    tail = f"{CFG.WORKSPACE_USER_SUBDIR.strip('/')}/etl_migrated_{ts}"
    job_name = f"{CFG.JOB_NAME_PREFIX}-etl_migrated"

    nb_path = import_notebook_source(CFG, tail, nb_src)
    job_id = create_job_for_notebook(CFG, job_name=job_name, notebook_path=nb_path, base_params=None)

    result: Dict[str, Any] = {"import_path": nb_path, "job_id": job_id}
    if state.get("run_now"):
        result["run"] = run_job_and_wait(CFG, job_id, notebook_params=None)

    print("\n=== DEPLOY RESULT ===")
    print(json.dumps(result, indent=2))
    return Command(update={"messages": [HumanMessage(content=json.dumps(result), name="deployer")]}, goto=END)


# =====================================================================
# Utilities
# =====================================================================

def _safe_json_loads(s: str) -> Dict[str, Any]:
    """Load JSON, attempting to salvage the last ``{...}`` object if needed."""
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"\{[\s\S]*\}$", s.strip(), re.MULTILINE)
        if m:
            return json.loads(m.group(0))
        raise


def trim_messages(state: State, keep_last: int = 4) -> State:
    """Return a state that keeps only the last ``keep_last`` messages."""
    msgs = state.get("messages", [])
    if len(msgs) > keep_last:
        state["messages"] = msgs[-keep_last:]
    return state


# =====================================================================
# Graph builder & CLI
# =====================================================================

def build_graph():
    """Compile the LangGraph workflow for this runner."""
    builder = StateGraph(State)
    builder.add_node("bootstrap", bootstrap_node)
    builder.add_node("etl_analyzer", etl_analyzer_node)
    builder.add_node("converter", converter_node)
    builder.add_node("deployer", deployer_node)
    builder.add_edge(START, "bootstrap")
    builder.add_edge("bootstrap", "etl_analyzer")
    builder.add_edge("etl_analyzer", "converter")
    builder.add_edge("converter", "deployer")
    builder.add_edge("deployer", END)
    return builder.compile()


def main() -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="GitHub → Databricks migration runner")
    parser.add_argument("--run-now", action="store_true", help="Run the Databricks job after creation")
    args = parser.parse_args()

    repo_path = clone_repo_to_temp(CFG)
    graph = build_graph()

    user_msg = "Please migrate the simplest ETL Python script from this cloned repo."
    result = graph.invoke({
        "messages": [{"role": "user", "content": user_msg}],
        "repo_path": str(repo_path),
        "run_now": bool(args.run_now),
    })

    print("\n=== FINAL MESSAGE ===")
    last = result["messages"][-1]
    print(last.content if hasattr(last, "content") else last)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
