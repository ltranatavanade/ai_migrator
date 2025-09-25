# ai_migrate_uc_agents.py
# -*- coding: utf-8 -*-

"""
End-to-end multi-agent pipeline that:
1) Clones a GitHub repo (HTTPS + PAT).
2) Selects an ETL entry python file programmatically (no LLM scanning).
3) Analyzer agent (LLM) reads a tiny snippet and produces a strict JSON PLAN.
4) Converter agent (LLM) generates a full Databricks SOURCE .py notebook,
   taking inspiration from the repo snippet + PLAN; NO templates, NO placeholders.
5) Deployer imports the notebook to your Databricks workspace (UC-enabled),
   creates a Job, and (optionally) runs it.

Artifacts saved under:  <cloned_repo>/.artifacts/
  - plan.json          (LLM PLAN)
  - ir.json            (deterministic IR extracted from Python via AST)
  - generated_notebook.py

Requirements (env in the agent runner, NOT in the ETL repo):
  AZURE_OPENAI_ENDPOINT
  AZURE_OPENAI_API_KEY
  AZURE_OPENAI_API_VERSION (e.g., 2024-08-01-preview)
  AZURE_OPENAI_DEPLOYMENT  (chat model deployment name)
  DATABRICKS_HOST          (per-workspace URL)
  DATABRICKS_TOKEN
  GIT_REPO_URL, GIT_BRANCH, GIT_TOKEN
  STORAGE_ACCOUNT, STORAGE_CONTAINER
  INPUT_BLOB_PATH, OUTPUT_BLOB_PATH  (relative blob paths)

Run:
  python ai_migrate_uc_agents.py --run-now
"""

from __future__ import annotations

import os
import re
import json
import base64
import argparse
import tempfile
from pathlib import Path
from typing import Optional, Literal, List, Dict, Any

import ast
from dotenv import load_dotenv
from git import Repo  # pip install GitPython

# LangChain / LangGraph
from typing_extensions import TypedDict
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.types import Command
from langgraph.prebuilt import create_react_agent

# Databricks SDK
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import workspace as ws_svc
from databricks.sdk.service import jobs as jobs_svc
from databricks.sdk.service import compute as compute_svc


# =======================================================
# Env & Config
# =======================================================
def load_env() -> Dict[str, str]:
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

    return {
        "AOAI_ENDPOINT": os.getenv("AZURE_OPENAI_ENDPOINT"),
        "AOAI_KEY": os.getenv("AZURE_OPENAI_API_KEY"),
        "AOAI_VERSION": os.getenv("AZURE_OPENAI_API_VERSION"),
        "AOAI_DEPLOYMENT": os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        "DBX_HOST": os.getenv("DATABRICKS_HOST"),
        "DBX_TOKEN": os.getenv("DATABRICKS_TOKEN"),
        "GIT_URL": os.getenv("GIT_REPO_URL"),
        "GIT_BRANCH": os.getenv("GIT_BRANCH", "main"),
        "GIT_TOKEN": os.getenv("GIT_TOKEN"),
        "ACCOUNT": os.getenv("STORAGE_ACCOUNT"),
        "CONTAINER": os.getenv("STORAGE_CONTAINER"),
        "IN_BLOB": os.getenv("INPUT_BLOB_PATH"),
        "OUT_BLOB": os.getenv("OUTPUT_BLOB_PATH"),
        "JOB_NAME_PREFIX": os.getenv("JOB_NAME_PREFIX", "etl-migrated"),
        "WORKSPACE_USER_SUBDIR": os.getenv("WORKSPACE_USER_SUBDIR", "etl-migration"),
    }


CFG = load_env()


def build_uc_paths() -> Dict[str, str]:
    """Construct UC abfss:// URIs from env."""
    account = CFG["ACCOUNT"]
    container = CFG["CONTAINER"]
    in_blob = CFG["IN_BLOB"]
    out_blob = CFG["OUT_BLOB"]
    return {
        "input_path": f"abfss://{container}@{account}.dfs.core.windows.net/{in_blob}",
        "output_path": f"abfss://{container}@{account}.dfs.core.windows.net/{out_blob}",
        "input_format": "csv",
        "output_format": "csv",
        "output_mode": "overwrite",
    }


# =======================================================
# Git clone helpers
# =======================================================
def mask_token(url: str) -> str:
    return re.sub(r":([^@/]{4})[^@/]*@", r":\1***@", url)


def inject_pat(url: str, token: str) -> str:
    """Insert PAT into https URL."""
    if not url.startswith("https://"):
        raise ValueError("Use an https:// URL for token-based clone")
    return f"https://x-access-token:{token}@{url[len('https://'):]}"


def clone_repo_to_temp() -> Path:
    """Clone repo at branch into a temp directory."""
    tmp = Path(tempfile.mkdtemp(prefix="repo_"))
    url = inject_pat(CFG["GIT_URL"], CFG["GIT_TOKEN"])
    print(f"[Clone] {mask_token(url)} -> {tmp}")
    Repo.clone_from(url, tmp, branch=CFG["GIT_BRANCH"], depth=1)
    return tmp


# =======================================================
# Databricks helpers (UC-ready)
# =======================================================
def dbx_client() -> WorkspaceClient:
    """Create an authenticated Databricks client."""
    return WorkspaceClient(host=CFG["DBX_HOST"], token=CFG["DBX_TOKEN"])


def dbx_user_home() -> str:
    """Return current user's home, e.g., /Users/you@company.com."""
    w = dbx_client()
    me = w.current_user.me().user_name
    return f"/Users/{me}"


def import_notebook_source(workspace_tail: str, source_code: str) -> str:
    """
    Import SOURCE .py notebook under /Users/<me>/<workspace_tail>.
    Returns absolute path. Raises on error.
    """
    w = dbx_client()
    tail = workspace_tail.lstrip("/")
    nb_path = f"{dbx_user_home().rstrip('/')}/{tail}"
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
    # Verify
    st = w.workspace.get_status(nb_path)
    if st.object_type != ws_svc.ObjectType.NOTEBOOK:
        raise RuntimeError("Imported object is not a notebook.")
    return nb_path


def create_job_for_notebook(job_name: str, notebook_path: str, base_params: Optional[Dict[str, str]] = None) -> int:
    """Create a small job pointing to the given notebook; return job_id."""
    w = dbx_client()
    cluster = compute_svc.ClusterSpec(
        spark_version=w.clusters.select_spark_version(long_term_support=True),
        node_type_id=w.clusters.select_node_type(local_disk=True),
        num_workers=1,
    )
    created = w.jobs.create(
        name=job_name,
        tasks=[jobs_svc.Task(
            task_key="main",
            new_cluster=cluster,
            notebook_task=jobs_svc.NotebookTask(
                notebook_path=notebook_path,
                base_parameters=base_params or {},
            ),
        )],
    )
    return created.job_id


def run_job_and_wait(job_id: int, notebook_params: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Run a job now and wait for terminal state."""
    w = dbx_client()
    waiter = w.jobs.run_now(job_id=job_id, notebook_params=notebook_params or {})
    final = w.jobs.wait_get_run_job_terminated_or_skipped(run_id=waiter.run_id)
    return {
        "run_id": waiter.run_id,
        "life_cycle_state": str(final.state.life_cycle_state),
        "result_state": str(final.state.result_state),
    }


# =======================================================
# Local FS utilities (artifacts)
# =======================================================
def save_artifact(repo_path: str, filename: str, content: str) -> str:
    """Save content under <repo_path>/.artifacts/filename."""
    out_dir = Path(repo_path) / ".artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / filename
    p.write_text(content, encoding="utf-8")
    return str(p)


# =======================================================
# ETL entry selection & IR extraction (deterministic)
# =======================================================
def find_candidate_etl_file(repo_path: str) -> Optional[Path]:
    """Prefer src/etl/main.py; else first *.py under src/etl, else any main.py."""
    base = Path(repo_path)
    preferred = base / "src" / "etl" / "main.py"
    if preferred.exists():
        return preferred
    for p in (base / "src" / "etl").rglob("*.py"):
        return p
    for p in base.rglob("main.py"):
        return p
    return None

# --------- small literal utils ---------
def _lit_string(node: ast.AST) -> Optional[str]:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


# --------- boolean expression -> PySpark expr string ---------
def _is_col_call(n: ast.AST) -> bool:
    """
    Matches col("x") or F.col("x").
    """
    return (
        isinstance(n, ast.Call)
        and (
            (isinstance(n.func, ast.Name) and n.func.id == "col") or
            (isinstance(n.func, ast.Attribute) and n.func.attr == "col")
        )
        and len(n.args) == 1
        and isinstance(n.args[0], ast.Constant)
        and isinstance(n.args[0].value, str)
    )

def _col_from_subscript(n: ast.AST) -> Optional[str]:
    # Matches df['col'] where df is any Name and 'col' is a string literal
    if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name):
        # handle Python 3.8/3.9 slice differences
        s = n.slice if not hasattr(n.slice, "value") else n.slice.value
        if isinstance(s, ast.Constant) and isinstance(s.value, str):
            return s.value
    return None


def _emit_expr(n: ast.AST) -> str:
    # pandas column reference like df['Score']
    col_name = _col_from_subscript(n)
    if col_name:
        return f"F.col('{col_name}')"

    # Column reference: col("x") / F.col("x") (existing)
    if isinstance(n, ast.Call) and _is_col_call(n):
        return f"F.col('{n.args[0].value}')"

    # Simple constants
    if isinstance(n, ast.Constant):
        v = n.value
        if isinstance(v, str):
            return f"'{v}'"
        return str(v)

    # Comparisons
    if isinstance(n, ast.Compare) and len(n.ops) == 1 and len(n.comparators) == 1:
        left = _emit_expr(n.left)
        right = _emit_expr(n.comparators[0])
        op = n.ops[0]
        if isinstance(op, ast.Eq):   sym = "="
        elif isinstance(op, ast.NotEq): sym = "!="
        elif isinstance(op, ast.Lt): sym = "<"
        elif isinstance(op, ast.LtE): sym = "<="
        elif isinstance(op, ast.Gt): sym = ">"
        elif isinstance(op, ast.GtE): sym = ">="
        else: sym = "??"
        return f"({left} {sym} {right})"

    # pandas uses bitwise & / | for boolean chaining -> translate to AND/OR
    if isinstance(n, ast.BinOp):
        if isinstance(n.op, ast.BitAnd):
            return f"({_emit_expr(n.left)} AND {_emit_expr(n.right)})"
        if isinstance(n.op, ast.BitOr):
            return f"({_emit_expr(n.left)} OR {_emit_expr(n.right)})"

    # AND/OR chains (existing)
    if isinstance(n, ast.BoolOp):
        sep = " AND " if isinstance(n.op, ast.And) else " OR "
        return "(" + sep.join(_emit_expr(v) for v in n.values) + ")"

    # NOT (~expr)
    if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.Invert):
        return f"(NOT {_emit_expr(n.operand)})"

    return "<unsupported_expr>"


# --------- FILTER extraction (PySpark + pandas forms) ---------
def _extract_filter_ops(tree: ast.AST) -> List[Dict[str, Any]]:
    """
    Returns a list of {"op":"filter","expr":"<pyspark_expr>"} objects.
    Handles:
      - df.filter("a > 5") / df.where("...")
      - df.filter(F.col("a") > 5) (serialized)
      - pandas boolean indexing: df[ <bool_expr> ]
      - df.query("a > 5 and b == 'x'")
    """
    ops: List[Dict[str, Any]] = []

    for node in ast.walk(tree):
        # df.filter(...) / df.where(...)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in {"filter", "where"} and node.args:
                arg = node.args[0]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    ops.append({"op": "filter", "expr": arg.value})
                else:
                    # Try to serialize a boolean AST into a PySpark expr
                    expr = _emit_expr(arg)
                    if expr != "<unsupported_expr>":
                        ops.append({"op": "filter", "expr": expr})

        # pandas: df[ <bool_expr> ]
        if isinstance(node, ast.Subscript):
            # Slice may be in node.slice (Py3.9+) or node.slice.value (older ASTs)
            slice_node = node.slice if not hasattr(node.slice, "value") else node.slice.value
            if isinstance(slice_node, (ast.BoolOp, ast.BinOp, ast.Compare, ast.UnaryOp, ast.Call)):
                expr = _emit_expr(slice_node)
                if expr != "<unsupported_expr>":
                    ops.append({"op": "filter", "expr": expr})

        # pandas: df.query("a > 5 and b == 'x'")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "query" and node.args:
                q = node.args[0]
                if isinstance(q, ast.Constant) and isinstance(q.value, str):
                    # Normalize == to = for Spark SQL flavor (optional)
                    candidate = q.value.replace("==", "=")
                    ops.append({"op": "filter", "expr": candidate})

    return ops

def _is_np_where_call(n: ast.AST) -> bool:
    return (
        isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and isinstance(n.func.value, ast.Name)
        and n.func.value.id in {"np", "numpy"}  # tolerate either
        and n.func.attr == "where"
        and len(n.args) == 3
    )


def _lit_value_expr(v: ast.AST) -> Optional[str]:
    if isinstance(v, ast.Constant):
        x = v.value
        if isinstance(x, str):
            # escape single quotes for safe embedding
            return "F.lit('" + x.replace("'", "\\'") + "')"
        if isinstance(x, (int, float, bool)) or x is None:
            return f"F.lit({repr(x)})"
    return None

def _extract_assign_scalar_ops(tree: ast.AST) -> List[Dict[str, Any]]:
    """
    Capture pandas scalar assignments: df['col'] = <scalar>
    -> {"op":"withColumnExpr","name":"col","expr":"F.lit(<scalar>)"}
    """
    ops: List[Dict[str, Any]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            tgt = node.targets[0]
            col = _col_from_subscript(tgt)
            if col:
                val_expr = _lit_value_expr(node.value)
                if val_expr:
                    ops.append({"op": "withColumnExpr", "name": col, "expr": val_expr})
    return ops

def _extract_assign_ops(tree: ast.AST) -> List[Dict[str, Any]]:
    ops: List[Dict[str, Any]] = []
    for node in ast.walk(tree):
        # df['col'] = <scalar>
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            tgt = node.targets[0]
            col = _col_from_subscript(tgt)
            if col:
                # np.where(...) assignment
                if _is_np_where_call(node.value):
                    cond, tval, fval = node.value.args
                    cond_expr = _emit_expr(cond)
                    t_expr = _lit_value_expr(tval)
                    f_expr = _lit_value_expr(fval)
                    if cond_expr != "<unsupported_expr>" and t_expr and f_expr:
                        expr = f"F.when({cond_expr}, {t_expr}).otherwise({f_expr})"
                        ops.append({"op": "withColumnExpr", "name": col, "expr": expr})
                    continue
                # scalar literal
                val_expr = _lit_value_expr(node.value)
                if val_expr:
                    ops.append({"op": "withColumnExpr", "name": col, "expr": val_expr})

        # df.loc[mask, 'col'] = <scalar>
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
# --------- GROUP BY extraction (PySpark + pandas common forms) ---------
def _lit_list(args: List[ast.AST]) -> List[str]:
    vals = []
    for a in args:
        if isinstance(a, ast.Constant) and isinstance(a.value, str):
            vals.append(a.value)
    return vals

def _collect_group_cols(args: List[ast.AST]) -> List[str]:
    """
    Collect group columns from groupBy/groupby(...).
    Supports: groupBy("a","b") or groupBy(["a","b"]) or tuple("a","b").
    """
    if not args:
        return []
    a0 = args[0]
    # groupBy(["a","b"])
    if isinstance(a0, (ast.List, ast.Tuple)):
        return _lit_list(a0.elts)
    # groupBy("a","b")
    return _lit_list(args)

def _parse_alias_call(alias_call: ast.Call) -> Optional[Dict[str, str]]:
    """
    Parse F.func("col").alias("name") -> {"func":"func","col":"col","alias":"name"}
    """
    if not (isinstance(alias_call, ast.Call) and isinstance(alias_call.func, ast.Attribute) and alias_call.func.attr == "alias"):
        return None
    fcall = alias_call.func.value   # F.func("col")
    if not isinstance(fcall, ast.Call):
        return None
    # func name
    func_name = None
    if isinstance(fcall.func, ast.Attribute):
        func_name = fcall.func.attr       # F.sum -> "sum"
    elif isinstance(fcall.func, ast.Name):
        func_name = fcall.func.id
    # col arg
    col = None
    if fcall.args and isinstance(fcall.args[0], ast.Constant) and isinstance(fcall.args[0].value, str):
        col = fcall.args[0].value
    # alias arg
    alias = None
    if alias_call.args and isinstance(alias_call.args[0], ast.Constant) and isinstance(alias_call.args[0].value, str):
        alias = alias_call.args[0].value
    if func_name and col and alias:
        return {"func": func_name, "col": col, "alias": alias}
    return None

def _extract_groupby_reduce_ops(tree: ast.AST) -> List[Dict[str, Any]]:
    ops: List[Dict[str, Any]] = []
    reducers = {"mean", "sum", "count", "min", "max", "avg", "median"}
    for node in ast.walk(tree):
        # matches df.groupby(...).mean()  (or .sum(), .count(), ...)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in reducers:
            gb_call = node.func.value
            if isinstance(gb_call, ast.Call) and isinstance(gb_call.func, ast.Attribute) and gb_call.func.attr in {"groupby", "groupBy"}:
                group_cols = _collect_group_cols(gb_call.args)
                if group_cols:
                    ops.append({
                        "op": "groupByReduce",
                        "group_cols": group_cols,
                        "func": node.func.attr
                    })
    return ops

def _extract_groupby_ops(tree: ast.AST) -> List[Dict[str, Any]]:
    """
    Returns a list of:
      {"op":"groupByAgg","group_cols":[...],
       "aggs":[{"func":"sum","col":"x","alias":"sum_x"}, ...]}
    Handles:
      - PySpark: df.groupBy("a","b").agg(F.sum("x").alias("sx"), ...)
      - PySpark: df.groupBy("a").agg({"x":"sum","y":"avg"})
      - pandas:  df.groupby(["a","b"]).agg({"x":"sum","y":["mean","count"]})
    """
    ops: List[Dict[str, Any]] = []

    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "agg"):
            continue

        gb_call = node.func.value  # the groupBy/groupby call
        if not (isinstance(gb_call, ast.Call) and isinstance(gb_call.func, ast.Attribute) and gb_call.func.attr in {"groupBy", "groupby"}):
            continue

        group_cols = _collect_group_cols(gb_call.args)
        aggs: List[Dict[str, str]] = []

        # agg({"x":"sum","y":"avg"})  -- dict-form
        if node.args and isinstance(node.args[0], ast.Dict):
            for k, v in zip(node.args[0].keys, node.args[0].values):
                if isinstance(k, ast.Constant) and isinstance(k.value, str):
                    col = k.value
                    if isinstance(v, ast.Constant) and isinstance(v.value, str):
                        func = v.value
                        alias = f"{func}_{col}"
                        aggs.append({"func": func, "col": col, "alias": alias})

        # agg(F.sum("x").alias("sx"), F.count("*").alias("c")) -- call-list form with alias()
        for a in node.args:
            if isinstance(a, ast.Call):
                parsed = _parse_alias_call(a)
                if parsed:
                    aggs.append(parsed)

        # pandas: agg({"x":"sum","y":["mean","count"]})
        # Already captured dict simple, extend when values are lists:
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


# --------- MAIN IR extractor (now includes add_column + filter + groupByAgg) ---------
def extract_ir_operations_from_python(code: str) -> List[Dict[str, Any]]:
    """
    Deterministically extract a minimal, portable IR from the ETL code.
    Covers:
      - add_column(df, "Name", "Value")  -> {"op":"add_column", ...}
      - filter(...) / where(...) / df[<bool>] / df.query("...") -> {"op":"filter","expr":"..."}
      - groupBy(...).agg(...) (PySpark & pandas common forms) -> {"op":"groupByAgg", ...}

    You can extend this function for join / fillna / window / select / drop / rename etc.,
    or rely on the analyzer LLM + helpers to infer those semantics.
    """
    ops: List[Dict[str, Any]] = []
    try:
        tree = ast.parse(code)
    except Exception:
        return ops

    # 1) add_column(df, "Name", "Value")
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func_name = None
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                func_name = node.func.attr
            if func_name == "add_column" and len(node.args) >= 3:
                name = _lit_string(node.args[1])
                value = _lit_string(node.args[2])
                if name is not None:
                    ops.append({"op": "add_column", "name": name, "value": value})

    # 2) filter family
    ops.extend(_extract_filter_ops(tree))

    # 2b) pandas assignments (.loc and scalar) and np.where
    ops.extend(_extract_assign_ops(tree))

    # 3) groupBy/Agg family
    ops.extend(_extract_groupby_ops(tree))

    # 3) groupBy/Agg family
    ops.extend(_extract_groupby_reduce_ops(tree))   

    ops.extend(_extract_assign_scalar_ops(tree))    


    return ops

# ---------------- Helper discovery & source loading ----------------
def find_helper_modules_in_code(code: str) -> List[str]:
    """
    Discover helper modules imported by the ETL code.
    We detect:
      - import src.etl.helpers
      - from src.etl import helpers
      - from src.etl.helpers import <symbols>
    Returns a list of module paths like 'src.etl.helpers'.
    """
    modules: List[str] = []
    try:
        tree = ast.parse(code)
    except Exception:
        return modules

    for node in ast.walk(tree):
        # import src.etl.helpers
        if isinstance(node, ast.Import):
            for alias in node.names:
                if isinstance(alias, ast.alias) and isinstance(alias.name, str):
                    if alias.name.endswith(".helpers"):
                        modules.append(alias.name)

        # from src.etl import helpers  OR  from src.etl.helpers import add_column
        elif isinstance(node, ast.ImportFrom):
            if node.module and isinstance(node.module, str):
                # from src.etl.helpers import foo
                if node.module.endswith(".helpers"):
                    modules.append(node.module)
                # from src.etl import helpers
                for n in node.names or []:
                    if n.name == "helpers":
                        modules.append(f"{node.module}.helpers")

    # de-duplicate preserving order
    out: List[str] = []
    for m in modules:
        if m not in out:
            out.append(m)
    return out


def module_to_file(repo_path: str, module_path: str) -> Optional[Path]:
    """
    Map a module like 'src.etl.helpers' -> <repo_path>/src/etl/helpers.py
    """
    p = Path(repo_path) / Path(module_path.replace(".", "/") + ".py")
    return p if p.exists() else None


def read_helper_sources(
    repo_path: str,
    code: str,
    extra_candidates: Optional[List[str]] = None,
    max_per_file: int = 4000,
) -> List[Dict[str, str]]:
    """
    Collect helper sources referenced by imports in the ETL code,
    plus any extra known candidates (e.g., 'src.etl.helpers').

    Returns: [{"path":"<abs path>", "source":"<first N chars>"}]
    Keeping source short avoids exceeding model token limits.
    """
    # 1) discover from ETL imports
    modules = find_helper_modules_in_code(code)

    # 2) add explicit candidates if your repo always has these
    if extra_candidates:
        for m in extra_candidates:
            if m not in modules:
                modules.append(m)

    # 3) map to files
    files: List[Path] = []
    for m in modules:
        f = module_to_file(repo_path, m)
        if f:
            files.append(f)

    # 4) collect sources (capped)
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



# =======================================================
# Placeholder & notebook validation (fail fast)
# =======================================================
FORBIDDEN_TOKENS = {
    "...",
    "<INPUT_PATH>",
    "<OUTPUT_PATH>",
    "path_to_input_csv_file.csv",
    "path_to_output_csv_file.csv",
    "path_to_input_json_file.json",
    "path_to_output_json_file.json",
}


def validate_nb_source(nb_src: str, input_path: str, output_path: str, required_columns: Optional[List[str]] = None) -> None:
    """
    Hard validation for the generated notebook:
      - Must start with "# Databricks notebook source"
      - Must import: from pyspark.sql import functions as F
      - Must read via spark.read and write via df.write
      - Must contain exact INPUT/OUTPUT UC paths
      - Must NOT contain placeholders (..., path_to_*)
      - If IR had required add_column names, ensure they appear in code
    """
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


# =======================================================
# LLM setup
# =======================================================
AOAI = AzureChatOpenAI(
    deployment_name=CFG["AOAI_DEPLOYMENT"],
    model=CFG["AOAI_DEPLOYMENT"],
    api_key=CFG["AOAI_KEY"],
    azure_endpoint=CFG["AOAI_ENDPOINT"],
    api_version=CFG["AOAI_VERSION"],
    temperature=0.2,
    max_tokens=700,  # keeps headroom under 16k context models
)

# Agents (analyzer & converter) – no tools for analyzer; save_artifact tool for converter.
analyzer_prompt = """You are an ETL analyzer.

You will be given:
- ETL_SNIPPET: short Python code from the repo (pandas or PySpark, may call helper functions).
- HELPERS: the actual source code of helper functions (e.g., add_column, filter_rows, fill_missing, etc.).

Goal:
Return STRICT JSON ONLY (no prose) describing a neutral plan for a Spark/PySpark notebook:

{
  "entrypoints": [
    {
      "source_path": "<relative or null>",
      "inputs":  [ {"path": null, "format": "csv|json|parquet|delta|null"} ],
      "outputs": [ {"path": null, "format": "csv|json|parquet|delta|null", "mode": "overwrite|append|null"} ],
      "operations": [
        // Infer semantics from HELPERS + ETL_SNIPPET:
        // Examples of neutral ops you may emit:
        {"type": "withColumnExpr", "params": {"name": "ColName", "expr": "F.lit('Value')"}},
        {"type": "filter",        "params": {"expr": "(col('a') > 5) AND (col('b') = 'x')"}},
        {"type": "select",        "params": {"columns": ["a","b","c"]}},
        {"type": "drop",          "params": {"columns": ["c"]}},
        {"type": "rename",        "params": {"mapping": {"old":"new"}}},
        {"type": "fillna",        "params": {"value": "0", "subset": ["col1","col2"]}},
        {"type": "withColumnExpr","params": {"name": "Col2", "expr": "F.col('a') + F.col('b')"}},
        {"type": "groupByAgg",    "params": {"group_cols": ["a","b"], "aggs": [{"func":"sum","col":"x","alias":"sum_x"}]}},
        {"type": "orderBy",       "params": {"cols": ["a","b"]}},
        {"type": "window",        "params": {"partitionBy":["a"], "orderBy":["ts"], "frame":"rowsBetween(-1, 1)"}},
        {"type": "join",          "params": {"target_alias":"<omit if not available>", "on":["a"], "how":"left"}}
      ],
      "helpers": [
        {"module_path": "src/etl/helpers.py"} // or others actually used
      ]
    }
  ]
}

Rules:
- DO NOT hardcode any mapping. Instead, READ helper implementations in HELPERS and infer semantics.
  Example: if helper sets a constant column on a DataFrame, emit a withColumnExpr with F.lit(...).
- If helper semantics are ambiguous, inspect the callsites in ETL_SNIPPET to disambiguate.
- If you truly cannot infer a helper, you may emit {"type":"unknown_helper","params":{"name":"...", "reason":"..."}}.
- If path/format/mode are unknown, use null (not "unknown").
- Output must be valid JSON only (no comments, no prose).
"""


converter_prompt = """You are a senior Spark data engineer.
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
- INPUT_PATH and OUTPUT_PATH must appear as literal strings.
- If INPUT_PATH or OUTPUT_PATH are missing, raise ValueError("Missing input/output path").
- If SOURCE_SNIPPET contains pandas idioms not in IR, implement them:
  * df['col'] = v → df = df.withColumn('col', F.lit(v))
  * df.loc[mask, 'col'] = v → df = df.withColumn('col', F.when(mask, F.lit(v)).otherwise(F.col('col')))
  * df['col'] = np.where(c, a, b) → df = df.withColumn('col', F.when(c, F.lit(a)).otherwise(F.lit(b)))
- Boolean masks: use F.col(...) and PySpark operators; support & (AND) and | (OR).
- For groupby().mean()/sum()/count() with no explicit columns:
  compute over numeric columns only:
    from pyspark.sql.types import (IntegerType, LongType, ShortType, FloatType, DoubleType, DecimalType)
    num_cols = [f.name for f in df.schema.fields if isinstance(f.dataType, (IntegerType, LongType, ShortType, FloatType, DoubleType, DecimalType))]
    agg_exprs = [F.avg(c).alias(f"mean_{c}") for c in num_cols]
    df = df.groupBy(*keys).agg(*agg_exprs)
- If SOURCE_SNIPPET contains pandas-style scalar assignment:
  df['col'] = <scalar>  ->  df = df.withColumn('col', F.lit(<scalar>))
Return ONLY the notebook source; no explanations.
"""

analyzer = create_react_agent(AOAI, tools=[], prompt=analyzer_prompt)
converter = create_react_agent(AOAI, tools=[], prompt=converter_prompt)


# =======================================================
# State (LangGraph)
# =======================================================
class State(MessagesState):
    next: str
    repo_path: Optional[str]
    etl_file_path: Optional[str]
    plan: Optional[str]
    notebook_source: Optional[str]
    run_now: Optional[bool]


# =======================================================
# Nodes
# =======================================================
def bootstrap_node(state: State) -> Command[Literal["etl_analyzer"]]:
    """Select ETL entry file deterministically (no LLM)."""
    repo_path = state.get("repo_path") or "."
    etl_file = find_candidate_etl_file(repo_path)
    if not etl_file or not etl_file.exists():
        raise ValueError(
            "No ETL entry file found. Expected 'src/etl/main.py' or any '*.py' under 'src/etl/'."
        )
    return Command(
        update={
            "etl_file_path": str(etl_file),
            "messages": [HumanMessage(content=f"ETL entry selected: {etl_file}")],
        },
        goto="etl_analyzer",
    )


def etl_analyzer_node(state: State) -> Command[Literal["converter"]]:
    """Feed ETL snippet + helper sources to the analyzer LLM and store plan.json."""
    repo_path = state.get("repo_path") or "."
    etl_path = state.get("etl_file_path")
    if not etl_path:
        raise ValueError("Missing etl_file_path; bootstrap must run first.")

    etl_code = Path(etl_path).read_text(encoding="utf-8", errors="ignore")
    etl_snippet = etl_code[:1200]  # keep prompt small

    # Discover and read helper modules used by the ETL
    helpers_payload = read_helper_sources(
        repo_path=repo_path,
        code=etl_code,
        # optional: include a known default if your repos always have it
        extra_candidates=["src.etl.helpers"]
    )

    # Build a compact context payload for the LLM
    ctx = {
        "ETL_SNIPPET": etl_snippet,
        "HELPERS": helpers_payload,  # [{"path":"...","source":"..."}]
        "PREFERRED_NEUTRAL_OPS": [
            "withColumnExpr", "filter", "select", "drop", "rename",
            "fillna", "groupByAgg", "orderBy", "window", "join", "limit",
            "union", "unionByName", "distinct"
        ]
    }

    res = analyzer.invoke({
        **state,
        "messages": [
            HumanMessage(content="Analyze ETL_SNIPPET and HELPERS; emit STRICT JSON plan only."),
            HumanMessage(content=json.dumps(ctx, indent=2))
        ]
    })
    content = res["messages"][-1].content if hasattr(res["messages"][-1], "content") else str(res["messages"][-1])

    # Save/validate plan
    plan_dict = _safe_json_loads(content)
    save_artifact(repo_path, "plan.json", json.dumps(plan_dict, indent=2))

    return Command(update={
        "messages": [HumanMessage(content="Plan (semantic) extracted.", name="etl_analyzer")],
        "plan": json.dumps(plan_dict)
    }, goto="converter")

def _collect_assigned_columns(code: str) -> List[str]:
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
    # de-dup, preserve order
    out = []
    for c in cols:
        if c not in out:
            out.append(c)
    return out

def converter_node(state: State) -> Command[Literal["deployer"]]:
    """LLM writes entire notebook (no template), validated with UC paths + no placeholders."""
    repo_path = state.get("repo_path") or "."
    etl_path = state.get("etl_file_path")
    if not etl_path:
        raise ValueError("Missing etl_file_path; bootstrap must run first.")

    # UC paths
    uc = build_uc_paths()
    input_path = uc["input_path"]
    output_path = uc["output_path"]
    input_fmt = uc["input_format"]
    output_fmt = uc["output_format"]
    output_mode = uc["output_mode"]

    # Deterministic IR (add_column, extensible)
    code = Path(etl_path).read_text(encoding="utf-8", errors="ignore")
    ir_ops = extract_ir_operations_from_python(code)
    required_cols = [op["name"] for op in ir_ops if op.get("op") == "add_column" and op.get("name")]
    required_cols += _collect_assigned_columns(code)

    save_artifact(repo_path, "ir.json", json.dumps(ir_ops, indent=2))

    # Envelope for converter agent
    IR = {
        "input": {"path": input_path, "format": input_fmt},
        "output": {"path": output_path, "format": output_fmt, "mode": output_mode},
        "operations": ir_ops,  # may be empty; SOURCE_SNIPPET provides context
    }
    source_snippet = code[:800]  # keep tiny
    context_msg = {
        "IR": IR,
        "INPUT_PATH": input_path,
        "OUTPUT_PATH": output_path,
        "INPUT_FORMAT": input_fmt,
        "OUTPUT_FORMAT": output_fmt,
        "OUTPUT_MODE": output_mode,
        "SOURCE_SNIPPET": source_snippet,
    }

    # Generate + validate (one retry)
    tries, nb_src, last_err = 0, None, None
    while tries < 2:
        tries += 1
        res = converter.invoke({
            **state,
            "messages": [
                HumanMessage(content="Use this IR and SOURCE_SNIPPET to generate the notebook (follow HARD RULES)."),
                HumanMessage(content=json.dumps(context_msg, indent=2)),
            ]
        })
        nb_src = res["messages"][-1].content if hasattr(res["messages"][-1], "content") else str(res["messages"][-1])
        try:
            validate_nb_source(nb_src, input_path, output_path, required_columns=required_cols)
            break
        except Exception as e:
            last_err = str(e)
            # Feed correction hint
            state["messages"] = state.get("messages", []) + [
                HumanMessage(content=f"Previous notebook violated constraints: {last_err}. "
                                     f"Regenerate strictly following HARD RULES and IR.")
            ]
            nb_src = None

    if nb_src is None:
        raise ValueError(f"Notebook generation failed after retries. Last error: {last_err}")

    save_artifact(repo_path, "generated_notebook.py", nb_src)

    return Command(update={
        "messages": [HumanMessage(content="Notebook generated by LLM (no placeholders, UC paths).", name="converter")],
        "notebook_source": nb_src
    }, goto="deployer")


def deployer_node(state: State) -> Command[Literal["__end__"]]:
    """Import notebook, create job, optionally run."""
    repo_path = state.get("repo_path") or "."
    nb_src = state.get("notebook_source") or ""
    if not nb_src:
        raise ValueError("No notebook_source found; converter must run first.")

    # Unique tail and job name
    import datetime
    ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M")
    tail = f"{CFG['WORKSPACE_USER_SUBDIR'].strip('/')}/etl_migrated_{ts}"
    job_name = f"{CFG['JOB_NAME_PREFIX']}-etl_migrated"

    nb_path = import_notebook_source(tail, nb_src)
    job_id = create_job_for_notebook(job_name, nb_path, base_params=None)

    result = {"import_path": nb_path, "job_id": job_id}
    if state.get("run_now"):
        run = run_job_and_wait(job_id, notebook_params=None)
        result["run"] = run

    # Print final summary
    print("\n=== DEPLOY RESULT ===")
    print(json.dumps(result, indent=2))

    return Command(update={
        "messages": [HumanMessage(content=json.dumps(result), name="deployer")]
    }, goto=END)


# =======================================================
# Small helpers
# =======================================================
def _safe_json_loads(s: str) -> Dict[str, Any]:
    """Load JSON and try to salvage the last {...} object if needed."""
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"\{[\s\S]*\}$", s.strip(), re.MULTILINE)
        if m:
            return json.loads(m.group(0))
        raise


def trim_messages(state: State, keep_last: int = 4) -> State:
    """Keep only the last N messages to control token growth."""
    msgs = state.get("messages", [])
    if len(msgs) > keep_last:
        state["messages"] = msgs[-keep_last:]
    return state


# =======================================================
# Build graph
# =======================================================
def build_graph():
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


# =======================================================
# Main
# =======================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-now", action="store_true", help="Run the Databricks job after creation")
    args = parser.parse_args()

    # 1) Clone repo
    repo_path = clone_repo_to_temp()

    # 2) Execute graph
    g = build_graph()
    user_msg = "Please migrate the simplest ETL Python script from this cloned repo."
    result = g.invoke({
        "messages": [{"role": "user", "content": user_msg}],
        "repo_path": str(repo_path),
        "run_now": bool(args.run_now),
    })

    # 3) Final message
    print("\n=== FINAL MESSAGE ===")
    last = result["messages"][-1]
    print(last.content if hasattr(last, "content") else last)


if __name__ == "__main__":
    main()