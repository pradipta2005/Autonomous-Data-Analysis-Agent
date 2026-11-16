import ast
import io
import re
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = "json"
go.Figure.show = lambda self, *args, **kwargs: None
plt.show = lambda *args, **kwargs: None


def load_csv(file_or_path: Union[str, Any]) -> pd.DataFrame:
    """Read a CSV source with automatic encoding detection and column whitespace cleanup."""
    encodings = ["utf-8", "utf-8-sig", "latin1", "cp1252"]
    err = None

    for enc in encodings:
        try:
            if hasattr(file_or_path, "seek"):
                file_or_path.seek(0)
            df = pd.read_csv(file_or_path, encoding=enc)
            df.columns = [str(col).strip() for col in df.columns]
            return df
        except Exception as e:
            err = e
            continue

    raise ValueError(f"Unable to parse CSV with standard encodings: {err}")


def generate_data_profile(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate structural metadata, schema summary, and health statistics for a dataframe."""
    row_count, col_count = df.shape
    total_cells = row_count * col_count
    missing_cells = int(df.isna().sum().sum())
    completeness_pct = round(((total_cells - missing_cells) / total_cells) * 100, 2) if total_cells else 100.0

    mem_bytes = df.memory_usage(deep=True).sum()
    memory_str = f"{mem_bytes / 1024:.2f} KB" if mem_bytes < 1024 * 1024 else f"{mem_bytes / (1024 * 1024):.2f} MB"

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    date_candidates = [
        col for col in categorical_cols
        if any(keyword in col.lower() for keyword in ("date", "time", "day", "year", "month"))
    ]

    schema_records = [
        {
            "Column Name": col,
            "Data Type": str(df[col].dtype),
            "Missing Values": int(df[col].isna().sum()),
            "Missing %": f"{round((df[col].isna().sum() / row_count) * 100, 2)}%" if row_count else "0.0%",
            "Unique Values": int(df[col].nunique()),
        }
        for col in df.columns
    ]

    try:
        stats_df = df.describe(include="all").transpose().reset_index().rename(columns={"index": "Column Name"})
    except Exception:
        stats_df = pd.DataFrame()

    return {
        "rows": row_count,
        "columns": col_count,
        "total_cells": total_cells,
        "missing_cells": missing_cells,
        "completeness_pct": completeness_pct,
        "duplicate_count": int(df.duplicated().sum()),
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "date_candidates": date_candidates,
        "memory_str": memory_str,
        "columns_list": df.columns.tolist(),
        "schema_df": pd.DataFrame(schema_records),
        "stats_df": stats_df,
        "head_df": df.head(10),
    }


class ChartCollector:
    """Thread-safe store for visual figures generated during agent execution."""

    def __init__(self):
        self.figures: List[Dict[str, Any]] = []

    def add_plotly(self, fig: go.Figure):
        if fig is not None:
            self.figures.append({"type": "plotly", "fig": fig})

    def add_matplotlib(self, fig: plt.Figure):
        if fig is not None:
            self.figures.append({"type": "matplotlib", "fig": fig})

    def get_figures(self) -> List[Dict[str, Any]]:
        return self.figures

    def clear(self):
        self.figures.clear()


class CodeGuardrail:
    """Static analyzer that rejects execution of system-level commands, disk I/O, and dangerous builtins."""

    FORBIDDEN_MODULES = {
        "os", "sys", "subprocess", "shutil", "pathlib", "socket",
        "builtins", "importlib", "requests", "urllib", "pty",
        "commands", "winreg", "ctypes", "inspect", "multiprocessing",
        "threading", "signal", "platform", "tempfile"
    }

    FORBIDDEN_FUNCTIONS = {
        "open", "eval", "exec", "compile", "__import__",
        "globals", "locals", "getattr", "setattr", "delattr",
        "exit", "quit", "breakpoint"
    }

    FORBIDDEN_ATTRIBUTES = {
        "__subclasses__", "__globals__", "__code__", "__bases__",
        "__builtins__", "__class__", "__import__"
    }

    @classmethod
    def validate(cls, code: str) -> Tuple[bool, Optional[str]]:
        for dunder in cls.FORBIDDEN_ATTRIBUTES:
            if dunder in code:
                return False, f"Access to private attribute '{dunder}' is prohibited."

        try:
            tree = ast.parse(code)
        except SyntaxError:
            # Let execution surface syntax errors back to the agent for self-correction
            return True, None

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    if root in cls.FORBIDDEN_MODULES:
                        return False, f"Importing '{alias.name}' is blocked by security policy."
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    root = node.module.split(".")[0]
                    if root in cls.FORBIDDEN_MODULES:
                        return False, f"Importing from '{node.module}' is blocked by security policy."
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in cls.FORBIDDEN_FUNCTIONS:
                    return False, f"Calling '{node.func.id}()' is prohibited."
                elif isinstance(node.func, ast.Attribute) and node.func.attr in cls.FORBIDDEN_FUNCTIONS:
                    return False, f"Calling method '{node.func.attr}()' is prohibited."

        return True, None


class SafePandasREPL:
    """In-memory Python/Pandas execution environment with figure capture and persistent state."""

    def __init__(self, df: pd.DataFrame, chart_collector: Optional[ChartCollector] = None):
        self.df = df
        self.chart_collector = chart_collector or ChartCollector()
        self.reset_namespace()

    def reset_namespace(self):
        self.namespace: Dict[str, Any] = {
            "df": self.df,
            "pd": pd,
            "np": np,
            "px": px,
            "go": go,
            "plt": plt,
        }

    def execute(self, code: str) -> str:
        is_safe, violation = CodeGuardrail.validate(code)
        if not is_safe:
            return f"Security Policy Violation: {violation}\nOnly in-memory operations using pandas, numpy, and plotly are allowed."

        output_buffer = io.StringIO()
        orig_stdout = sys.stdout
        plt.close("all")

        captured_result = None
        has_chart = False

        try:
            sys.stdout = output_buffer
            parsed = ast.parse(code)

            if not parsed.body:
                return "Empty code block."

            # Evaluate the trailing statement as an expression if applicable
            if isinstance(parsed.body[-1], ast.Expr):
                if len(parsed.body) > 1:
                    preceding = ast.Module(body=parsed.body[:-1], type_ignores=[])
                    exec(compile(preceding, "<repl>", "exec"), self.namespace)

                last_expr = ast.Expression(body=parsed.body[-1].value)
                captured_result = eval(compile(last_expr, "<repl>", "eval"), self.namespace)
            else:
                exec(compile(parsed, "<repl>", "exec"), self.namespace)

        except Exception as exc:
            sys.stdout = orig_stdout
            err_msg = f"{type(exc).__name__}: {exc}"
            if isinstance(exc, KeyError):
                err_msg += f"\nHint: Verify column names in df. Available columns: {self.df.columns.tolist()}"
            return f"Execution Error: {err_msg}"
        finally:
            sys.stdout = orig_stdout

        # Intercept assigned Plotly figure
        fig_obj = self.namespace.get("fig")
        if fig_obj is not None and (isinstance(fig_obj, go.Figure) or hasattr(fig_obj, "data")):
            try:
                fig_obj.update_layout(
                    template="plotly_white",
                    margin=dict(l=40, r=40, t=50, b=40),
                    font=dict(family="Inter, sans-serif"),
                )
            except Exception:
                pass
            self.chart_collector.add_plotly(fig_obj)
            has_chart = True

        # Capture expression return if it's a figure
        if isinstance(captured_result, go.Figure) and not has_chart:
            try:
                captured_result.update_layout(
                    template="plotly_white",
                    margin=dict(l=40, r=40, t=50, b=40),
                    font=dict(family="Inter, sans-serif"),
                )
            except Exception:
                pass
            self.chart_collector.add_plotly(captured_result)
            has_chart = True

        # Intercept any matplotlib plot
        if plt.get_fignums():
            for num in plt.get_fignums():
                self.chart_collector.add_matplotlib(plt.figure(num))
                has_chart = True
            plt.close("all")

        # Format stdout and expression result
        stdout_text = output_buffer.getvalue().strip()
        parts = []

        if stdout_text:
            parts.append(stdout_text)

        if captured_result is not None:
            if isinstance(captured_result, pd.DataFrame):
                parts.append(captured_result.to_string(max_rows=15, max_cols=8))
            elif isinstance(captured_result, pd.Series):
                parts.append(captured_result.to_string(max_rows=15))
            elif isinstance(captured_result, go.Figure):
                title = captured_result.layout.title.text if captured_result.layout.title else "Chart"
                parts.append(f"[Plotly Figure: '{title}' created]")
            else:
                parts.append(str(captured_result))

        output = "\n".join(parts) if parts else "Executed without output."
        if has_chart:
            output += "\n[Visual chart rendered to interface]"

        return output
