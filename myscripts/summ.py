"""按 run 汇总打印项目 MLflow 实验记录。"""

from __future__ import annotations

import argparse
import ast
import base64
import math
import os
import pickle
import re
import sqlite3
import sys
from collections import defaultdict
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence
from urllib.parse import unquote, urlparse


# ═══ 默认路径与输出字段 ═══════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DEFAULT_DB_PATH = PROJECT_ROOT / "mlruns" / "mlflow.db"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "myscripts" / "exp_reports" / "exp_summaries.md"

MODEL_PARAM_KEY = "qlib.model.class"
INSTRUMENT_PARAM_KEY = "qlib.dataset.instruments"
TEST_START_PARAM_KEY = "qlib.dataset.test_start"
TEST_END_PARAM_KEY = "qlib.dataset.test_end"

LEGACY_MODEL_PARAM_KEY = "model.class"
LEGACY_INSTRUMENT_PARAM_KEY = "dataset.kwargs.handler.kwargs.instruments"
LEGACY_TEST_PERIOD_PARAM_KEY = "dataset.kwargs.segments.test"

OUTPUT_COLUMNS = (
    "experiment_id",
    "experiment_name",
    "run_id",
    "model",
    "instrument",
    "test_period",
    "start_time",
    "end_time",
    "rank_ic",
    "rank_icir",
    "annualized_return",
    "information_ratio",
    "max_drawdown",
)


def positive_int(value: str) -> int:
    """将命令行文本解析为正整数。"""
    parsed_value = int(value)
    if parsed_value <= 0:
        raise argparse.ArgumentTypeError("N 必须是大于 0 的整数。")
    return parsed_value


def parse_args() -> argparse.Namespace:
    """解析命令行参数并返回配置。"""
    parser = argparse.ArgumentParser(description="逐行打印 MLflow run 摘要，并生成 Markdown 汇总。")
    parser.add_argument(
        "n",
        nargs="?",
        type=positive_int,
        default=None,
        help="选取按开始时间倒序排列的最新 N 条 run，再按 experiment_id、开始时间正序打印；不指定时打印全部。",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB_PATH,
        help=f"MLflow SQLite 数据库路径，默认：{DEFAULT_DB_PATH}",
    )
    parser.add_argument(
        "--frequency",
        default="1day",
        help="PortAnaRecord 指标频率前缀，默认：1day",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Markdown 输出路径，默认：{DEFAULT_OUTPUT_PATH}",
    )
    parser.add_argument(
        "--include-deleted",
        action="store_true",
        help="同时显示 lifecycle_stage 为 deleted 的实验和 run。",
    )
    parser.add_argument(
        "--include-child-runs",
        action="store_true",
        help="同时显示 Rolling 训练产生的中间子 run；默认只显示最终拼接结果。",
    )
    detail_group = parser.add_mutually_exclusive_group()
    detail_group.add_argument(
        "--detail",
        nargs="?",
        const="latest",
        default=None,
        metavar="RUN_ID",
        help="生成单个 run 的详细 Markdown 报告；不提供 RUN_ID 时选择最近一个具有完整回测产物的 run。",
    )
    detail_group.add_argument(
        "--detail-shap",
        nargs="?",
        const="latest",
        default=None,
        metavar="RUN_ID",
        help="生成详细 Markdown 报告并额外计算 SHAP；不提供 RUN_ID 时选择最近一个完整 run。",
    )
    parser.add_argument(
        "--detail-output",
        type=Path,
        default=None,
        help="详细报告 Markdown 路径；默认写入 myscripts/exp_reports/<run_id>/report.md。",
    )
    parser.add_argument(
        "--provider-uri",
        type=Path,
        default=Path("~/.qlib/qlib_data/cn_data"),
        help="计算 SHAP 时使用的 Qlib 数据目录。",
    )
    parser.add_argument(
        "--groups",
        type=positive_int,
        default=5,
        help="分组累计收益的组数，默认 5。",
    )
    parser.add_argument(
        "--shap-samples",
        type=positive_int,
        default=2000,
        help="计算 SHAP 时跨模型抽取的目标样本数，默认 2000。",
    )
    return parser.parse_args()


def timestamp_to_text(timestamp_ms: Optional[int]) -> str:
    """将 MLflow 毫秒时间戳转换为本地时间文本。"""
    if timestamp_ms is None:
        return "-"
    return datetime.fromtimestamp(timestamp_ms / 1000).astimezone().strftime("%Y-%m-%d %H:%M:%S")


def metric_to_text(
    value: Optional[float],
    is_nan: Optional[int],
    as_percentage: bool = False,
    decimal_places: int = 6,
) -> str:
    """格式化 MLflow 指标，缺失值和 NaN 显示为短横线。

    Parameters
    ----------
    value : float, optional
        MLflow 保存的指标原始值。
    is_nan : int, optional
        MLflow 的 NaN 标记。
    as_percentage : bool
        是否将原始小数乘以 100 并添加百分号。
    decimal_places : int
        结果保留的小数位数。

    Returns
    -------
    str
        格式化后的指标文本。
    """
    if value is None or is_nan or math.isnan(value):
        return "-"
    if as_percentage:
        return f"{value * 100:.{decimal_places}f}%"
    return f"{value:.{decimal_places}f}"


def normalize_date_text(value: Any) -> str:
    """将配置日期统一格式化为 YYYY-MM-DD；无法识别时保留原文本。"""
    value_text = str(value)
    date_match = re.search(r"\d{4}-\d{2}-\d{2}|\d{8}", value_text)
    if date_match is None:
        return value_text
    date_text = date_match.group(0)
    if len(date_text) == 8:
        return f"{date_text[:4]}-{date_text[4:6]}-{date_text[6:]}"
    return date_text


def format_test_period(
    configured_period: Optional[str],
    configured_start: Optional[str] = None,
    configured_end: Optional[str] = None,
) -> str:
    """优先格式化统一字段，再兼容 Trainer 保存的旧 test segment。"""
    if configured_start and configured_end:
        return f"{normalize_date_text(configured_start)}~{normalize_date_text(configured_end)}"
    if not configured_period:
        return "-"

    try:
        parsed_period = ast.literal_eval(configured_period)
    except (SyntaxError, ValueError):
        parsed_period = None
    if isinstance(parsed_period, (list, tuple)) and len(parsed_period) == 2:
        return f"{normalize_date_text(parsed_period[0])}~{normalize_date_text(parsed_period[1])}"

    date_tokens = re.findall(r"\d{4}-\d{2}-\d{2}|\d{8}", configured_period)
    if len(date_tokens) >= 2:
        normalized_dates = [
            f"{date[:4]}-{date[4:6]}-{date[6:]}" if len(date) == 8 else date
            for date in date_tokens[:2]
        ]
        return "~".join(normalized_dates)
    return configured_period


def load_run_rows(
    database_path: Path,
    frequency: str,
    include_deleted: bool,
    limit: Optional[int] = None,
    include_child_runs: bool = False,
) -> List[Dict[str, str]]:
    """从 MLflow SQLite 数据库只读加载每个 run 的摘要。

    Parameters
    ----------
    database_path : Path
        MLflow SQLite 数据库文件。
    frequency : str
        PortAnaRecord 写入指标时使用的频率前缀，例如 ``1day``。
    include_deleted : bool
        是否包含已删除的实验和 run。
    limit : int, optional
        先选取按开始时间倒序排列的前 N 条 run；最终按 experiment_id、开始时间正序返回。
    include_child_runs : bool
        是否包含 Rolling 训练产生的中间子 run；默认只保留最终拼接结果。

    Returns
    -------
    list[dict[str, str]]
        已格式化的非 FAILED run 摘要，每个元素对应一次 run。
    """
    resolved_database_path = database_path.expanduser().resolve()
    if not resolved_database_path.is_file():
        raise FileNotFoundError(f"MLflow database not found: {resolved_database_path}")

    metric_prefix = f"{frequency}.excess_return_with_cost"
    metric_keys = {
        "annualized_return": f"{metric_prefix}.annualized_return",
        "information_ratio": f"{metric_prefix}.information_ratio",
        "max_drawdown": f"{metric_prefix}.max_drawdown",
    }
    # FAILED run 没有可比较的完整结果，默认始终从实验摘要中排除。
    query_filters = ["COALESCE(r.status, '') <> 'FAILED'"]
    if not include_deleted:
        query_filters.extend(["e.lifecycle_stage = 'active'", "r.lifecycle_stage = 'active'"])
    if not include_child_runs:
        # 默认前缀覆盖尚未生成最终 run 的 Rolling；exp_name 引用兼容自定义 rolling_exp。
        query_filters.append(
            """
            e.name NOT LIKE 'rolling_models_%'
            AND NOT EXISTS (
                SELECT 1
                FROM params AS rolling_parent_param
                WHERE rolling_parent_param.key = 'exp_name'
                  AND rolling_parent_param.value = e.name
                  AND rolling_parent_param.run_uuid <> r.run_uuid
            )
            """
        )
    where_clause = "WHERE " + " AND ".join(query_filters)

    limit_clause = " LIMIT ?" if limit is not None else ""
    query = f"""
        SELECT
            e.experiment_id,
            e.name AS experiment_name,
            r.run_uuid AS run_id,
            COALESCE(r.status, '-') AS status,
            r.start_time,
            r.end_time,
            CASE
                WHEN rolling_exp_param.value IS NOT NULL
                 AND COALESCE(model_param.value, legacy_model_param.value) NOT LIKE 'Roll %'
                THEN 'Roll ' || COALESCE(model_param.value, legacy_model_param.value)
                ELSE COALESCE(model_param.value, legacy_model_param.value)
            END AS model,
            COALESCE(instrument_param.value, legacy_instrument_param.value) AS instrument,
            test_start_param.value AS test_start,
            test_end_param.value AS test_end,
            legacy_test_period_param.value AS legacy_test_period,
            annualized.value AS annualized_return,
            annualized.is_nan AS annualized_return_is_nan,
            information.value AS information_ratio,
            information.is_nan AS information_ratio_is_nan,
            drawdown.value AS max_drawdown,
            drawdown.is_nan AS max_drawdown_is_nan,
            rank_ic.value AS rank_ic,
            rank_ic.is_nan AS rank_ic_is_nan,
            rank_icir.value AS rank_icir,
            rank_icir.is_nan AS rank_icir_is_nan
        FROM runs AS r
        JOIN experiments AS e
          ON e.experiment_id = r.experiment_id
        LEFT JOIN latest_metrics AS annualized
          ON annualized.run_uuid = r.run_uuid AND annualized.key = ?
        LEFT JOIN latest_metrics AS information
          ON information.run_uuid = r.run_uuid AND information.key = ?
        LEFT JOIN latest_metrics AS drawdown
          ON drawdown.run_uuid = r.run_uuid AND drawdown.key = ?
        LEFT JOIN latest_metrics AS rank_ic
          ON rank_ic.run_uuid = r.run_uuid AND rank_ic.key = ?
        LEFT JOIN latest_metrics AS rank_icir
          ON rank_icir.run_uuid = r.run_uuid AND rank_icir.key = ?
        LEFT JOIN params AS instrument_param
          ON instrument_param.run_uuid = r.run_uuid AND instrument_param.key = ?
        LEFT JOIN params AS test_start_param
          ON test_start_param.run_uuid = r.run_uuid AND test_start_param.key = ?
        LEFT JOIN params AS test_end_param
          ON test_end_param.run_uuid = r.run_uuid AND test_end_param.key = ?
        LEFT JOIN params AS model_param
          ON model_param.run_uuid = r.run_uuid AND model_param.key = ?
        LEFT JOIN params AS legacy_model_param
          ON legacy_model_param.run_uuid = r.run_uuid AND legacy_model_param.key = ?
        LEFT JOIN params AS rolling_exp_param
          ON rolling_exp_param.run_uuid = r.run_uuid AND rolling_exp_param.key = 'exp_name'
        LEFT JOIN params AS legacy_instrument_param
          ON legacy_instrument_param.run_uuid = r.run_uuid AND legacy_instrument_param.key = ?
        LEFT JOIN params AS legacy_test_period_param
          ON legacy_test_period_param.run_uuid = r.run_uuid AND legacy_test_period_param.key = ?
        {where_clause}
        ORDER BY r.start_time DESC, r.run_uuid
        {limit_clause}
    """

    query_parameters: List[Any] = [
        metric_keys["annualized_return"],
        metric_keys["information_ratio"],
        metric_keys["max_drawdown"],
        "Rank IC",
        "Rank ICIR",
        INSTRUMENT_PARAM_KEY,
        TEST_START_PARAM_KEY,
        TEST_END_PARAM_KEY,
        MODEL_PARAM_KEY,
        LEGACY_MODEL_PARAM_KEY,
        LEGACY_INSTRUMENT_PARAM_KEY,
        LEGACY_TEST_PERIOD_PARAM_KEY,
    ]
    if limit is not None:
        query_parameters.append(limit)

    database_uri = f"file:{resolved_database_path.as_posix()}?mode=ro"
    with closing(sqlite3.connect(database_uri, uri=True)) as connection:
        records = connection.execute(query, query_parameters).fetchall()

    # LIMIT 仍按开始时间选取“最新 N 条”；选取完成后再按实验和时间正序展示。
    records.sort(
        key=lambda record: (
            int(record[0]),
            record[4] if record[4] is not None else -1,
            record[2],
        )
    )

    rows = []
    for record in records:
        rows.append(
            {
                "experiment_id": str(record[0]),
                "experiment_name": record[1],
                "run_id": record[2],
                "model": record[6] or "-",
                "instrument": record[7] or "-",
                "test_period": format_test_period(record[10], record[8], record[9]),
                "status": record[3],
                "start_time": timestamp_to_text(record[4]),
                "end_time": timestamp_to_text(record[5]),
                "annualized_return": metric_to_text(
                    record[11], record[12], as_percentage=True, decimal_places=2
                ),
                "information_ratio": metric_to_text(record[13], record[14], decimal_places=3),
                "max_drawdown": metric_to_text(
                    record[15], record[16], as_percentage=True, decimal_places=2
                ),
                "rank_ic": metric_to_text(record[17], record[18], decimal_places=3),
                "rank_icir": metric_to_text(record[19], record[20], decimal_places=3),
            }
        )
    return rows


def print_table(rows: Sequence[Dict[str, Any]], columns: Sequence[str] = OUTPUT_COLUMNS) -> None:
    """将字典行打印为便于终端阅读的等宽表格。"""
    if not rows:
        print("No MLflow runs found.")
        return

    widths = {
        column: max(len(column), *(len(str(row.get(column, ""))) for row in rows))
        for column in columns
    }
    print("  ".join(column.ljust(widths[column]) for column in columns))
    print("  ".join("-" * widths[column] for column in columns))
    for row in rows:
        print("  ".join(str(row.get(column, "")).ljust(widths[column]) for column in columns))


def markdown_cell(value: Any) -> str:
    """转义 Markdown 表格单元格中的特殊字符。"""
    return str(value).replace("|", r"\|").replace("\r\n", "<br>").replace("\n", "<br>")


def write_markdown(
    rows: Sequence[Dict[str, Any]],
    output_path: Path,
    database_path: Path,
    frequency: str,
    columns: Sequence[str] = OUTPUT_COLUMNS,
) -> Path:
    """将 run 摘要写入 Markdown 文件并返回绝对路径。"""
    resolved_output_path = output_path.expanduser().resolve()
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)

    generated_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z")
    lines = [
        "# Qlib Experiment Summaries",
        "",
        f"- Generated at: `{generated_at}`",
        f"- MLflow database: `{database_path.expanduser().resolve()}`",
        f"- Metric frequency: `{frequency}`",
        "",
    ]
    if not rows:
        lines.append("No MLflow runs found.")
    else:
        lines.append("| " + " | ".join(columns) + " |")
        lines.append("| " + " | ".join("---" for _ in columns) + " |")
        for row in rows:
            lines.append(
                "| "
                + " | ".join(markdown_cell(row.get(column, "")) for column in columns)
                + " |"
            )

    resolved_output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return resolved_output_path


REQUIRED_REPORT_ARTIFACTS = (
    "pred.pkl",
    "label.pkl",
    "portfolio_analysis/report_normal_{frequency}.pkl",
)


@dataclass(frozen=True)
class RunInfo:
    experiment_id: str
    experiment_name: str
    run_id: str
    status: str
    start_time: Optional[int]
    end_time: Optional[int]
    artifact_dir: Path


def _artifact_uri_to_path(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme != "file":
        return Path(uri).expanduser().resolve()
    raw_path = unquote(parsed.path)
    if os.name == "nt" and re.match(r"^/[A-Za-z]:", raw_path):
        raw_path = raw_path[1:]
    return Path(raw_path).resolve()


def _fetch_runs(database_path: Path, run_id: Optional[str] = None) -> list[RunInfo]:
    filters = ["COALESCE(r.status, '') <> 'FAILED'", "r.lifecycle_stage = 'active'", "e.lifecycle_stage = 'active'"]
    params: list[Any] = []
    if run_id:
        filters.append("r.run_uuid LIKE ?")
        params.append(f"{run_id}%")
    query = f"""
        SELECT e.experiment_id, e.name, r.run_uuid, COALESCE(r.status, '-'),
               r.start_time, r.end_time, r.artifact_uri
        FROM runs AS r
        JOIN experiments AS e ON e.experiment_id = r.experiment_id
        WHERE {' AND '.join(filters)}
        ORDER BY r.start_time DESC, r.run_uuid
    """
    database_uri = f"file:{database_path.expanduser().resolve().as_posix()}?mode=ro"
    with sqlite3.connect(database_uri, uri=True) as connection:
        records = connection.execute(query, params).fetchall()
    return [
        RunInfo(
            experiment_id=str(record[0]),
            experiment_name=record[1],
            run_id=record[2],
            status=record[3],
            start_time=record[4],
            end_time=record[5],
            artifact_dir=_artifact_uri_to_path(record[6]),
        )
        for record in records
    ]


def _is_reportable(run: RunInfo, frequency: str) -> bool:
    return all(
        (run.artifact_dir / relative_path.format(frequency=frequency)).is_file()
        for relative_path in REQUIRED_REPORT_ARTIFACTS
    )


def select_run(database_path: Path, frequency: str, run_id: Optional[str]) -> tuple[RunInfo, int]:
    runs = _fetch_runs(database_path, run_id)
    if run_id:
        if not runs:
            raise ValueError(f"Run not found: {run_id}")
        if len(runs) > 1:
            raise ValueError(f"Run ID prefix is ambiguous: {run_id}")
        if not _is_reportable(runs[0], frequency):
            missing = [
                relative_path.format(frequency=frequency)
                for relative_path in REQUIRED_REPORT_ARTIFACTS
                if not (runs[0].artifact_dir / relative_path.format(frequency=frequency)).is_file()
            ]
            raise ValueError(f"Run {runs[0].run_id} is missing required artifacts: {', '.join(missing)}")
        return runs[0], 0

    for skipped, run in enumerate(runs):
        if _is_reportable(run, frequency):
            return run, skipped
    raise ValueError(f"No run contains pred.pkl, label.pkl and report_normal_{frequency}.pkl")


def _load_params(database_path: Path, run_id: str) -> dict[str, str]:
    query = "SELECT key, value FROM params WHERE run_uuid = ? ORDER BY key"
    database_uri = f"file:{database_path.expanduser().resolve().as_posix()}?mode=ro"
    with sqlite3.connect(database_uri, uri=True) as connection:
        return dict(connection.execute(query, [run_id]).fetchall())


def _load_latest_metrics(database_path: Path, run_id: str) -> dict[str, float]:
    query = "SELECT key, value FROM latest_metrics WHERE run_uuid = ? ORDER BY key"
    database_uri = f"file:{database_path.expanduser().resolve().as_posix()}?mode=ro"
    with sqlite3.connect(database_uri, uri=True) as connection:
        return dict(connection.execute(query, [run_id]).fetchall())


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as file:
        return pickle.load(file)


def _as_single_column_frame(value: Any, column_name: str):
    import pandas as pd

    if isinstance(value, pd.Series):
        return value.rename(column_name).to_frame()
    if not isinstance(value, pd.DataFrame):
        raise TypeError(f"Expected pandas Series or DataFrame for {column_name}, got {type(value).__name__}")
    if column_name in value.columns:
        return value[[column_name]]
    if value.shape[1] != 1:
        raise ValueError(f"Cannot identify {column_name!r} in columns: {list(value.columns)}")
    result = value.copy()
    result.columns = [column_name]
    return result


def load_pred_label(run: RunInfo):
    import pandas as pd

    pred = _as_single_column_frame(_load_pickle(run.artifact_dir / "pred.pkl"), "score")
    label = _as_single_column_frame(_load_pickle(run.artifact_dir / "label.pkl"), "label")
    result = pd.concat([label, pred], axis=1, sort=True).reindex(label.index)
    if not isinstance(result.index, pd.MultiIndex):
        raise ValueError("pred.pkl and label.pkl must use a MultiIndex")
    if "datetime" not in result.index.names:
        raise ValueError(f"Prediction index has no 'datetime' level: {result.index.names}")
    return result


def load_backtest_report(run: RunInfo, frequency: str):
    import pandas as pd

    path = run.artifact_dir / "portfolio_analysis" / f"report_normal_{frequency}.pkl"
    report = _load_pickle(path)
    if not isinstance(report, pd.DataFrame):
        raise TypeError(f"Backtest report is not a DataFrame: {type(report).__name__}")
    missing = {"return", "bench", "cost"}.difference(report.columns)
    if missing:
        raise ValueError(f"Backtest report is missing columns: {sorted(missing)}")
    result = report.copy()
    result.index = pd.to_datetime(result.index)
    return result.sort_index()


def calculate_group_returns(pred_label, groups: int = 5):
    import pandas as pd

    ordered = pred_label.sort_values("score", ascending=False)
    valid = ordered.dropna(subset=["score"])
    grouped = pd.DataFrame(
        {
            f"Group{index + 1}": valid.groupby(level="datetime", group_keys=False)["label"].apply(
                lambda values, index=index: values[
                    len(values) // groups * index : len(values) // groups * (index + 1)
                ].mean()
            )
            for index in range(groups)
        }
    )
    grouped.index = pd.to_datetime(grouped.index)
    grouped["long-short"] = grouped["Group1"] - grouped[f"Group{groups}"]
    grouped["long-average"] = grouped["Group1"] - ordered.groupby(
        level="datetime", group_keys=False
    )["label"].mean()
    return grouped.dropna(how="all").sort_index()


def _annualization_factor(frequency: str) -> float:
    from qlib.utils.resam import Freq

    count, normalized_frequency = Freq.parse(frequency)
    factors = {
        Freq.NORM_FREQ_MINUTE: 240 * 238,
        Freq.NORM_FREQ_DAY: 238,
        Freq.NORM_FREQ_WEEK: 50,
        Freq.NORM_FREQ_MONTH: 12,
    }
    return factors[normalized_frequency] / count


def calculate_annual_metrics(report, frequency: str):
    import numpy as np
    import pandas as pd
    from qlib.contrib.evaluate import risk_analysis

    excess = (report["return"] - report["bench"] - report["cost"]).dropna()
    annualization_factor = _annualization_factor(frequency)
    rows = []
    for year, values in excess.groupby(excess.index.year):
        risk = risk_analysis(values, freq=frequency)["risk"]
        rows.append(
            {
                "year": int(year),
                "observations": int(values.count()),
                "excess_return": float(risk["annualized_return"]),
                "excess_volatility": float(risk["std"] * np.sqrt(annualization_factor)),
                "excess_sharpe": float(risk["information_ratio"]),
                "excess_max_drawdown": float(risk["max_drawdown"]),
            }
        )
    return pd.DataFrame(rows).set_index("year")


def _configure_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    return plt


def _figure_data_uri(figure, dpi: int) -> str:
    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def plot_cumulative_excess(report) -> str:
    from matplotlib.ticker import PercentFormatter

    plt = _configure_matplotlib()
    excess_without_cost = (report["return"] - report["bench"]).cumsum()
    excess_with_cost = (report["return"] - report["bench"] - report["cost"]).cumsum()
    figure, axis = plt.subplots(figsize=(12, 5.8))
    axis.plot(excess_without_cost.index, excess_without_cost, label="Excess return without cost", linewidth=1.5)
    axis.plot(excess_with_cost.index, excess_with_cost, label="Excess return with cost", linewidth=1.8)
    axis.axhline(0, color="black", linewidth=0.7)
    axis.set_title("Cumulative Excess Return")
    axis.set_xlabel("Date")
    axis.set_ylabel("Cumulative return")
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.legend()
    figure.tight_layout()
    image_uri = _figure_data_uri(figure, dpi=160)
    plt.close(figure)
    return image_uri


def plot_group_returns(group_returns) -> str:
    from matplotlib.ticker import PercentFormatter

    plt = _configure_matplotlib()
    cumulative = group_returns.cumsum()
    figure, axis = plt.subplots(figsize=(12, 6.5))
    for column in cumulative.columns:
        is_spread = column in {"long-short", "long-average"}
        axis.plot(
            cumulative.index,
            cumulative[column],
            label=column,
            linewidth=2.0 if is_spread else 1.2,
            linestyle="--" if is_spread else "-",
        )
    axis.axhline(0, color="black", linewidth=0.7)
    axis.set_title("Cumulative Return by Score Group")
    axis.set_xlabel("Date")
    axis.set_ylabel("Cumulative return")
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.legend(ncol=2)
    figure.tight_layout()
    image_uri = _figure_data_uri(figure, dpi=160)
    plt.close(figure)
    return image_uri


def _child_model_runs(database_path: Path, parent_params: dict[str, str]) -> list[RunInfo]:
    rolling_experiment_name = parent_params.get("exp_name")
    if not rolling_experiment_name:
        return []
    query = """
        SELECT e.experiment_id, e.name, r.run_uuid, COALESCE(r.status, '-'),
               r.start_time, r.end_time, r.artifact_uri
        FROM runs AS r
        JOIN experiments AS e ON e.experiment_id = r.experiment_id
        WHERE e.name = ?
          AND COALESCE(r.status, '') <> 'FAILED'
          AND r.lifecycle_stage = 'active'
          AND e.lifecycle_stage = 'active'
        ORDER BY r.start_time, r.run_uuid
    """
    database_uri = f"file:{database_path.expanduser().resolve().as_posix()}?mode=ro"
    with sqlite3.connect(database_uri, uri=True) as connection:
        records = connection.execute(query, [rolling_experiment_name]).fetchall()
    return [
        RunInfo(
            experiment_id=str(record[0]),
            experiment_name=record[1],
            run_id=record[2],
            status=record[3],
            start_time=record[4],
            end_time=record[5],
            artifact_dir=_artifact_uri_to_path(record[6]),
        )
        for record in records
    ]


def _model_dataset_candidates(
    database_path: Path, run: RunInfo, params: dict[str, str]
) -> list[tuple[Path, Path, str]]:
    candidates = []
    model_paths = [run.artifact_dir / "params.pkl", run.artifact_dir / "trained_model"]
    dataset_path = run.artifact_dir / "dataset"
    for model_path in model_paths:
        if model_path.is_file() and dataset_path.is_file():
            candidates.append((model_path, dataset_path, run.run_id))
            break
    if candidates:
        return candidates

    for child in _child_model_runs(database_path, params):
        child_dataset = child.artifact_dir / "dataset"
        child_model = child.artifact_dir / "params.pkl"
        if child_model.is_file() and child_dataset.is_file():
            candidates.append((child_model, child_dataset, child.run_id))
    return candidates


def _feature_names(columns: Iterable[Any]) -> list[str]:
    names = []
    for column in columns:
        if isinstance(column, tuple):
            names.append("/".join(str(part) for part in column if str(part)))
        else:
            names.append(str(column))
    return names


def _tree_shap_values(model: Any, features):
    import numpy as np

    booster = getattr(model, "model", None)
    if booster is None:
        raise TypeError(f"{type(model).__name__} does not expose a fitted model")
    module_name = type(booster).__module__.lower()
    values = features.to_numpy()
    if "lightgbm" in module_name:
        contributions = booster.predict(values, pred_contrib=True)
    elif "xgboost" in module_name:
        import xgboost as xgb

        contributions = booster.predict(xgb.DMatrix(values), pred_contribs=True)
    elif "catboost" in module_name:
        from catboost import Pool

        contributions = booster.get_feature_importance(Pool(values), type="ShapValues")
    else:
        raise TypeError(f"SHAP is not implemented for {type(model).__name__}")
    contributions = np.asarray(contributions)
    if contributions.ndim != 2 or contributions.shape[1] != features.shape[1] + 1:
        raise ValueError(
            f"Unexpected SHAP shape {contributions.shape}; expected (*, {features.shape[1] + 1})"
        )
    return contributions[:, :-1]


def calculate_shap_importance(
    database_path: Path,
    run: RunInfo,
    params: dict[str, str],
    provider_uri: Path,
    max_samples: int,
):
    import numpy as np
    import pandas as pd
    import qlib
    from qlib.constant import REG_CN
    from qlib.data.dataset.handler import DataHandlerLP

    candidates = _model_dataset_candidates(database_path, run, params)
    if not candidates:
        raise FileNotFoundError("No saved model-and-dataset pair is available for SHAP")

    qlib.init(provider_uri=provider_uri.expanduser().resolve(), region=REG_CN, kernels=1)
    samples_per_model = max(1, math.ceil(max_samples / len(candidates)))
    absolute_sums: dict[str, float] = defaultdict(float)
    observation_counts: dict[str, int] = defaultdict(int)
    used_run_ids = []
    warnings = []

    for model_path, dataset_path, model_run_id in candidates:
        try:
            model = _load_pickle(model_path)
            dataset = _load_pickle(dataset_path)
            handler = getattr(dataset, "handler", None)
            if handler is not None and not hasattr(handler, "_infer"):
                dataset.setup_data(handler_kwargs={"init_type": DataHandlerLP.IT_LS})
            features = dataset.prepare("test", col_set="feature", data_key=DataHandlerLP.DK_I)
            if features.empty:
                raise ValueError("test feature set is empty")
            if len(features) > samples_per_model:
                features = features.sample(samples_per_model, random_state=42)
            names = _feature_names(features.columns)
            shap_values = _tree_shap_values(model, features)
            for index, name in enumerate(names):
                absolute_sums[name] += float(np.abs(shap_values[:, index]).sum())
                observation_counts[name] += int(shap_values.shape[0])
            used_run_ids.append(model_run_id)
        except Exception as error:  # keep a report when one rolling window is unusable
            warnings.append(f"{model_run_id}: {type(error).__name__}: {error}")

    if not used_run_ids:
        detail = "; ".join(warnings) if warnings else "no compatible tree model"
        raise RuntimeError(f"Unable to calculate SHAP: {detail}")

    importance = pd.Series(
        {
            name: absolute_sums[name] / observation_counts[name]
            for name in absolute_sums
            if observation_counts[name]
        },
        name="mean_abs_shap",
    ).sort_values(ascending=False)
    importance.index.name = "feature"
    return importance, used_run_ids, warnings


def plot_shap_importance(importance, top_n: int = 30) -> str:
    plt = _configure_matplotlib()
    selected = importance.head(top_n).sort_values()
    figure_height = max(6.0, 0.28 * len(selected) + 1.5)
    figure, axis = plt.subplots(figsize=(10, figure_height))
    axis.barh(selected.index, selected.values, color="#4C78A8")
    axis.set_title(f"TreeSHAP Feature Importance (Top {len(selected)})")
    axis.set_xlabel("Mean absolute SHAP value")
    axis.set_ylabel("Feature")
    figure.tight_layout()
    image_uri = _figure_data_uri(figure, dpi=170)
    plt.close(figure)
    return image_uri


def _time_text(timestamp_ms: Optional[int]) -> str:
    if timestamp_ms is None:
        return "-"
    return datetime.fromtimestamp(timestamp_ms / 1000).astimezone().strftime("%Y-%m-%d %H:%M:%S %z")


def _markdown_table(headers: list[str], rows: Iterable[Iterable[Any]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value).replace("|", r"\|") for value in row) + " |")
    return lines


def _percentage(value: float) -> str:
    return "-" if not math.isfinite(value) else f"{value * 100:.2f}%"


def generate_detail_report(
    database_path: Path,
    frequency: str,
    run_id: Optional[str] = None,
    output_path: Optional[Path] = None,
    provider_uri: Path = Path("~/.qlib/qlib_data/cn_data"),
    groups: int = 5,
    shap_samples: int = 2000,
    include_shap: bool = False,
) -> tuple[Path, RunInfo, list[str]]:
    database_path = database_path.expanduser().resolve()
    run, skipped_runs = select_run(database_path, frequency, run_id)
    if output_path is None:
        output_path = Path(__file__).resolve().parent / "exp_reports" / run.run_id / "report.md"
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    params = _load_params(database_path, run.run_id)
    metrics = _load_latest_metrics(database_path, run.run_id)
    warnings = []
    if skipped_runs:
        warnings.append(
            f"Skipped {skipped_runs} newer run(s) without complete prediction/label/backtest artifacts."
        )

    pred_label = load_pred_label(run)
    report = load_backtest_report(run, frequency)
    group_returns = calculate_group_returns(pred_label, groups=groups)
    annual_metrics = calculate_annual_metrics(report, frequency)

    cumulative_image = plot_cumulative_excess(report)
    group_image = plot_group_returns(group_returns)

    # 旧版本曾输出独立 PNG/CSV；现在报告目录只保留 Markdown。
    for stale_name in (
        "cumulative_excess.png",
        "group_cumulative_returns.png",
        "annual_excess_metrics.csv",
        "shap_importance.png",
        "shap_importance.csv",
    ):
        (output_path.parent / stale_name).unlink(missing_ok=True)

    shap_image: Optional[str] = None
    shap_used_run_ids: list[str] = []
    if include_shap:
        try:
            shap_importance, shap_used_run_ids, shap_warnings = calculate_shap_importance(
                database_path=database_path,
                run=run,
                params=params,
                provider_uri=provider_uri,
                max_samples=shap_samples,
            )
            shap_image = plot_shap_importance(shap_importance)
            warnings.extend(f"SHAP window skipped: {warning}" for warning in shap_warnings)
        except Exception as error:
            warnings.append(f"SHAP unavailable: {type(error).__name__}: {error}")

    model_name = params.get("qlib.model.class", params.get("model.class", "-"))
    instrument = params.get(
        "qlib.dataset.instruments", params.get("dataset.kwargs.handler.kwargs.instruments", "-")
    )
    test_start = params.get("qlib.dataset.test_start", "-")
    test_end = params.get("qlib.dataset.test_end", "-")
    generated_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z")

    lines = [
        f"# Qlib Run Report: {run.run_id}",
        "",
        f"- Generated at: `{generated_at}`",
        f"- MLflow database: `{database_path}`",
        f"- Artifact directory: `{run.artifact_dir}`",
        "",
        "## Run Summary",
        "",
    ]
    lines.extend(
        _markdown_table(
            ["Field", "Value"],
            [
                ("Experiment", f"{run.experiment_id} / {run.experiment_name}"),
                ("Run ID", run.run_id),
                ("Status", run.status),
                ("Model", model_name),
                ("Instrument", instrument),
                ("Test period", f"{test_start} ~ {test_end}"),
                ("Start time", _time_text(run.start_time)),
                ("End time", _time_text(run.end_time)),
                ("Rank IC", metrics.get("Rank IC", "-")),
                ("Rank ICIR", metrics.get("Rank ICIR", "-")),
            ],
        )
    )
    lines.extend(
        [
            "",
            "## Cumulative Excess Return",
            "",
            "The curves use Qlib's arithmetic accumulation convention. The net curve deducts transaction costs.",
            "",
            f"![Cumulative excess return]({cumulative_image})",
            "",
            "## Cumulative Return by Score Group",
            "",
            f"Predictions are divided into {groups} cross-sectional groups each day; Group1 has the highest scores.",
            "",
            f"![Grouped cumulative returns]({group_image})",
            "",
            "## Annual Excess Performance (After Cost)",
            "",
        ]
    )
    annual_rows = []
    for year, row in annual_metrics.iterrows():
        annual_rows.append(
            (
                year,
                int(row["observations"]),
                _percentage(row["excess_return"]),
                _percentage(row["excess_volatility"]),
                f"{row['excess_sharpe']:.3f}" if math.isfinite(row["excess_sharpe"]) else "-",
                _percentage(row["excess_max_drawdown"]),
            )
        )
    lines.extend(
        _markdown_table(
            ["Year", "Days", "Excess return", "Excess volatility", "Excess Sharpe", "Excess max drawdown"],
            annual_rows,
        )
    )
    if include_shap:
        lines.extend(["", "## SHAP Feature Importance", ""])
        if shap_image is not None:
            lines.extend(
                [
                    "Mean absolute TreeSHAP values are aggregated across the saved rolling models and sampled test rows.",
                    "",
                    f"![SHAP feature importance]({shap_image})",
                    "",
                    f"Model run IDs used: `{', '.join(shap_used_run_ids)}`",
                ]
            )
        else:
            lines.append("SHAP could not be generated from the artifacts saved by this run.")
    if warnings:
        lines.extend(["", "## Notes", ""])
        lines.extend(f"- {warning}" for warning in warnings)

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path, run, warnings


def main() -> None:
    """加载 MLflow run 摘要，打印到终端并写入 Markdown。"""
    args = parse_args()
    frequency = args.frequency.strip()
    detail_request = args.detail_shap if args.detail_shap is not None else args.detail
    if detail_request is not None:
        requested_run_id = None if detail_request == "latest" else detail_request
        try:
            output_path, run, warnings = generate_detail_report(
                database_path=args.db,
                frequency=frequency,
                run_id=requested_run_id,
                output_path=args.detail_output,
                provider_uri=args.provider_uri,
                groups=args.groups,
                shap_samples=args.shap_samples,
                include_shap=args.detail_shap is not None,
            )
        except (FileNotFoundError, RuntimeError, ValueError) as error:
            raise SystemExit(f"Unable to generate detailed report: {error}") from error
        print(f"Detailed report run: {run.run_id} ({run.experiment_name})")
        print(f"Detailed Markdown saved to: {output_path}")
        for warning in warnings:
            print(f"Warning: {warning}")
        return
    rows = load_run_rows(
        args.db,
        frequency,
        args.include_deleted,
        args.n,
        include_child_runs=args.include_child_runs,
    )
    print_table(rows)
    output_path = write_markdown(rows, args.output, args.db, frequency)
    print(f"Markdown saved to: {output_path}")


if __name__ == "__main__":
    main()
