"""按 run 汇总打印项目 MLflow 实验记录。"""

from __future__ import annotations

import argparse
import ast
import math
import re
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


# ═══ 默认路径与输出字段 ═══════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = PROJECT_ROOT / "mlruns" / "mlflow.db"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "myscripts" / "exp_summaries.md"

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


def main() -> None:
    """加载 MLflow run 摘要，打印到终端并写入 Markdown。"""
    args = parse_args()
    frequency = args.frequency.strip()
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
