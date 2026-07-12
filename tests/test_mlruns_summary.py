import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from myscripts.summ import load_run_rows, write_markdown


class MlrunsSummaryTest(unittest.TestCase):
    """验证 MLflow 摘要优先读取统一 metadata，并兼容 Trainer 旧字段。"""

    @staticmethod
    def create_database(database_path: Path) -> None:
        """创建包含新旧两种参数格式的最小 MLflow 测试数据库。"""
        with closing(sqlite3.connect(database_path)) as connection:
            connection.executescript(
                """
                CREATE TABLE experiments (
                    experiment_id INTEGER PRIMARY KEY,
                    name TEXT,
                    lifecycle_stage TEXT
                );
                CREATE TABLE runs (
                    run_uuid TEXT PRIMARY KEY,
                    status TEXT,
                    start_time INTEGER,
                    end_time INTEGER,
                    experiment_id INTEGER,
                    lifecycle_stage TEXT
                );
                CREATE TABLE latest_metrics (
                    key TEXT,
                    value REAL,
                    is_nan INTEGER,
                    run_uuid TEXT
                );
                CREATE TABLE params (
                    key TEXT,
                    value TEXT,
                    run_uuid TEXT
                );
                INSERT INTO experiments VALUES (1, 'test_exp', 'active');
                INSERT INTO experiments VALUES (2, 'custom_child_pool', 'active');
                INSERT INTO experiments VALUES (3, 'rolling_models_pending', 'active');
                INSERT INTO runs VALUES (
                    'new_run', 'FINISHED', 1700000002000, 1700000003000, 1, 'active'
                );
                INSERT INTO runs VALUES (
                    'legacy_run', 'FINISHED', 1700000000000, 1700000001000, 1, 'active'
                );
                INSERT INTO runs VALUES (
                    'custom_child_run', 'FINISHED', 1700000004000, 1700000005000, 2, 'active'
                );
                INSERT INTO runs VALUES (
                    'pending_child_run', 'FINISHED', 1700000006000, 1700000007000, 3, 'active'
                );
                INSERT INTO params VALUES ('qlib.model.class', 'LGBModel', 'new_run');
                INSERT INTO params VALUES ('qlib.dataset.instruments', 'csi300', 'new_run');
                INSERT INTO params VALUES ('qlib.dataset.test_start', '2017-01-01', 'new_run');
                INSERT INTO params VALUES ('qlib.dataset.test_end', '2020-08-01', 'new_run');
                INSERT INTO params VALUES ('exp_name', 'custom_child_pool', 'new_run');
                INSERT INTO params VALUES ('model.class', 'XGBModel', 'legacy_run');
                INSERT INTO params VALUES (
                    'dataset.kwargs.handler.kwargs.instruments', 'csi500', 'legacy_run'
                );
                INSERT INTO params VALUES (
                    'dataset.kwargs.segments.test', '(''2021-01-01'', ''2022-01-01'')', 'legacy_run'
                );
                """
            )

    def test_new_and_legacy_metadata_are_loaded(self):
        """统一字段应优先读取，旧 Trainer 字段应作为兼容回退。"""
        with tempfile.TemporaryDirectory() as temporary_directory:
            database_path = Path(temporary_directory) / "mlflow.db"
            self.create_database(database_path)

            rows = load_run_rows(database_path, "1day", include_deleted=False)

        rows_by_run_id = {row["run_id"]: row for row in rows}
        self.assertEqual(rows_by_run_id["new_run"]["model"], "Roll LGBModel")
        self.assertEqual(rows_by_run_id["new_run"]["instrument"], "csi300")
        self.assertEqual(rows_by_run_id["new_run"]["test_period"], "2017-01-01~2020-08-01")
        self.assertEqual(rows_by_run_id["legacy_run"]["model"], "XGBModel")
        self.assertEqual(rows_by_run_id["legacy_run"]["instrument"], "csi500")
        self.assertEqual(rows_by_run_id["legacy_run"]["test_period"], "2021-01-01~2022-01-01")

    def test_child_runs_are_hidden_unless_explicitly_requested(self):
        """默认隐藏引用型和默认前缀型 Rolling 子 run，显式开关可恢复显示。"""
        with tempfile.TemporaryDirectory() as temporary_directory:
            database_path = Path(temporary_directory) / "mlflow.db"
            self.create_database(database_path)

            default_rows = load_run_rows(database_path, "1day", include_deleted=False)
            all_rows = load_run_rows(
                database_path,
                "1day",
                include_deleted=False,
                include_child_runs=True,
            )

        self.assertEqual({row["run_id"] for row in default_rows}, {"new_run", "legacy_run"})
        self.assertEqual(
            {row["run_id"] for row in all_rows},
            {"new_run", "legacy_run", "custom_child_run", "pending_child_run"},
        )

    def test_markdown_output_has_separate_run_id_column(self):
        """Markdown 应为 experiment_name 和 run_id 分别生成独立列。"""
        row = {
            "experiment_id": "1",
            "experiment_name": "test_exp",
            "run_id": "abc123",
            "model": "LGBModel",
            "instrument": "csi300",
            "test_period": "2017-01-01~2020-08-01",
            "status": "FINISHED",
            "start_time": "2026-07-12 10:00:00",
            "end_time": "2026-07-12 10:10:00",
            "rank_ic": "0.050000",
            "rank_icir": "0.400000",
            "annualized_return": "0.100000",
            "information_ratio": "1.200000",
            "max_drawdown": "-0.080000",
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "exp_summaries.md"
            write_markdown([row], output_path, Path("mlflow.db"), "1day")
            markdown = output_path.read_text(encoding="utf-8")

        self.assertIn("| experiment_id | experiment_name | run_id | model |", markdown)
        self.assertIn("| 1 | test_exp | abc123 | LGBModel |", markdown)


if __name__ == "__main__":
    unittest.main()
