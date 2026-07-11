# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import unittest
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from mlflow.tracking._tracking_service import utils as tracking_utils

from qlib.config import C, get_datacache_dir, get_default_mlflow_storage_uris, get_model_cache_path, get_model_log_path
from qlib.tests import TestAutoData
from qlib.workflow import R
from qlib.workflow.task.utils import replace_task_handler_with_cache


class HandlerCachePathTest(unittest.TestCase):
    """验证 Data Handler 与模型专用缓存不会写入运行目录。"""

    def test_default_handler_cache_path(self):
        """未指定 cache_dir 时使用 datacache 下的 handler_cache。"""
        original_datacache_path = C["datacache_path"]
        try:
            with TemporaryDirectory() as temporary_directory:
                datacache_path = Path(temporary_directory).resolve()
                expected_cache_dir = datacache_path / "handler_cache"
                C["datacache_path"] = datacache_path
                task = {
                    "dataset": {
                        "kwargs": {
                            "handler": {
                                "class": "DummyHandler",
                                "module_path": "tests.mock",
                                "kwargs": {},
                            }
                        }
                    }
                }
                mock_handler = Mock()

                with patch("qlib.workflow.task.utils.init_instance_by_config", return_value=mock_handler):
                    cached_task = replace_task_handler_with_cache(task)

                handler_uri = cached_task["dataset"]["kwargs"]["handler"]
                self.assertTrue(handler_uri.startswith(expected_cache_dir.as_uri()))
                cached_path = mock_handler.to_pickle.call_args.args[0]
                self.assertEqual(cached_path.parent, expected_cache_dir)
                self.assertTrue(expected_cache_dir.is_dir())
        finally:
            C["datacache_path"] = original_datacache_path

    def test_rolling_uses_global_handler_cache(self):
        """Rolling 不再把 handler cache 写入配置文件所在目录。"""
        from qlib.contrib.rolling.base import Rolling

        rolling = Rolling(conf_path=Path("dummy.yaml"))
        task = {"dataset": {"kwargs": {"handler": {"class": "DummyHandler"}}}}
        with patch(
            "qlib.contrib.rolling.base.replace_task_handler_with_cache",
            return_value=task,
        ) as replace_handler:
            result = rolling._replace_handler_with_cache(task)

        replace_handler.assert_called_once_with(task)
        self.assertIs(result, task)

    def test_tra_relative_logdir_uses_model_cache(self):
        """TRA 的历史 output/... 配置统一解析到 datacache/TRA。"""
        with TemporaryDirectory() as temporary_directory:
            datacache_path = Path(temporary_directory).resolve()
            logdir = get_model_cache_path("TRA", "output/Alpha158", datacache_path)

            self.assertEqual(logdir, datacache_path / "TRA" / "Alpha158")
            self.assertTrue(logdir.is_dir())

    def test_ddgda_default_working_dir(self):
        """DDG-DA 模型目录与共享 handler_cache 保持同级。"""
        with TemporaryDirectory() as temporary_directory:
            datacache_path = Path(temporary_directory).resolve()
            model_cache_dir = get_datacache_dir("DDG-DA", datacache_path)
            handler_cache_dir = get_datacache_dir("handler_cache", datacache_path)

            self.assertEqual(model_cache_dir, datacache_path / "DDG-DA")
            self.assertEqual(handler_cache_dir, datacache_path / "handler_cache")
            self.assertTrue(model_cache_dir.is_dir())
            self.assertTrue(handler_cache_dir.is_dir())


class ModelLogPathTest(unittest.TestCase):
    """验证模型日志统一写入项目级 logs 目录。"""

    def test_model_log_path_uses_independent_run_directory(self):
        """相同模型的不同运行应分配不同日志目录。"""
        with TemporaryDirectory() as temporary_directory:
            logs_path = Path(temporary_directory).resolve()
            first_log_path = get_model_log_path("CatBoost", root_path=logs_path)
            second_log_path = get_model_log_path("CatBoost", root_path=logs_path)

            self.assertEqual(first_log_path.parent, logs_path / "CatBoost")
            self.assertEqual(second_log_path.parent, logs_path / "CatBoost")
            self.assertNotEqual(first_log_path, second_log_path)
            self.assertTrue(first_log_path.is_dir())
            self.assertTrue(second_log_path.is_dir())

    def test_catboost_uses_logs_directory_by_default(self):
        """CatBoost 未显式配置 train_dir 时使用全局日志目录。"""
        try:
            from qlib.contrib.model.catboost_model import CatBoostModel
        except ModuleNotFoundError:
            self.skipTest("CatBoost is an optional dependency")

        original_logs_path = C["logs_path"]
        try:
            with TemporaryDirectory() as temporary_directory:
                logs_path = Path(temporary_directory).resolve()
                C["logs_path"] = logs_path
                model = CatBoostModel()

                train_dir = Path(model._params["train_dir"])
                self.assertEqual(train_dir.parent, logs_path / "CatBoost")
                self.assertTrue(train_dir.is_dir())
        finally:
            C["logs_path"] = original_logs_path

    def test_catboost_preserves_explicit_log_settings(self):
        """显式 train_dir 或关闭文件输出时不应生成默认日志目录。"""
        try:
            from qlib.contrib.model.catboost_model import CatBoostModel
        except ModuleNotFoundError:
            self.skipTest("CatBoost is an optional dependency")

        with TemporaryDirectory() as temporary_directory:
            configured_path = Path(temporary_directory).resolve() / "custom"
            configured_model = CatBoostModel(train_dir=str(configured_path))
            disabled_model = CatBoostModel(allow_writing_files=False)

            self.assertEqual(Path(configured_model._params["train_dir"]), configured_path)
            self.assertNotIn("train_dir", disabled_model._params)


class WorkflowTest(TestAutoData):
    TMP_PATH = Path("./.mlruns_tmp/workflow").resolve()
    TRACKING_URI, ARTIFACT_ROOT = get_default_mlflow_storage_uris(TMP_PATH)
    _setup_kwargs = {
        "exp_manager": {
            "class": "MLflowExpManager",
            "module_path": "qlib.workflow.expm",
            "kwargs": {
                "uri": TRACKING_URI,
                "artifact_root": ARTIFACT_ROOT,
                "default_exp_name": "Experiment",
            },
        }
    }

    def tearDown(self) -> None:
        """释放 SQLite 连接后清理测试产物。"""
        client = R.exp_manager.client
        store = client._tracking_client.store
        engine = getattr(store, "engine", None)
        if engine is not None:
            engine.dispose()
        tracking_utils._tracking_store_registry._get_store_with_resolved_uri.cache_clear()
        if self.TMP_PATH.exists():
            shutil.rmtree(self.TMP_PATH)

    def test_get_local_dir(self):
        """验证 qlib 对象保存、读取及本地 artifact 目录定位。"""
        self.TMP_PATH.mkdir(parents=True, exist_ok=True)

        with R.start(experiment_name="workflow_test"):
            R.save_objects(**{"result.pkl": {"value": 1}})
            recorder_id = R.get_recorder().id

        resume_recorder = R.get_recorder(
            experiment_name="workflow_test",
            recorder_id=recorder_id,
        )
        self.assertEqual(resume_recorder.load_object("result.pkl"), {"value": 1})

        local_dir = Path(resume_recorder.get_local_dir())
        expected_experiment_dir = self.TMP_PATH / "artifacts" / "workflow_test"
        self.assertTrue(local_dir.is_relative_to(expected_experiment_dir))


if __name__ == "__main__":
    unittest.main()
