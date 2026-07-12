# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import unittest
from pathlib import Path
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


class RollingFinalMetadataTest(unittest.TestCase):
    """验证滚动模型的最终拼接 run 会保存完整实验元数据。"""

    @staticmethod
    def create_rolling():
        """创建不访问真实配置文件的 Rolling 测试对象。"""
        from qlib.contrib.rolling.base import Rolling

        rolling = Rolling(
            conf_path=Path("dummy.yaml"),
            exp_name="rolling_final",
            rolling_exp="rolling_children",
        )
        rolling._raw_conf = Mock(
            return_value={
                "task": {
                    "model": {
                        "class": "ConfiguredModel",
                        "module_path": "configured.model",
                    },
                    "dataset": {
                        "kwargs": {
                            "handler": {
                                "class": "ConfiguredHandler",
                                "kwargs": {"instruments": "configured_market"},
                            },
                            "segments": {
                                "train": ("2008-01-01", "2014-12-31"),
                                "test": ("2017-01-01", "2020-08-01"),
                            },
                        }
                    },
                }
            }
        )
        return rolling

    def test_final_metadata_uses_child_run_and_full_test_period(self):
        """模型和股票池取实际子 run，测试区间取滚动前的完整配置。"""
        rolling = self.create_rolling()
        child_recorder = Mock()
        child_recorder.list_params.return_value = {
            "qlib.model.class": "LGBModel",
            "qlib.model.module": "qlib.contrib.model.gbdt",
            "qlib.dataset.instruments": "csi300",
            "qlib.dataset.test_start": "2017-01-03",
            "qlib.dataset.test_end": "2017-06-26",
        }

        recorder_manager = Mock()
        recorder_manager.list_recorders.return_value = {"child_run": child_recorder}
        with patch("qlib.contrib.rolling.base.R", recorder_manager):
            metadata = rolling._get_final_run_metadata()

        self.assertEqual(
            metadata,
            {
                "qlib.model.class": "LGBModel",
                "qlib.model.module": "qlib.contrib.model.gbdt",
                "qlib.dataset.instruments": "csi300",
                "qlib.dataset.test_start": "2017-01-01",
                "qlib.dataset.test_end": "2020-08-01",
            },
        )

    def test_ensemble_logs_final_metadata(self):
        """最终 recorder 应在保存拼接预测时同步写入标准元数据。"""
        rolling = self.create_rolling()
        metadata = {
            "qlib.model.class": "LGBModel",
            "qlib.model.module": "qlib.contrib.model.gbdt",
            "qlib.dataset.instruments": "csi300",
            "qlib.dataset.test_start": "2017-01-01",
            "qlib.dataset.test_end": "2020-08-01",
        }
        rolling._get_final_run_metadata = Mock(return_value=metadata)
        collector = Mock(return_value={"pred": "combined_pred", "label": "combined_label"})
        final_recorder = Mock(id="final_run")
        recorder_manager = Mock()
        recorder_manager.start.return_value.__enter__ = Mock()
        recorder_manager.start.return_value.__exit__ = Mock(return_value=False)
        recorder_manager.get_recorder.return_value = final_recorder

        with patch(
            "qlib.contrib.rolling.base.RecorderCollector", return_value=collector
        ), patch("qlib.contrib.rolling.base.R", recorder_manager):
            rolling._ens_rolling()

        recorder_manager.log_params.assert_called_once_with(exp_name="rolling_children", **metadata)
        recorder_manager.save_objects.assert_called_once_with(
            **{"pred.pkl": "combined_pred", "label.pkl": "combined_label"}
        )
        self.assertEqual(rolling._rid, "final_run")


class RollingTypeTest(unittest.TestCase):
    """验证 Rolling 可以选择 expanding 或 sliding 训练窗口。"""

    def test_default_uses_expanding_window(self):
        """未指定 rtype 时保持原有 expanding 行为。"""
        from qlib.contrib.rolling.base import Rolling
        from qlib.workflow.task.gen import RollingGen

        rolling = Rolling(conf_path=Path("dummy.yaml"))

        self.assertEqual(rolling.rtype, RollingGen.ROLL_EX)

    def test_get_task_list_passes_sliding_type(self):
        """指定 sliding 后应将该类型传给 RollingGen。"""
        from qlib.contrib.rolling.base import Rolling
        from qlib.workflow.task.gen import RollingGen

        rolling = Rolling(conf_path=Path("dummy.yaml"), horizon=20, step=120, rtype=RollingGen.ROLL_SD)
        rolling.basic_task = Mock(return_value={"record": []})
        rolling_generator = Mock()

        with patch("qlib.contrib.rolling.base.RollingGen", return_value=rolling_generator) as create_rolling, patch(
            "qlib.contrib.rolling.base.task_generator", return_value=[]
        ) as generate_tasks:
            task_list = rolling.get_task_list()

        self.assertEqual(task_list, [])
        create_rolling.assert_called_once_with(step=120, rtype=RollingGen.ROLL_SD, trunc_days=21)
        generate_tasks.assert_called_once_with({"record": []}, rolling_generator)

    def test_invalid_type_is_rejected(self):
        """非法 rtype 应立即报错，避免静默使用错误窗口。"""
        from qlib.contrib.rolling.base import Rolling

        with self.assertRaisesRegex(ValueError, "rtype must be one of"):
            Rolling(conf_path=Path("dummy.yaml"), rtype="unknown")


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
    @classmethod
    def setUpClass(cls) -> None:
        """在系统临时目录中初始化测试专用的 MLflow 存储。"""
        cls._temporary_directory = TemporaryDirectory(prefix="qlib_workflow_test_")
        cls.TMP_PATH = Path(cls._temporary_directory.name).resolve()
        tracking_uri, artifact_root = get_default_mlflow_storage_uris(cls.TMP_PATH)
        cls._setup_kwargs = {
            "exp_manager": {
                "class": "MLflowExpManager",
                "module_path": "qlib.workflow.expm",
                "kwargs": {
                    "uri": tracking_uri,
                    "artifact_root": artifact_root,
                    "default_exp_name": "Experiment",
                },
            }
        }
        try:
            super().setUpClass()
        except Exception:
            cls._temporary_directory.cleanup()
            raise

    @classmethod
    def tearDownClass(cls) -> None:
        """释放 SQLite 连接并清理系统临时目录。"""
        client = R.exp_manager.client
        store = client._tracking_client.store
        engine = getattr(store, "engine", None)
        if engine is not None:
            engine.dispose()
        tracking_utils._tracking_store_registry._get_store_with_resolved_uri.cache_clear()
        cls._temporary_directory.cleanup()
        super().tearDownClass()

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
