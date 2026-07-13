# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import unittest
from unittest.mock import Mock, call, patch

from ruamel.yaml import YAML

from qlib.contrib.rolling.base import Rolling
from qlib.workflow.task.manage import TaskManager

from myscripts.rolling_method import DEFAULT_CONF, RollingMethod, _run_trainer_rm_worker


class RollingMethodTest(unittest.TestCase):
    """验证用户滚动脚本的缓存边界和任务时间同步。"""

    def test_constructor_derives_task_pool_from_rolling_experiment(self):
        """worker 大于 0 时应自动使用 rolling_exp 作为任务池。"""
        rolling = RollingMethod(
            conf_path=DEFAULT_CONF,
            worker=2,
            rolling_exp="rolling_models_test",
            start_delay=15,
        )

        self.assertEqual(rolling.worker_count, 2)
        self.assertEqual(rolling.task_pool, "rolling_models_test")
        self.assertEqual(rolling.start_delay, 15.0)

    def test_single_process_training_reuses_parent_implementation(self):
        """未指定任务池时应完整沿用父类 TrainerR 流程。"""
        rolling = RollingMethod.__new__(RollingMethod)
        rolling.worker_count = 0
        rolling.task_pool = None

        with patch.object(Rolling, "_train_rolling_tasks", return_value=None) as train_tasks:
            rolling._train_rolling_tasks()

        train_tasks.assert_called_once_with()

    def test_basic_task_keeps_outer_handler_and_updates_label(self):
        """基础任务应保留滚动 Handler 字典，并按 horizon 更新其 label。"""
        rolling = RollingMethod(conf_path=DEFAULT_CONF, horizon=10, rolling_exp="rolling_method_test")

        task = rolling.basic_task()

        handler = task["dataset"]["kwargs"]["handler"]
        self.assertEqual(handler["class"], "RollingDataHandler")
        self.assertEqual(handler["module_path"], "examples.rolling_process_data.rolling_handler")
        self.assertEqual(handler["kwargs"]["handler_config"]["class"], "Alpha158")
        self.assertEqual(handler["kwargs"]["label"], ["Ref($close, -11) / Ref($close, -1) - 1"])
        self.assertEqual(str(rolling._full_sample_start_time), "2008-01-01")
        self.assertEqual(str(rolling._full_sample_end_time), "2020-08-01")
        self.assertEqual(str(handler["kwargs"]["window_start_time"]), "2008-01-01")
        self.assertEqual(str(handler["kwargs"]["window_end_time"]), "2020-08-01")
        self.assertEqual(str(handler["kwargs"]["fit_start_time"]), "2008-01-01")
        self.assertEqual(str(handler["kwargs"]["fit_end_time"]), "2014-12-31")
        self.assertEqual(handler["kwargs"]["freq"], "day")

    def test_task_segments_update_handler_times(self):
        """每个滚动任务应使用自身 segments 设置读取和拟合范围。"""
        task = {
            "dataset": {
                "kwargs": {
                    "handler": {
                        "kwargs": {
                            "start_time": "2008-01-01",
                            "end_time": "2020-12-31",
                        }
                    },
                    "segments": {
                        "train": ("2010-01-01", "2014-12-31"),
                        "valid": ("2015-01-01", "2016-12-31"),
                        "test": ("2017-01-01", "2020-12-31"),
                    },
                }
            }
        }
        rolling = RollingMethod.__new__(RollingMethod)
        rolling._full_sample_start_time = "2008-01-01"
        rolling._full_sample_end_time = "2020-08-01"

        with patch.object(Rolling, "get_task_list", return_value=[task]):
            task_list = rolling.get_task_list()

        handler_kwargs = task_list[0]["dataset"]["kwargs"]["handler"]["kwargs"]
        self.assertEqual(handler_kwargs["window_start_time"], "2010-01-01")
        self.assertEqual(handler_kwargs["window_end_time"], "2020-08-01")
        self.assertEqual(handler_kwargs["fit_start_time"], "2010-01-01")
        self.assertEqual(handler_kwargs["fit_end_time"], "2014-12-31")
        self.assertEqual(handler_kwargs["start_time"], "2008-01-01")
        self.assertEqual(handler_kwargs["end_time"], "2020-08-01")

    def test_default_yaml_contains_rolling_processors(self):
        """默认 YAML 应使用滚动 Alpha158，并显式配置特征标准化。"""
        with DEFAULT_CONF.open("r", encoding="utf-8") as config_file:
            config = YAML(typ="safe", pure=True).load(config_file)

        handler = config["task"]["dataset"]["kwargs"]["handler"]
        processor_names = [processor["class"] for processor in handler["kwargs"]["infer_processors"]]
        self.assertEqual(handler["class"], "RollingDataHandler")
        self.assertEqual(handler["module_path"], "examples.rolling_process_data.rolling_handler")
        self.assertEqual(handler["kwargs"]["handler_config"]["class"], "Alpha158")
        self.assertEqual(processor_names, ["ProcessInf", "RobustZScoreNorm", "Fillna"])
        self.assertEqual(str(handler["kwargs"]["start_time"]), "2008-01-01")
        self.assertEqual(str(handler["kwargs"]["end_time"]), "2020-08-01")
        self.assertNotIn("window_start_time", handler["kwargs"])
        self.assertNotIn("window_end_time", handler["kwargs"])
        self.assertNotIn("fit_start_time", handler["kwargs"])
        self.assertNotIn("fit_end_time", handler["kwargs"])

    def test_prepare_raw_cache_reuses_handler_public_entry(self):
        """主进程预建 cache 时应复用 RollingDataHandler 的公共入口。"""
        rolling = RollingMethod.__new__(RollingMethod)
        rolling.logger = Mock()
        task = {
            "dataset": {
                "kwargs": {
                    "handler": {
                        "class": "RollingDataHandler",
                        "kwargs": {
                            "handler_config": {"class": "Alpha158", "kwargs": {}},
                            "instruments": "csi300",
                            "start_time": "2008-01-01",
                            "end_time": "2020-08-01",
                            "label": ["LABEL"],
                            "freq": "day",
                        },
                    }
                }
            }
        }
        cache_uri = "file:///C:/cache/Alpha158.test.pkl"

        with patch(
            "myscripts.rolling_method.RollingDataHandler.prepare_raw_handler",
            return_value=cache_uri,
        ) as prepare_handler:
            result = rolling._prepare_raw_handler_cache(task)

        self.assertEqual(result, cache_uri)
        prepare_handler.assert_called_once_with(
            handler_config={"class": "Alpha158", "kwargs": {}},
            instruments="csi300",
            start_time="2008-01-01",
            end_time="2020-08-01",
            label=["LABEL"],
            freq="day",
            cache_raw_handler=True,
            cache_dir=None,
        )

    def test_trainer_rm_prepares_tasks_and_collects_worker_results(self):
        """多进程流程应预建 cache、提交任务、运行 worker 并收集结果。"""
        rolling = RollingMethod.__new__(RollingMethod)
        rolling.worker_count = 2
        rolling.task_pool = "rolling_tasks_test"
        rolling.rolling_exp = "rolling_models_test"
        rolling.logger = Mock()
        task_list = [{"task": 1}, {"task": 2}]
        worker_processes = [Mock(), Mock()]

        with patch.object(rolling, "get_task_list", return_value=task_list), patch.object(
            rolling,
            "_prepare_raw_handler_cache",
            return_value="file:///C:/cache/Alpha158.test.pkl",
        ) as prepare_cache, patch("myscripts.rolling_method.TaskManager") as task_manager_class, patch(
            "myscripts.rolling_method.R"
        ) as recorder, patch.object(
            rolling,
            "_start_worker_processes",
            return_value=worker_processes,
        ) as start_workers, patch.object(
            rolling,
            "_wait_for_worker_processes",
        ) as wait_workers, patch("myscripts.rolling_method.TrainerRM") as trainer_class:
            task_manager_class.return_value.task_stat.side_effect = [
                {},
                {TaskManager.STATUS_DONE: 2},
            ]
            recorder.delete_exp.side_effect = ValueError("missing")
            rolling._train_rolling_tasks()

        task_manager_class.assert_called_once_with(task_pool="rolling_tasks_test")
        task_manager_class.return_value.create_task.assert_called_once_with(task_list)
        prepare_cache.assert_called_once_with(task_list[0])
        start_workers.assert_called_once_with()
        wait_workers.assert_called_once_with(worker_processes)
        trainer_class.assert_called_once_with(
            experiment_name="rolling_models_test",
            task_pool="rolling_tasks_test",
            skip_run_task=True,
        )
        trainer_class.return_value.assert_called_once_with(task_list)

    def test_trainer_rm_rejects_nonempty_task_pool(self):
        """已有任务的 MongoDB 集合不得被新实验静默复用。"""
        rolling = RollingMethod.__new__(RollingMethod)
        rolling.worker_count = 2
        rolling.task_pool = "rolling_tasks_test"
        rolling.rolling_exp = "rolling_models_test"
        rolling.logger = Mock()

        with patch.object(rolling, "get_task_list", return_value=[{"task": 1}]), patch(
            "myscripts.rolling_method.TaskManager"
        ) as task_manager_class:
            task_manager_class.return_value.task_stat.return_value = {TaskManager.STATUS_DONE: 1}
            with self.assertRaisesRegex(ValueError, "is not empty"):
                rolling._train_rolling_tasks()

    def test_start_worker_processes_uses_spawn_and_staggers_launches(self):
        """主进程应使用 spawn 创建指定数量的 worker，并错峰启动。"""
        rolling = RollingMethod.__new__(RollingMethod)
        rolling.worker_count = 3
        rolling.task_pool = "rolling_tasks_test"
        rolling.rolling_exp = "rolling_models_test"
        rolling.start_delay = 5.0
        rolling.logger = Mock()
        worker_processes = [Mock(pid=101), Mock(pid=102), Mock(pid=103)]
        process_context = Mock()
        process_context.Process.side_effect = worker_processes
        qlib_config = Mock(
            provider_uri={"__DEFAULT_FREQ": "C:/qlib_data"},
            region="cn",
            mongo={"task_url": "mongodb://localhost:27017/", "task_db_name": "default_task_db"},
            exp_manager={"class": "MLflowExpManager"},
        )

        with patch("myscripts.rolling_method.C", qlib_config), patch(
            "myscripts.rolling_method.multiprocessing.get_context",
            return_value=process_context,
        ) as get_context, patch("myscripts.rolling_method.time.sleep") as sleep:
            result = rolling._start_worker_processes()

        self.assertEqual(result, worker_processes)
        get_context.assert_called_once_with("spawn")
        self.assertEqual(process_context.Process.call_count, 3)
        self.assertEqual(
            [process_call.kwargs["name"] for process_call in process_context.Process.call_args_list],
            ["rolling-worker-1", "rolling-worker-2", "rolling-worker-3"],
        )
        for process in worker_processes:
            process.start.assert_called_once_with()
        self.assertEqual(sleep.call_args_list, [call(5.0), call(5.0)])

    def test_wait_for_worker_processes_rejects_failed_process(self):
        """子进程异常退出时应显式报错。"""
        successful_process = Mock(name="worker-1", pid=101, exitcode=0)
        failed_process = Mock(name="worker-2", pid=102, exitcode=1)

        with self.assertRaisesRegex(RuntimeError, "exitcode=1"):
            RollingMethod._wait_for_worker_processes([successful_process, failed_process])

        successful_process.join.assert_called_once_with()
        failed_process.join.assert_called_once_with()

    def test_worker_process_initializes_qlib_and_runs_trainer(self):
        """子进程入口应使用主进程配置初始化 Qlib。"""
        init_kwargs = {"provider_uri": "C:/qlib_data"}

        with patch("myscripts.rolling_method.auto_init") as initialize_qlib, patch(
            "myscripts.rolling_method.TrainerRM"
        ) as trainer_class:
            _run_trainer_rm_worker("rolling_models_test", "rolling_tasks_test", init_kwargs)

        initialize_qlib.assert_called_once_with(**init_kwargs)
        trainer_class.assert_called_once_with(
            experiment_name="rolling_models_test",
            task_pool="rolling_tasks_test",
        )
        trainer_class.return_value.worker.assert_called_once_with()

    def test_successful_run_cleans_mongodb_tasks(self):
        """完整实验成功后应删除本次 MongoDB 任务数据。"""
        rolling = RollingMethod.__new__(RollingMethod)
        rolling.worker_count = 2
        rolling.task_pool = "rolling_models_test"
        rolling.logger = Mock()

        with patch.object(Rolling, "run") as parent_run, patch(
            "myscripts.rolling_method.TaskManager"
        ) as task_manager_class:
            task_manager_class.return_value.task_stat.return_value = {}
            rolling.run()

        parent_run.assert_called_once_with()
        task_manager_class.assert_called_once_with(task_pool="rolling_models_test")
        task_manager_class.return_value.remove.assert_called_once_with()

    def test_failed_run_retains_mongodb_tasks(self):
        """完整实验失败时应保留 MongoDB 任务以便排查。"""
        rolling = RollingMethod.__new__(RollingMethod)
        rolling.worker_count = 2
        rolling.task_pool = "rolling_models_test"
        rolling.logger = Mock()

        with patch.object(Rolling, "run", side_effect=RuntimeError("training failed")), patch(
            "myscripts.rolling_method.TaskManager"
        ) as task_manager_class:
            with self.assertRaisesRegex(RuntimeError, "training failed"):
                rolling.run()

        task_manager_class.assert_not_called()
        rolling.logger.warning.assert_called_once()


if __name__ == "__main__":
    unittest.main()
