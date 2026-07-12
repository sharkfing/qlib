# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import unittest
from unittest.mock import Mock, patch

from ruamel.yaml import YAML

from qlib.contrib.rolling.base import Rolling

from myscripts.rolling_method import DEFAULT_CONF, RollingMethod


class RollingMethodTest(unittest.TestCase):
    """验证用户滚动脚本的缓存边界和任务时间同步。"""

    def test_basic_task_keeps_outer_handler_and_updates_label(self):
        """基础任务应保留滚动 Handler 字典，并按 horizon 更新其 label。"""
        rolling = RollingMethod(conf_path=DEFAULT_CONF, horizon=10, rolling_exp="rolling_method_test")

        task = rolling.basic_task()

        handler = task["dataset"]["kwargs"]["handler"]
        self.assertEqual(handler["class"], "Alpha158")
        self.assertEqual(handler["module_path"], "examples.rolling_process_data.rolling_handler")
        self.assertEqual(handler["kwargs"]["label"], ["Ref($close, -11) / Ref($close, -1) - 1"])
        self.assertEqual(str(rolling._full_sample_start_time), "2008-01-01")
        self.assertEqual(str(rolling._full_sample_end_time), "2020-08-01")
        self.assertEqual(str(handler["kwargs"]["window_start_time"]), "2008-01-01")
        self.assertEqual(str(handler["kwargs"]["window_end_time"]), "2020-08-01")
        self.assertEqual(str(handler["kwargs"]["fit_start_time"]), "2008-01-01")
        self.assertEqual(str(handler["kwargs"]["fit_end_time"]), "2014-12-31")

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

    def test_outer_cache_replacement_is_skipped(self):
        """外层 Handler 不应被整体替换为固定 Processor 状态的 PKL。"""
        rolling = RollingMethod.__new__(RollingMethod)
        rolling.logger = Mock()
        task = {"dataset": {"kwargs": {"handler": {"class": "Alpha158"}}}}

        result = rolling._replace_handler_with_cache(task)

        self.assertIs(result, task)
        rolling.logger.info.assert_called_once()

    def test_default_yaml_contains_rolling_processors(self):
        """默认 YAML 应使用滚动 Alpha158，并显式配置特征标准化。"""
        with DEFAULT_CONF.open("r", encoding="utf-8") as config_file:
            config = YAML(typ="safe", pure=True).load(config_file)

        handler = config["task"]["dataset"]["kwargs"]["handler"]
        processor_names = [processor["class"] for processor in handler["kwargs"]["infer_processors"]]
        self.assertEqual(handler["module_path"], "examples.rolling_process_data.rolling_handler")
        self.assertEqual(processor_names, ["ProcessInf", "ZScoreNorm", "Fillna"])
        self.assertEqual(str(handler["kwargs"]["start_time"]), "2008-01-01")
        self.assertEqual(str(handler["kwargs"]["end_time"]), "2020-08-01")
        self.assertNotIn("window_start_time", handler["kwargs"])
        self.assertNotIn("window_end_time", handler["kwargs"])
        self.assertNotIn("fit_start_time", handler["kwargs"])
        self.assertNotIn("fit_end_time", handler["kwargs"])


if __name__ == "__main__":
    unittest.main()
