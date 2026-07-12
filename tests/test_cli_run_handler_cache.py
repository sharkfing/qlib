# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import unittest
from unittest.mock import patch

from qlib.cli.run import _prepare_task_config


class QrunHandlerCacheTest(unittest.TestCase):
    """验证 qrun 顶层 Handler cache 开关。"""

    def test_enabled_cache_replaces_task_handler(self):
        """启用缓存时应在 task_train 前替换 Handler。"""
        task = {"dataset": {"kwargs": {"handler": {"class": "Alpha158"}}}}
        cached_task = {"dataset": {"kwargs": {"handler": "file:///handler.pkl"}}}
        config = {"task": task, "handler_cache": {"enabled": True}}

        with patch("qlib.cli.run.replace_task_handler_with_cache", return_value=cached_task) as replace_handler:
            result = _prepare_task_config(config)

        replace_handler.assert_called_once_with(task)
        self.assertIs(result, cached_task)

    def test_cache_is_disabled_by_default(self):
        """未声明开关时应保持标准 qrun 行为。"""
        task = {"dataset": {"kwargs": {"handler": {"class": "Alpha158"}}}}

        with patch("qlib.cli.run.replace_task_handler_with_cache") as replace_handler:
            result = _prepare_task_config({"task": task})

        replace_handler.assert_not_called()
        self.assertIs(result, task)

    def test_invalid_enabled_value_is_rejected(self):
        """非布尔开关应明确报错，避免静默改变缓存行为。"""
        config = {"task": {}, "handler_cache": {"enabled": "true"}}

        with self.assertRaisesRegex(TypeError, "handler_cache.enabled must be a boolean"):
            _prepare_task_config(config)


if __name__ == "__main__":
    unittest.main()
