# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import io
import pickle
import unittest
from unittest.mock import Mock

from mlflow.entities import Run

from qlib.utils.pickle_utils import RestrictedUnpickler
from qlib.workflow.task.manage import TaskManager


class TaskManagerQueryTest(unittest.TestCase):
    """验证 TaskManager 查询是否遵守反序列化开关。"""

    @staticmethod
    def _build_task_manager(task_documents):
        """构造不连接 MongoDB 的 TaskManager 测试实例。

        Parameters
        ----------
        task_documents : list[dict]
            模拟 MongoDB 返回的原始任务文档。

        Returns
        -------
        TaskManager
            已注入模拟 Collection 的实例。
        """
        task_manager = TaskManager.__new__(TaskManager)
        task_manager.task_pool = Mock()
        task_manager.task_pool.find.return_value = task_documents
        task_manager._decode_query = Mock(side_effect=lambda query: query)
        return task_manager

    def test_query_decode_false_returns_raw_documents(self):
        """decode=False 时不应反序列化任务定义或结果。"""
        raw_task = {"status": TaskManager.STATUS_DONE, "res": b"runtime-object"}
        task_manager = self._build_task_manager([raw_task])
        task_manager._decode_task = Mock(side_effect=AssertionError("must not decode"))

        tasks = list(task_manager.query(decode=False))

        self.assertEqual(tasks, [raw_task])
        task_manager._decode_task.assert_not_called()

    def test_query_decode_true_keeps_existing_behavior(self):
        """decode=True 时应继续返回反序列化后的任务。"""
        raw_task = {"status": TaskManager.STATUS_DONE, "res": b"serialized"}
        decoded_task = {"status": TaskManager.STATUS_DONE, "res": "recorder"}
        task_manager = self._build_task_manager([raw_task])
        task_manager._decode_task = Mock(return_value=decoded_task)

        tasks = list(task_manager.query(decode=True))

        self.assertEqual(tasks, [decoded_task])
        task_manager._decode_task.assert_called_once_with(raw_task)

    def test_task_stat_does_not_decode_task_results(self):
        """状态统计只应读取 status，不应触碰 Recorder 结果。"""
        task_documents = [
            {"status": TaskManager.STATUS_DONE, "res": b"recorder-1"},
            {"status": TaskManager.STATUS_DONE, "res": b"recorder-2"},
            {"status": TaskManager.STATUS_WAITING, "def": b"task"},
        ]
        task_manager = self._build_task_manager(task_documents)
        task_manager._decode_task = Mock(side_effect=AssertionError("must not decode"))

        task_status = task_manager.task_stat()

        self.assertEqual(
            task_status,
            {
                TaskManager.STATUS_DONE: 2,
                TaskManager.STATUS_WAITING: 1,
            },
        )
        task_manager._decode_task.assert_not_called()

    def test_mlflow_package_is_whitelisted(self):
        """MLflow 升级后新增的运行时类也应由包前缀规则放行。"""
        loaded_class = RestrictedUnpickler(io.BytesIO()).find_class("mlflow.entities.run", "Run")

        self.assertIs(loaded_class, Run)

    def test_similar_module_name_is_not_whitelisted(self):
        """mlflow 前缀必须带包边界，不能误放行名称相似的第三方模块。"""
        with self.assertRaises(pickle.UnpicklingError):
            RestrictedUnpickler(io.BytesIO()).find_class("mlflow_untrusted", "UnsafeClass")


if __name__ == "__main__":
    unittest.main()
