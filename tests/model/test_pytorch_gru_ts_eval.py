# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import unittest
from unittest.mock import Mock, patch

from qlib.contrib.model.pytorch_gru_ts import GRU


class _MockSampler:
    """为 GRU 训练循环测试提供最小数据接口。"""

    empty = False

    def config(self, **kwargs):
        """接收模型设置的缺失值填充方式。"""

    def __len__(self):
        """返回一个非空样本数量。"""
        return 1


class GRUTrainEvaluationTest(unittest.TestCase):
    """验证 GRU 可选训练集评估行为。"""

    @staticmethod
    def _dataset():
        """创建返回非空训练集和验证集的模拟 Dataset。"""
        dataset = Mock()
        dataset.prepare.side_effect = [_MockSampler(), _MockSampler()]
        return dataset

    def test_disabled_train_evaluation_only_checks_validation(self):
        """关闭开关后每个 epoch 只应调用一次验证集评估。"""
        train_loader = object()
        valid_loader = object()
        evals_result = {}

        with (
            patch("qlib.contrib.model.pytorch_gru_ts.DataLoader", side_effect=[train_loader, valid_loader]),
            patch("qlib.contrib.model.pytorch_gru_ts.get_or_create_path", return_value="unused"),
            patch("qlib.contrib.model.pytorch_gru_ts.torch.save"),
        ):
            model = GRU(n_epochs=2, early_stop=1, eval_train=False, GPU=-1)
            model.train_epoch = Mock()
            model.test_epoch = Mock(side_effect=[(0.0, 0.5), (0.0, 0.4)])
            model.fit(self._dataset(), evals_result=evals_result)

        self.assertEqual(model.train_epoch.call_count, 2)
        self.assertEqual(model.test_epoch.call_count, 2)
        for evaluation_call in model.test_epoch.call_args_list:
            self.assertIs(evaluation_call.args[0], valid_loader)
        self.assertEqual(evals_result["train"], [])
        self.assertEqual(evals_result["valid"], [0.5, 0.4])

    def test_enabled_train_evaluation_preserves_original_behavior(self):
        """默认开启时仍应分别评估验证集和训练集。"""
        train_loader = object()
        valid_loader = object()
        evals_result = {}

        with (
            patch("qlib.contrib.model.pytorch_gru_ts.DataLoader", side_effect=[train_loader, valid_loader]),
            patch("qlib.contrib.model.pytorch_gru_ts.get_or_create_path", return_value="unused"),
            patch("qlib.contrib.model.pytorch_gru_ts.torch.save"),
        ):
            model = GRU(n_epochs=1, eval_train=True, GPU=-1)
            model.train_epoch = Mock()
            model.test_epoch = Mock(side_effect=[(0.0, 0.6), (0.0, 0.5)])
            model.fit(self._dataset(), evals_result=evals_result)

        self.assertEqual(model.test_epoch.call_args_list[0].args[0], train_loader)
        self.assertEqual(model.test_epoch.call_args_list[1].args[0], valid_loader)
        self.assertEqual(evals_result["train"], [0.6])
        self.assertEqual(evals_result["valid"], [0.5])

    def test_eval_train_requires_boolean(self):
        """字符串等非布尔值不应静默改变评估行为。"""
        with self.assertRaisesRegex(TypeError, "eval_train must be a boolean"):
            GRU(eval_train="false", GPU=-1)


if __name__ == "__main__":
    unittest.main()
