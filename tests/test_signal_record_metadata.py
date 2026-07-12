# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from qlib.workflow.record_temp import SignalRecord


class DummyModel:
    """用于验证模型类元数据的测试模型。"""


class SignalRecordMetadataTest(unittest.TestCase):
    """验证 SignalRecord 自动记录模型与数据集配置。"""

    def test_log_run_metadata(self):
        """模型、股票池和配置测试区间应写入当前 recorder。"""
        dataset = SimpleNamespace(
            handler=SimpleNamespace(instruments="csi300"),
            segments={"test": ("2017-01-01", "2020-08-01")},
        )
        recorder = Mock()
        signal_record = SignalRecord(model=DummyModel(), dataset=dataset, recorder=recorder)

        signal_record._log_run_metadata()

        recorder.log_params.assert_called_once_with(
            **{
                "qlib.model.class": "DummyModel",
                "qlib.model.module": __name__,
                "qlib.dataset.instruments": "csi300",
                "qlib.dataset.test_start": "2017-01-01",
                "qlib.dataset.test_end": "2020-08-01",
            }
        )

    def test_slice_test_segment_is_supported(self):
        """DatasetH 使用 slice 表示测试区间时也应正确记录。"""
        dataset = SimpleNamespace(
            handler=SimpleNamespace(instruments="csi500"),
            segments={"test": slice("2021-01-01", "2022-01-01")},
        )
        recorder = Mock()

        SignalRecord(model=DummyModel(), dataset=dataset, recorder=recorder)._log_run_metadata()

        metadata = recorder.log_params.call_args.kwargs
        self.assertEqual(metadata["qlib.dataset.test_start"], "2021-01-01")
        self.assertEqual(metadata["qlib.dataset.test_end"], "2022-01-01")

    def test_rolling_handler_metadata_instruments_are_supported(self):
        """滚动外层 Handler 应从专用字段记录原始股票池。"""
        dataset = SimpleNamespace(
            handler=SimpleNamespace(instruments=None, metadata_instruments="csi300"),
            segments={"test": ("2017-01-01", "2020-08-01")},
        )
        recorder = Mock()

        SignalRecord(model=DummyModel(), dataset=dataset, recorder=recorder)._log_run_metadata()

        metadata = recorder.log_params.call_args.kwargs
        self.assertEqual(metadata["qlib.dataset.instruments"], "csi300")


if __name__ == "__main__":
    unittest.main()
