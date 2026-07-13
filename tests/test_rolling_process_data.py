# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import pickle
import unittest
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import pandas as pd

from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP

from examples.rolling_process_data.rolling_handler import Alpha158 as RollingAlpha158
from examples.rolling_process_data.rolling_handler import HandlerCacheLoader, RollingDataHandler
from examples.rolling_process_data.workflow import RollingDataWorkflow


def contains_dataframe(value, visited=None):
    """递归检查对象是否包含 pandas DataFrame。"""
    if isinstance(value, pd.DataFrame):
        return True
    if visited is None:
        visited = set()
    value_id = id(value)
    if value_id in visited:
        return False
    visited.add(value_id)

    if isinstance(value, dict):
        children = value.values()
    elif isinstance(value, (list, tuple, set)):
        children = value
    elif hasattr(value, "__dict__"):
        children = value.__dict__.values()
    else:
        return False
    return any(contains_dataframe(child, visited) for child in children)


class RollingDataHandlerTest(unittest.TestCase):
    """验证 RollingDataHandler 仅缓存原始 Handler，并保留滚动 Processor。"""

    def test_raw_handler_config_uses_shared_cache(self):
        """字典形式的原始 Handler 应清空 Processor 后交给统一缓存工具。"""
        handler_config = {
            "class": "Alpha158",
            "module_path": "qlib.contrib.data.handler",
            "kwargs": {"instruments": "csi300"},
        }
        cached_uri = "file:///C:/cache/Alpha158.test.pkl"
        cached_task = {"dataset": {"kwargs": {"handler": cached_uri}}}
        cache_dir = Path("datacache/handler_cache")

        with patch(
            "examples.rolling_process_data.rolling_handler.replace_task_handler_with_cache",
            return_value=cached_task,
        ) as replace_handler:
            result = RollingDataHandler._prepare_raw_handler_config(
                handler_config=handler_config,
                cache_raw_handler=True,
                cache_dir=cache_dir,
            )

        cache_task = replace_handler.call_args.args[0]
        raw_kwargs = cache_task["dataset"]["kwargs"]["handler"]["kwargs"]
        self.assertEqual(result, cached_uri)
        self.assertEqual(raw_kwargs["infer_processors"], [])
        self.assertEqual(raw_kwargs["learn_processors"], [])
        self.assertNotIn("infer_processors", handler_config["kwargs"])
        replace_handler.assert_called_once_with(cache_task, cache_dir=cache_dir)

    def test_existing_handler_uri_skips_cache_creation(self):
        """已有 Handler URI 应直接复用，不重复构建缓存。"""
        handler_uri = "file:///C:/cache/Alpha158.test.pkl"

        with patch(
            "examples.rolling_process_data.rolling_handler.replace_task_handler_with_cache"
        ) as replace_handler:
            result = RollingDataHandler._prepare_raw_handler_config(
                handler_config=handler_uri,
                cache_raw_handler=True,
                cache_dir=None,
            )

        self.assertEqual(result, handler_uri)
        replace_handler.assert_not_called()

    def test_constructor_routes_cached_handler_to_data_loader(self):
        """外层 Processor 与原始缓存 URI 应分别交给 Handler 和延迟 Loader。"""
        with patch.object(
            RollingDataHandler,
            "_prepare_raw_handler_config",
            return_value="file:///C:/cache/Alpha158.test.pkl",
        ), patch.object(DataHandlerLP, "__init__", return_value=None) as initialize_handler:
            RollingDataHandler(
                handler_config={"class": "Alpha158"},
                window_start_time="2010-01-01",
                window_end_time="2017-12-31",
                infer_processors=[],
                learn_processors=[],
            )

        data_loader = initialize_handler.call_args.kwargs["data_loader"]
        self.assertIsInstance(data_loader, HandlerCacheLoader)
        self.assertEqual(data_loader.handler_uri, "file:///C:/cache/Alpha158.test.pkl")
        self.assertEqual(initialize_handler.call_args.kwargs["start_time"], "2010-01-01")
        self.assertEqual(initialize_handler.call_args.kwargs["end_time"], "2017-12-31")

    def test_dataset_artifact_keeps_uri_without_raw_dataframe(self):
        """dataset artifact 应保留 cache URI 和处理状态，但不包含原始 DataFrame。"""
        index = pd.MultiIndex.from_product(
            [pd.to_datetime(["2020-01-02", "2020-01-03"]), ["SH600000"]],
            names=["datetime", "instrument"],
        )
        columns = pd.MultiIndex.from_tuples(
            [("feature", "F0"), ("label", "LABEL0")],
        )
        raw_data = pd.DataFrame([[1.0, 0.1], [2.0, 0.2]], index=index, columns=columns)

        with TemporaryDirectory() as temporary_directory:
            cache_path = Path(temporary_directory) / "raw_handler.pkl"
            DataHandlerLP.from_df(raw_data).to_pickle(cache_path, dump_all=True)
            rolling_handler = RollingDataHandler(
                handler_config=cache_path.resolve().as_uri(),
                window_start_time="2020-01-02",
                window_end_time="2020-01-03",
                infer_processors=[],
                learn_processors=[],
            )
            dataset = DatasetH(
                handler=rolling_handler,
                segments={"test": ("2020-01-02", "2020-01-03")},
            )
            dataset.config(dump_all=False, recursive=True)

            serialized_dataset = pickle.dumps(dataset)
            restored_dataset = pickle.loads(serialized_dataset)

            self.assertFalse(contains_dataframe(restored_dataset))
            self.assertEqual(
                restored_dataset.handler.data_loader.handler_uri,
                cache_path.resolve().as_uri(),
            )
            self.assertFalse(hasattr(restored_dataset.handler.data_loader, "_handler"))

            restored_dataset.setup_data(handler_kwargs={"init_type": DataHandlerLP.IT_LS})
            restored_feature = restored_dataset.prepare(
                "test",
                col_set="feature",
                data_key=DataHandlerLP.DK_I,
            )
            pd.testing.assert_frame_equal(restored_feature, raw_data["feature"])

    def test_missing_handler_cache_raises_explicit_error(self):
        """公共 cache 缺失时应暴露 URI，不允许静默回退。"""
        loader = HandlerCacheLoader("file:///C:/missing/Alpha158.pkl")
        with patch(
            "examples.rolling_process_data.rolling_handler.init_instance_by_config",
            side_effect=FileNotFoundError("missing"),
        ):
            with self.assertRaisesRegex(FileNotFoundError, "Handler cache does not exist"):
                loader.load(start_time="2020-01-01", end_time="2020-01-02")

    def test_config_maps_window_times_to_data_handler(self):
        """滚动窗口参数应映射到 DataHandlerLP 的原生读取范围。"""
        rolling_handler = RollingDataHandler.__new__(RollingDataHandler)

        with patch.object(DataHandlerLP, "config", return_value=None) as configure_handler:
            rolling_handler.config(
                window_start_time="2011-01-01",
                window_end_time="2018-12-31",
                processor_kwargs={"fit_start_time": "2011-01-01"},
            )

        configure_handler.assert_called_once_with(
            start_time="2011-01-01",
            end_time="2018-12-31",
            processor_kwargs={"fit_start_time": "2011-01-01"},
        )


class RollingDataWorkflowTest(unittest.TestCase):
    """验证示例工作流通过 RollingDataHandler 统一管理原始缓存。"""

    def test_workflow_uses_unified_alpha158_handler(self):
        """workflow 应通过统一 Alpha158 入口管理缓存与滚动 Processor。"""
        workflow = RollingDataWorkflow()
        workflow.rolling_cnt = 1
        workflow._init_qlib = Mock()
        dataset = Mock()
        dataset.prepare.return_value = ("train", "valid", "test")

        with patch(
            "examples.rolling_process_data.workflow.init_instance_by_config",
            return_value=dataset,
        ) as initialize_dataset:
            workflow.rolling_process()

        dataset_config = initialize_dataset.call_args.args[0]
        rolling_handler = dataset_config["kwargs"]["handler"]
        handler_kwargs = rolling_handler["kwargs"]
        self.assertEqual(rolling_handler["class"], "Alpha158")
        self.assertEqual(handler_kwargs["instruments"], "csi300")
        self.assertEqual(handler_kwargs["start_time"], "2010-01-01")
        self.assertEqual(handler_kwargs["end_time"], "2019-12-31")
        self.assertEqual(handler_kwargs["window_start_time"], datetime(2010, 1, 1))
        self.assertEqual(handler_kwargs["window_end_time"], datetime(2014, 12, 31))


class RollingAlpha158Test(unittest.TestCase):
    """验证统一 Alpha158 包装会正确分离原始缓存与滚动 Processor。"""

    def test_wrapper_builds_raw_alpha158_config(self):
        """包装类应将固定数据范围、标签和股票池传给无 Processor 的原生 Alpha158。"""
        infer_processors = [{"class": "ZScoreNorm", "kwargs": {"fields_group": "feature"}}]
        learn_processors = [{"class": "DropnaLabel"}]
        label = ["Ref($close, -11) / Ref($close, -1) - 1"]

        with patch.object(RollingDataHandler, "__init__", return_value=None) as initialize_handler:
            rolling_alpha158 = RollingAlpha158(
                instruments="csi300",
                start_time="2008-01-01",
                end_time="2020-08-01",
                window_start_time="2010-01-01",
                window_end_time="2017-12-31",
                fit_start_time="2010-01-01",
                fit_end_time="2014-12-31",
                infer_processors=infer_processors,
                learn_processors=learn_processors,
                label=label,
            )

        call_kwargs = initialize_handler.call_args.kwargs
        raw_handler = call_kwargs["handler_config"]
        raw_kwargs = raw_handler["kwargs"]
        self.assertEqual(raw_handler["class"], "Alpha158")
        self.assertEqual(raw_handler["module_path"], "qlib.contrib.data.handler")
        self.assertEqual(raw_kwargs["instruments"], "csi300")
        self.assertEqual(raw_kwargs["start_time"], "2008-01-01")
        self.assertEqual(raw_kwargs["end_time"], "2020-08-01")
        self.assertEqual(raw_kwargs["label"], label)
        self.assertEqual(raw_kwargs["infer_processors"], [])
        self.assertEqual(raw_kwargs["learn_processors"], [])
        self.assertIs(call_kwargs["infer_processors"], infer_processors)
        self.assertIs(call_kwargs["learn_processors"], learn_processors)
        self.assertEqual(call_kwargs["window_start_time"], "2010-01-01")
        self.assertEqual(call_kwargs["window_end_time"], "2017-12-31")
        self.assertEqual(rolling_alpha158.metadata_instruments, "csi300")


if __name__ == "__main__":
    unittest.main()
