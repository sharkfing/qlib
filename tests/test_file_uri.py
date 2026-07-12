# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pickle
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from qlib.contrib.data.handler import Alpha158
from qlib.utils import init_instance_by_config


class FileUriTest(unittest.TestCase):
    """验证配置对象能够跨平台加载绝对 file URI。"""

    def test_absolute_file_uri_is_converted_to_local_path(self):
        """Windows 盘符 URI 和 POSIX URI 都应正确加载 pickle。"""
        expected = {"cache": "loaded"}
        with TemporaryDirectory() as temporary_directory:
            cache_path = Path(temporary_directory) / "handler cache.pkl"
            with cache_path.open("wb") as file:
                pickle.dump(expected, file)

            result = init_instance_by_config(cache_path.resolve().as_uri())

        self.assertEqual(result, expected)

    def test_qlib_handler_uri_passes_restricted_loader(self):
        """用户生成的 Qlib Handler 子类缓存应能通过安全反序列化。"""
        index = pd.MultiIndex.from_tuples(
            [(pd.Timestamp("2020-01-02"), "SH600000")],
            names=["datetime", "instrument"],
        )
        frame = pd.DataFrame({"feature": [1.0]}, index=index)
        handler = object.__new__(Alpha158)
        handler._data = frame
        handler._infer = frame
        handler._learn = frame
        handler.instruments = "csi300"
        handler.start_time = pd.Timestamp("2020-01-02")
        handler.end_time = pd.Timestamp("2020-01-02")
        handler.fetch_orig = True
        handler.drop_raw = False

        with TemporaryDirectory() as temporary_directory:
            cache_path = Path(temporary_directory) / "Alpha158.test.pkl"
            handler.to_pickle(cache_path, dump_all=True)
            loaded_handler = init_instance_by_config(cache_path.resolve().as_uri())

        self.assertIsInstance(loaded_handler, Alpha158)
        pd.testing.assert_frame_equal(loaded_handler._learn, frame)


if __name__ == "__main__":
    unittest.main()
