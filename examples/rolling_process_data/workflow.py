#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import qlib
import fire

from datetime import datetime

from qlib.constant import REG_CN
from qlib.data.dataset.handler import DataHandlerLP
from qlib.utils import init_instance_by_config
from qlib.tests.data import GetData


class RollingDataWorkflow:
    MARKET = "csi300"
    start_time = "2010-01-01"
    end_time = "2019-12-31"
    rolling_cnt = 5

    def _init_qlib(self):
        """初始化示例使用的 Qlib 中国市场数据。"""
        provider_uri = "~/.qlib/qlib_data/cn_data"  # target_dir
        GetData().qlib_data(target_dir=provider_uri, region=REG_CN, exists_skip=True)
        qlib.init(provider_uri=provider_uri, region=REG_CN)

    def rolling_process(self):
        """逐年移动窗口，并利用统一原始 Handler cache 重新拟合 Processor。"""
        self._init_qlib()

        train_start_time = (2010, 1, 1)
        train_end_time = (2012, 12, 31)
        valid_start_time = (2013, 1, 1)
        valid_end_time = (2013, 12, 31)
        test_start_time = (2014, 1, 1)
        test_end_time = (2014, 12, 31)

        dataset_config = {
            "class": "DatasetH",
            "module_path": "qlib.data.dataset",
            "kwargs": {
                "handler": {
                    "class": "RollingDataHandler",
                    "module_path": "rolling_handler",
                    "kwargs": {
                        "handler_config": {
                            "class": "Alpha158",
                            "module_path": "qlib.contrib.data.handler",
                            "kwargs": {},
                        },
                        "instruments": self.MARKET,
                        "start_time": self.start_time,
                        "end_time": self.end_time,
                        "freq": "day",
                        "window_start_time": datetime(*train_start_time),
                        "window_end_time": datetime(*test_end_time),
                        "fit_start_time": datetime(*train_start_time),
                        "fit_end_time": datetime(*train_end_time),
                        "infer_processors": [
                            {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature"}},
                        ],
                        "learn_processors": [
                            {"class": "DropnaLabel"},
                            {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}},
                        ],
                    },
                },
                "segments": {
                    "train": (datetime(*train_start_time), datetime(*train_end_time)),
                    "valid": (datetime(*valid_start_time), datetime(*valid_end_time)),
                    "test": (datetime(*test_start_time), datetime(*test_end_time)),
                },
            },
        }

        dataset = init_instance_by_config(dataset_config)

        for rolling_offset in range(self.rolling_cnt):
            print(f"===========rolling{rolling_offset} start===========")
            if rolling_offset:
                dataset.config(
                    handler_kwargs={
                        "window_start_time": datetime(
                            train_start_time[0] + rolling_offset, *train_start_time[1:]
                        ),
                        "window_end_time": datetime(test_end_time[0] + rolling_offset, *test_end_time[1:]),
                        "processor_kwargs": {
                            "fit_start_time": datetime(train_start_time[0] + rolling_offset, *train_start_time[1:]),
                            "fit_end_time": datetime(train_end_time[0] + rolling_offset, *train_end_time[1:]),
                        },
                    },
                    segments={
                        "train": (
                            datetime(train_start_time[0] + rolling_offset, *train_start_time[1:]),
                            datetime(train_end_time[0] + rolling_offset, *train_end_time[1:]),
                        ),
                        "valid": (
                            datetime(valid_start_time[0] + rolling_offset, *valid_start_time[1:]),
                            datetime(valid_end_time[0] + rolling_offset, *valid_end_time[1:]),
                        ),
                        "test": (
                            datetime(test_start_time[0] + rolling_offset, *test_start_time[1:]),
                            datetime(test_end_time[0] + rolling_offset, *test_end_time[1:]),
                        ),
                    },
                )
                dataset.setup_data(
                    handler_kwargs={
                        "init_type": DataHandlerLP.IT_FIT_SEQ,
                    }
                )

            dtrain, dvalid, dtest = dataset.prepare(["train", "valid", "test"])
            print(dtrain, dvalid, dtest)
            ## print or dump data
            print(f"===========rolling{rolling_offset} end===========")


if __name__ == "__main__":
    fire.Fire(RollingDataWorkflow)
