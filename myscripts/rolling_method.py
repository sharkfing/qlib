# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
from pathlib import Path
from typing import Union

import fire
import pandas as pd

from qlib import auto_init
from qlib.contrib.rolling.base import Rolling
from qlib.tests.data import GetData


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONF = SCRIPT_DIR / "rolling_method_lgbm_Alpha158.yaml"


class RollingMethod(Rolling):
    """使用原始 Alpha158 cache，并按滚动窗口重新拟合 Processor。"""

    def __init__(
        self,
        conf_path: Union[str, Path] = DEFAULT_CONF,
        horizon: int = 20,
        **kwargs,
    ) -> None:
        """初始化滚动模型用户脚本。

        Parameters
        ----------
        conf_path : Union[str, Path]
            使用 rolling Handler 的任务配置文件。
        horizon : int
            预测标签的未来交易日跨度。
        **kwargs
            传给 ``qlib.contrib.rolling.base.Rolling`` 的其他参数。

        Returns
        -------
        None
            初始化完成后可通过 Fire 调用 ``run`` 等 Rolling 方法。
        """
        conf_path = Path(conf_path).expanduser().resolve()
        if not conf_path.is_file():
            raise FileNotFoundError(f"Rolling config does not exist: {conf_path}")

        super().__init__(conf_path=conf_path, horizon=horizon, **kwargs)
        if self.h_path is not None:
            raise ValueError("rolling_method does not support h_path; raw cache is managed by rolling_handler.Alpha158")

    def basic_task(self, enable_handler_cache: bool = True) -> dict:
        """构造基础任务，并保存 YAML 中固定的全样本范围。

        Parameters
        ----------
        enable_handler_cache : bool
            是否启用父类的 Handler cache 流程；本类会跳过外层缓存。

        Returns
        -------
        dict
            已更新 horizon、但仍保留固定全样本范围的基础任务。
        """
        task = super().basic_task(enable_handler_cache=enable_handler_cache)
        handler_kwargs = task["dataset"]["kwargs"]["handler"]["kwargs"]
        if handler_kwargs.get("start_time") is None or handler_kwargs.get("end_time") is None:
            raise ValueError("rolling_method requires finite handler start_time and end_time in YAML")
        self._full_sample_start_time = handler_kwargs["start_time"]
        self._full_sample_end_time = handler_kwargs["end_time"]
        return self._sync_handler_times(task)

    def _sync_handler_times(self, task: dict) -> dict:
        """同步单个任务的固定样本、滚动窗口和 Processor 拟合范围。

        Parameters
        ----------
        task : dict
            包含 DatasetH handler 和 segments 的任务配置。

        Returns
        -------
        dict
            已补全后台时间参数的原任务。
        """
        dataset_kwargs = task["dataset"]["kwargs"]
        segments = dataset_kwargs["segments"]
        train_start_time, train_end_time = segments["train"]
        test_end_time = segments["test"][1]

        window_end_time = test_end_time
        if pd.Timestamp(test_end_time) > pd.Timestamp(self._full_sample_end_time):
            window_end_time = self._full_sample_end_time

        handler_kwargs = dataset_kwargs["handler"]["kwargs"]
        # RollingGen.handler_mod 可能扩展原 Handler end_time；这里恢复 YAML 的固定缓存范围。
        handler_kwargs["start_time"] = self._full_sample_start_time
        handler_kwargs["end_time"] = self._full_sample_end_time
        handler_kwargs["window_start_time"] = train_start_time
        handler_kwargs["window_end_time"] = window_end_time
        handler_kwargs["fit_start_time"] = train_start_time
        handler_kwargs["fit_end_time"] = train_end_time
        return task

    def _replace_handler_with_cache(self, task: dict) -> dict:
        """保留外层滚动 Handler 配置，将原始缓存交给 Handler 内部管理。

        Parameters
        ----------
        task : dict
            尚未拆分的基础任务配置。

        Returns
        -------
        dict
            未替换为 PKL URI 的外层任务配置。
        """
        self.logger.info("Outer rolling Handler cache is skipped; raw Alpha158 cache is managed internally")
        return task

    def get_task_list(self) -> list[dict]:
        """生成滚动任务，并按各任务 segments 更新 Handler 时间。

        Returns
        -------
        list[dict]
            已同步数据读取范围与 Processor 拟合范围的滚动任务。
        """
        task_list = super().get_task_list()
        for task in task_list:
            self._sync_handler_times(task)
        return task_list


if __name__ == "__main__":
    qlib_init_kwargs = {}
    if os.environ.get("PROVIDER_URI", "") == "":
        GetData().qlib_data(exists_skip=True)
    else:
        qlib_init_kwargs["provider_uri"] = os.environ["PROVIDER_URI"]
    auto_init(**qlib_init_kwargs)
    fire.Fire(RollingMethod)
