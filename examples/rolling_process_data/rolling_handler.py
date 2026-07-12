from copy import deepcopy
from pathlib import Path
from typing import Optional, Union

from qlib.contrib.data.handler import check_transform_proc
from qlib.data.dataset.handler import DataHandlerLP
from qlib.workflow.task.utils import replace_task_handler_with_cache


class RollingDataHandler(DataHandlerLP):
    """复用原始 Handler 缓存，并按滚动训练窗口重新拟合 Processor。"""

    def __init__(
        self,
        start_time=None,
        end_time=None,
        infer_processors=None,
        learn_processors=None,
        fit_start_time=None,
        fit_end_time=None,
        handler_config=None,
        cache_raw_handler: bool = True,
        cache_dir: Optional[Union[str, Path]] = None,
    ):
        """初始化滚动数据处理 Handler。

        Parameters
        ----------
        start_time
            当前滚动任务读取数据的开始时间。
        end_time
            当前滚动任务读取数据的结束时间。
        infer_processors
            推理数据 Processor 配置；需要拟合的 Processor 使用当前训练窗口。
        learn_processors
            训练数据 Processor 配置。
        fit_start_time
            Processor 拟合区间的开始时间。
        fit_end_time
            Processor 拟合区间的结束时间。
        handler_config
            原始 Data Handler 配置、实例或已有缓存 URI。
        cache_raw_handler : bool
            是否将字典形式的原始 Handler 配置写入共享 Handler cache。
        cache_dir : Optional[Union[str, Path]]
            原始 Handler cache 目录；未指定时使用项目统一目录。

        Returns
        -------
        None
            初始化完成后，数据由 ``DataHandlerLP`` 管理。
        """
        infer_processors = [] if infer_processors is None else infer_processors
        learn_processors = [] if learn_processors is None else learn_processors

        if handler_config is None:
            raise ValueError("handler_config is required")

        prepared_handler_config = self._prepare_raw_handler_config(
            handler_config=handler_config,
            cache_raw_handler=cache_raw_handler,
            cache_dir=cache_dir,
        )
        infer_processors = check_transform_proc(infer_processors, fit_start_time, fit_end_time)
        learn_processors = check_transform_proc(learn_processors, fit_start_time, fit_end_time)

        data_loader = {
            "class": "DataLoaderDH",
            "kwargs": {
                "handler_config": prepared_handler_config,
            },
        }

        super().__init__(
            instruments=None,
            start_time=start_time,
            end_time=end_time,
            data_loader=data_loader,
            infer_processors=infer_processors,
            learn_processors=learn_processors,
        )

    @staticmethod
    def _prepare_raw_handler_config(
        handler_config,
        cache_raw_handler: bool,
        cache_dir: Optional[Union[str, Path]],
    ):
        """规范化原始 Handler，并按需返回共享缓存 URI。

        Parameters
        ----------
        handler_config
            原始 Data Handler 配置、实例或已有缓存 URI。
        cache_raw_handler : bool
            是否缓存字典形式的 Handler 配置。
        cache_dir : Optional[Union[str, Path]]
            自定义缓存目录。

        Returns
        -------
        object
            可直接传入 ``DataLoaderDH`` 的 Handler 配置、实例或 URI。
        """
        if not isinstance(cache_raw_handler, bool):
            raise TypeError("cache_raw_handler must be a boolean")
        if not isinstance(handler_config, dict):
            return handler_config

        raw_handler_config = deepcopy(handler_config)
        raw_handler_kwargs = raw_handler_config.setdefault("kwargs", {})
        raw_handler_kwargs["infer_processors"] = []
        raw_handler_kwargs["learn_processors"] = []

        if not cache_raw_handler:
            return raw_handler_config

        cache_task = {"dataset": {"kwargs": {"handler": raw_handler_config}}}
        cached_task = replace_task_handler_with_cache(cache_task, cache_dir=cache_dir)
        return cached_task["dataset"]["kwargs"]["handler"]


class Alpha158(RollingDataHandler):
    """提供原始特征缓存和滚动 Processor 拟合的统一 Alpha158 Handler。"""

    def __init__(
        self,
        instruments="csi500",
        start_time=None,
        end_time=None,
        fit_start_time=None,
        fit_end_time=None,
        infer_processors=None,
        learn_processors=None,
        data_start_time=None,
        data_end_time=None,
        freq="day",
        label=None,
        filter_pipe=None,
        inst_processors=None,
        cache_raw_handler: bool = True,
        cache_dir: Optional[Union[str, Path]] = None,
        **raw_handler_kwargs,
    ):
        """初始化支持滚动拟合的 Alpha158。

        Parameters
        ----------
        instruments
            股票池名称或股票列表。
        start_time
            当前滚动任务读取数据的开始时间。
        end_time
            当前滚动任务读取数据的结束时间。
        fit_start_time
            当前 Processor 拟合区间的开始时间。
        fit_end_time
            当前 Processor 拟合区间的结束时间。
        infer_processors
            每个滚动窗口重新拟合的推理数据 Processor。
        learn_processors
            每个滚动窗口重新执行的训练数据 Processor。
        data_start_time
            原始 Alpha158 cache 的固定开始时间；默认等于 ``start_time``。
        data_end_time
            原始 Alpha158 cache 的固定结束时间；默认等于 ``end_time``。
        freq
            原始 Alpha158 数据频率。
        label
            原始 Alpha158 标签表达式；支持 Rolling 动态覆盖 horizon。
        filter_pipe
            传给原生 Alpha158 的股票过滤器。
        inst_processors
            传给原生 Alpha158 DataLoader 的股票 Processor。
        cache_raw_handler : bool
            是否创建或复用原始 Alpha158 Handler cache。
        cache_dir : Optional[Union[str, Path]]
            原始 Handler cache 目录。
        **raw_handler_kwargs
            传给原生 ``qlib.contrib.data.handler.Alpha158`` 的其他参数。

        Returns
        -------
        None
            初始化后的对象由 ``RollingDataHandler`` 负责加载和处理数据。
        """
        raw_start_time = start_time if data_start_time is None else data_start_time
        raw_end_time = end_time if data_end_time is None else data_end_time
        raw_kwargs = {
            **raw_handler_kwargs,
            "instruments": instruments,
            "start_time": raw_start_time,
            "end_time": raw_end_time,
            "freq": freq,
            "infer_processors": [],
            "learn_processors": [],
        }
        if label is not None:
            raw_kwargs["label"] = label
        if filter_pipe is not None:
            raw_kwargs["filter_pipe"] = filter_pipe
        if inst_processors is not None:
            raw_kwargs["inst_processors"] = inst_processors

        raw_handler_config = {
            "class": "Alpha158",
            "module_path": "qlib.contrib.data.handler",
            "kwargs": raw_kwargs,
        }
        super().__init__(
            start_time=start_time,
            end_time=end_time,
            fit_start_time=fit_start_time,
            fit_end_time=fit_end_time,
            infer_processors=infer_processors,
            learn_processors=learn_processors,
            handler_config=raw_handler_config,
            cache_raw_handler=cache_raw_handler,
            cache_dir=cache_dir,
        )
