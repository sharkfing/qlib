from copy import deepcopy
from pathlib import Path
from typing import Optional, Union

from qlib.contrib.data.handler import check_transform_proc
from qlib.data.dataset.handler import DataHandler, DataHandlerLP
from qlib.data.dataset.loader import DataLoader
from qlib.log import get_module_logger
from qlib.utils import init_instance_by_config
from qlib.utils.serial import Serializable
from qlib.workflow.task.utils import replace_task_handler_with_cache


class HandlerCacheLoader(DataLoader, Serializable):
    """通过公共 Handler cache URI 延迟加载原始数据。"""

    def __init__(self, handler_uri: str, fetch_kwargs: Optional[dict] = None):
        """初始化只保留 cache 引用的 DataLoader。

        Parameters
        ----------
        handler_uri : str
            公共 Handler cache 的文件 URI。
        fetch_kwargs : Optional[dict]
            传给底层 Handler ``fetch`` 方法的参数。

        Returns
        -------
        None
            实际 Handler 在第一次读取数据时才加载。
        """
        if not isinstance(handler_uri, str) or not handler_uri:
            raise TypeError("handler_uri must be a non-empty string")

        Serializable.__init__(self)
        self.handler_uri = handler_uri
        self.fetch_kwargs = {"col_set": DataHandler.CS_RAW}
        if fetch_kwargs is not None:
            self.fetch_kwargs.update(fetch_kwargs)
        # 私有属性在 dump_all=False 时不会写入 MLflow dataset artifact。
        self._handler = None

    def _get_handler(self) -> DataHandler:
        """加载并缓存公共 Data Handler。

        Returns
        -------
        DataHandler
            从 ``handler_uri`` 恢复的公共 Handler。
        """
        if getattr(self, "_handler", None) is None:
            try:
                self._handler = init_instance_by_config(self.handler_uri, accept_types=DataHandler)
            except FileNotFoundError as exception:
                raise FileNotFoundError(
                    f"Handler cache does not exist: {self.handler_uri}"
                ) from exception
        return self._handler

    def load(self, instruments=None, start_time=None, end_time=None):
        """读取指定时间窗口的原始数据。

        Parameters
        ----------
        instruments
            为兼容 DataLoader 接口而保留；公共 Handler 已确定股票池。
        start_time
            数据窗口开始时间。
        end_time
            数据窗口结束时间。

        Returns
        -------
        pandas.DataFrame
            从公共 Handler cache 读取的原始数据。
        """
        if instruments is not None:
            get_module_logger(self.__class__.__name__).warning(
                f"instruments[{instruments}] is ignored"
            )
        return self._get_handler().fetch(
            selector=slice(start_time, end_time),
            level="datetime",
            **self.fetch_kwargs,
        )


class RollingDataHandler(DataHandlerLP):
    """复用原始 Handler 缓存，并按滚动训练窗口重新拟合 Processor。"""

    def __init__(
        self,
        window_start_time=None,
        window_end_time=None,
        infer_processors=None,
        learn_processors=None,
        fit_start_time=None,
        fit_end_time=None,
        handler_config=None,
        instruments=None,
        start_time=None,
        end_time=None,
        label=None,
        freq=None,
        cache_raw_handler: bool = True,
        cache_dir: Optional[Union[str, Path]] = None,
    ):
        """初始化滚动数据处理 Handler。

        Parameters
        ----------
        window_start_time
            当前滚动任务读取数据的开始时间。
        window_end_time
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
        instruments
            原始 Handler 使用的股票池；未指定时保留 ``handler_config`` 中的值。
        start_time
            原始 Handler cache 的固定开始时间。
        end_time
            原始 Handler cache 的固定结束时间。
        label
            原始 Handler 使用的标签表达式；支持 Rolling 动态覆盖 horizon。
        freq
            原始 Handler 使用的数据频率；未指定时保留其默认值或配置值。
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

        configured_handler_config = self._configure_handler_config(
            handler_config=handler_config,
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            label=label,
            freq=freq,
        )
        prepared_handler_config = self._prepare_raw_handler_config(
            handler_config=configured_handler_config,
            cache_raw_handler=cache_raw_handler,
            cache_dir=cache_dir,
        )
        infer_processors = check_transform_proc(infer_processors, fit_start_time, fit_end_time)
        learn_processors = check_transform_proc(learn_processors, fit_start_time, fit_end_time)

        if isinstance(prepared_handler_config, str):
            # cache URI 只作为引用序列化；实际 Handler 保存在私有延迟加载属性中。
            data_loader = HandlerCacheLoader(handler_uri=prepared_handler_config)
        else:
            # 未启用公共 cache 时保持 DataLoaderDH 对配置或 Handler 实例的兼容。
            data_loader = {
                "class": "DataLoaderDH",
                "kwargs": {
                    "handler_config": prepared_handler_config,
                },
            }

        super().__init__(
            instruments=None,
            start_time=window_start_time,
            end_time=window_end_time,
            data_loader=data_loader,
            infer_processors=infer_processors,
            learn_processors=learn_processors,
        )
        if instruments is None and isinstance(configured_handler_config, dict):
            instruments = configured_handler_config.get("kwargs", {}).get("instruments")
        # 外层 Handler 的 instruments=None 供底层 cache Loader 使用；单独保留 run 元数据。
        self.metadata_instruments = instruments

    @staticmethod
    def _configure_handler_config(
        handler_config,
        instruments=None,
        start_time=None,
        end_time=None,
        label=None,
        freq=None,
    ):
        """将通用数据参数合并到底层 Handler 配置。

        Parameters
        ----------
        handler_config
            底层 Data Handler 配置、实例或已有缓存 URI。
        instruments
            股票池覆盖值。
        start_time
            固定数据范围开始时间。
        end_time
            固定数据范围结束时间。
        label
            标签表达式覆盖值。
        freq
            数据频率覆盖值。

        Returns
        -------
        object
            合并公共参数后的 Handler 配置；非字典输入仅在没有覆盖参数时保持不变。
        """
        if not isinstance(handler_config, dict):
            common_kwargs = {
                "instruments": instruments,
                "start_time": start_time,
                "end_time": end_time,
                "label": label,
                "freq": freq,
            }
            configured_keys = [key for key, value in common_kwargs.items() if value is not None]
            if configured_keys:
                raise ValueError(
                    "Cannot apply common Handler parameters to an existing cache URI or instance: "
                    + ", ".join(configured_keys)
                )
            return handler_config

        configured_handler = deepcopy(handler_config)
        configured_kwargs = configured_handler.setdefault("kwargs", {})
        common_kwargs = {
            "instruments": instruments,
            "start_time": start_time,
            "end_time": end_time,
            "label": label,
            "freq": freq,
        }
        for key, value in common_kwargs.items():
            if value is not None:
                configured_kwargs[key] = value
        return configured_handler

    def config(self, window_start_time=None, window_end_time=None, **kwargs):
        """更新滚动窗口，并映射到 DataHandlerLP 的原生时间参数。

        Parameters
        ----------
        window_start_time
            新滚动窗口的开始时间；未指定时不修改现有值。
        window_end_time
            新滚动窗口的结束时间；未指定时不修改现有值。
        **kwargs
            传给 ``DataHandlerLP.config`` 的其他配置，例如 Processor 拟合区间。

        Returns
        -------
        None
            配置直接更新到当前 Handler 实例。
        """
        if window_start_time is not None:
            kwargs["start_time"] = window_start_time
        if window_end_time is not None:
            kwargs["end_time"] = window_end_time
        super().config(**kwargs)

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
