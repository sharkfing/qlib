# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""运行带动态 Processor 拟合和可选 TrainerRM worker 的滚动实验。

``worker=0`` 时使用单进程 TrainerR；``worker=n`` 且 ``n>0`` 时，单条命令会自动
创建 n 个 TrainerRM 子进程。MongoDB 任务池内部复用本次 ``rolling_exp``，完整
实验成功后删除任务数据，失败时保留以便排查。
"""

import multiprocessing
import os
import time
from copy import deepcopy
from pathlib import Path
from typing import Union

import fire
import pandas as pd

from qlib import auto_init
from qlib.config import C
from qlib.contrib.rolling.base import Rolling
from qlib.model.trainer import TrainerRM
from qlib.tests.data import GetData
from qlib.workflow import R
from qlib.workflow.task.manage import TaskManager
from examples.rolling_process_data.rolling_handler import RollingDataHandler


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONF = SCRIPT_DIR / "rolling_method_lgbm_Alpha158.yaml"
DEFAULT_PROVIDER_URI = Path("~/.qlib/qlib_data/cn_data").expanduser()


def _run_trainer_rm_worker(experiment_name: str, task_pool: str, qlib_init_kwargs: dict) -> None:
    """在独立子进程中初始化 Qlib，并消费 MongoDB 滚动任务。

    Parameters
    ----------
    experiment_name : str
        子模型写入的 MLflow experiment 名称。
    task_pool : str
        MongoDB 任务池名称。
    qlib_init_kwargs : dict
        从主进程复制的 Qlib 数据、MongoDB 和 MLflow 配置。

    Returns
    -------
    None
        任务池中没有待处理任务后返回。
    """
    auto_init(**qlib_init_kwargs)
    trainer = TrainerRM(experiment_name=experiment_name, task_pool=task_pool)
    trainer.worker()


class RollingMethod(Rolling):
    """使用原始 Alpha158 cache，并按滚动窗口重新拟合 Processor。"""

    # 覆写父类 Rolling.__init__()：增加多 worker 参数，并继续复用父类初始化。
    def __init__(
        self,
        conf_path: Union[str, Path] = DEFAULT_CONF,
        horizon: int = 20,
        worker: int = 0,
        start_delay: float = 0.0,
        **kwargs,
    ) -> None:
        """初始化滚动模型用户脚本。

        Parameters
        ----------
        conf_path : Union[str, Path]
            使用 rolling Handler 的任务配置文件。
        horizon : int
            预测标签的未来交易日跨度。
        worker : int
            worker 数量。0 表示单进程 ``TrainerR``；大于 0 时自动启动指定
            数量的 ``TrainerRM`` 子进程。
        start_delay : float
            相邻 worker 子进程的启动间隔秒数。
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
        self.worker_count = worker
        self.task_pool = self.rolling_exp if worker > 0 else None
        self.start_delay = start_delay
        if self.task_pool is not None and ("$" in self.task_pool or "\x00" in self.task_pool):
            raise ValueError("rolling_exp contains characters unsupported by MongoDB")
        if self.h_path is not None:
            raise ValueError(
                "rolling_method does not support h_path; raw cache is managed by RollingDataHandler"
            )

    # 覆写父类 Rolling.basic_task()：保留固定全样本范围，并同步滚动窗口时间。
    def basic_task(self) -> dict:
        """构造基础任务，并保存 YAML 中固定的全样本范围。

        Returns
        -------
        dict
            已更新 horizon、但仍保留固定全样本范围的基础任务。
        """
        # 外层 RollingDataHandler 必须保留各窗口的动态 Processor 配置；原始数据缓存由其内部管理。
        task = super().basic_task(enable_handler_cache=False)
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

    # 覆写父类 Rolling.get_task_list()：为每个滚动任务补充 Handler 时间参数。
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

    def _prepare_raw_handler_cache(self, task: dict) -> str:
        """在向 MongoDB 提交任务前创建或复用公共原始 Handler 缓存。

        Parameters
        ----------
        task : dict
            已完成滚动时间同步的单个任务配置。

        Returns
        -------
        str
            公共原始 Handler 缓存 URI。
        """
        handler_config = task["dataset"]["kwargs"]["handler"]
        if not isinstance(handler_config, dict) or handler_config.get("class") != "RollingDataHandler":
            raise TypeError("Multi-worker training requires a RollingDataHandler config")

        handler_kwargs = handler_config.get("kwargs", {})
        cache_raw_handler = handler_kwargs.get("cache_raw_handler", True)
        if cache_raw_handler is not True:
            raise ValueError("Multi-worker training requires cache_raw_handler=True")

        cache_uri = RollingDataHandler.prepare_raw_handler(
            handler_config=handler_kwargs.get("handler_config"),
            instruments=handler_kwargs.get("instruments"),
            start_time=handler_kwargs.get("start_time"),
            end_time=handler_kwargs.get("end_time"),
            label=handler_kwargs.get("label"),
            freq=handler_kwargs.get("freq"),
            cache_raw_handler=True,
            cache_dir=handler_kwargs.get("cache_dir"),
        )
        if not isinstance(cache_uri, str):
            raise TypeError("Raw Handler cache preparation did not return a cache URI")
        self.logger.info(f"Raw Handler cache is ready before worker startup: {cache_uri}")
        return cache_uri

    # 覆写父类 Rolling._train_rolling_tasks()：在单进程和 TrainerRM 多 worker 之间切换。
    def _train_rolling_tasks(self) -> None:
        """使用 TrainerR 或 TrainerRM 训练滚动任务。

        ``worker_count=0`` 时完全沿用父类单进程流程。否则主进程预建
        缓存和 MongoDB 任务，再错峰启动子进程并收集结果。

        Returns
        -------
        None
            所有滚动任务完成后返回。
        """
        if self.worker_count == 0:
            super()._train_rolling_tasks()
            return

        task_list = self.get_task_list()
        if not task_list:
            raise ValueError("No rolling tasks were generated")

        task_manager = TaskManager(task_pool=self.task_pool)
        existing_task_stat = task_manager.task_stat()
        if existing_task_stat:
            raise ValueError(
                f"MongoDB task pool {self.task_pool!r} is not empty: {existing_task_stat}. "
                "Use a new task_pool name to avoid mixing runs."
            )

        # 主进程先创建共享原始 cache，避免多个 worker 同时首次写入同一个文件。
        self._prepare_raw_handler_cache(task_list[0])

        self.logger.info("Deleting previous Rolling results")
        try:
            R.delete_exp(experiment_name=self.rolling_exp)
        except ValueError:
            self.logger.info("No previous rolling results")

        # 先提交任务，确保第一个子进程启动后可以立即领取。
        task_manager.create_task(task_list)
        worker_processes = self._start_worker_processes()
        self._wait_for_worker_processes(worker_processes)

        task_stat = task_manager.task_stat()
        unfinished_count = sum(
            task_stat.get(status, 0)
            for status in (
                TaskManager.STATUS_WAITING,
                TaskManager.STATUS_RUNNING,
                TaskManager.STATUS_PART_DONE,
            )
        )
        if unfinished_count > 0:
            raise RuntimeError(f"TrainerRM workers exited with unfinished tasks: {task_stat}")

        self.logger.info(
            f"TrainerRM workers completed: rolling_exp={self.rolling_exp}, "
            f"task_pool={self.task_pool}, task_stat={task_stat}"
        )
        trainer = TrainerRM(
            experiment_name=self.rolling_exp,
            task_pool=self.task_pool,
            skip_run_task=True,
        )
        trainer(task_list)

    def _start_worker_processes(self) -> list:
        """使用 Windows 可兼容的 spawn 方式错峰启动 TrainerRM worker。

        Returns
        -------
        list
            已启动的 ``multiprocessing.Process`` 列表。
        """
        qlib_init_kwargs = {
            "provider_uri": deepcopy(C.provider_uri),
            "region": C.region,
            "mongo": deepcopy(C.mongo),
            "exp_manager": deepcopy(C.exp_manager),
        }
        process_context = multiprocessing.get_context("spawn")
        worker_processes = []

        try:
            for worker_index in range(self.worker_count):
                process = process_context.Process(
                    target=_run_trainer_rm_worker,
                    args=(self.rolling_exp, self.task_pool, qlib_init_kwargs),
                    name=f"rolling-worker-{worker_index + 1}",
                )
                process.start()
                worker_processes.append(process)
                self.logger.info(
                    f"Started worker {worker_index + 1}/{self.worker_count}: pid={process.pid}"
                )
                if self.start_delay > 0 and worker_index + 1 < self.worker_count:
                    time.sleep(self.start_delay)
        except BaseException:
            self._terminate_worker_processes(worker_processes)
            raise
        return worker_processes

    @staticmethod
    def _wait_for_worker_processes(worker_processes: list) -> None:
        """等待所有 worker 退出，并显式报告异常退出码。

        Parameters
        ----------
        worker_processes : list
            已启动的 worker 进程。

        Returns
        -------
        None
            所有 worker 正常退出后返回。
        """
        try:
            for process in worker_processes:
                process.join()
        except BaseException:
            RollingMethod._terminate_worker_processes(worker_processes)
            raise

        failed_workers = [
            f"{process.name}(pid={process.pid}, exitcode={process.exitcode})"
            for process in worker_processes
            if process.exitcode != 0
        ]
        if failed_workers:
            raise RuntimeError("TrainerRM worker process failed: " + ", ".join(failed_workers))

    @staticmethod
    def _terminate_worker_processes(worker_processes: list) -> None:
        """终止并回收仍在运行的 worker 进程。

        Parameters
        ----------
        worker_processes : list
            需要清理的 worker 进程。

        Returns
        -------
        None
            所有进程均已回收后返回。
        """
        for process in worker_processes:
            if process.is_alive():
                process.terminate()
        for process in worker_processes:
            process.join()

    # 覆写父类 Rolling.run()：实验成功后清理 MongoDB 任务，失败时保留现场。
    def run(self) -> None:
        """运行完整滚动实验，并在成功后清理 MongoDB 任务数据。

        Returns
        -------
        None
            训练、预测拼接、回测和任务清理全部完成后返回。
        """
        try:
            super().run()
        except BaseException:
            if self.worker_count > 0:
                self.logger.warning(
                    f"Rolling experiment failed; MongoDB tasks are retained in {self.task_pool!r}"
                )
            raise

        if self.worker_count > 0:
            task_manager = TaskManager(task_pool=self.task_pool)
            task_manager.remove()
            remaining_task_stat = task_manager.task_stat()
            if remaining_task_stat:
                raise RuntimeError(
                    f"MongoDB task cleanup is incomplete for {self.task_pool!r}: {remaining_task_stat}"
                )
            self.logger.info(f"MongoDB tasks cleaned after successful experiment: {self.task_pool}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    qlib_init_kwargs = {}
    provider_uri = os.environ.get("PROVIDER_URI", "")
    if provider_uri == "":
        if not DEFAULT_PROVIDER_URI.exists():
            GetData().qlib_data(target_dir=DEFAULT_PROVIDER_URI)
        qlib_init_kwargs["provider_uri"] = DEFAULT_PROVIDER_URI
    else:
        qlib_init_kwargs["provider_uri"] = provider_uri
    auto_init(**qlib_init_kwargs)
    fire.Fire(RollingMethod)
