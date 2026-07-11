# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import platform
from pathlib import Path
from tempfile import TemporaryDirectory
import time
import unittest
from urllib.parse import quote

import mlflow
from mlflow.tracking._tracking_service import utils as tracking_utils

from qlib.config import get_default_mlflow_storage_uris
from qlib.workflow.expm import MLflowExpManager


class MLflowTest(unittest.TestCase):
    def setUp(self) -> None:
        """为每个测试准备独立的 SQLite 与 artifact 目录。"""
        self._temporary_directory = TemporaryDirectory(prefix="qlib_mlflow_test_")
        self.TMP_PATH = Path(self._temporary_directory.name).resolve()
        self.tracking_uri, self.artifact_root = get_default_mlflow_storage_uris(self.TMP_PATH)
        self.clients = []

    def tearDown(self) -> None:
        """释放 Windows 上的 SQLite 文件句柄并清理临时目录。"""
        for client in self.clients:
            store = client._tracking_client.store
            engine = getattr(store, "engine", None)
            if engine is not None:
                engine.dispose()
        tracking_utils._tracking_store_registry._get_store_with_resolved_uri.cache_clear()
        self._temporary_directory.cleanup()

    def test_creating_client(self):
        """
        Please refer to qlib/workflow/expm.py:MLflowExpManager._client
        we don't cache _client (this is helpful to reduce maintainance work when MLflowExpManager's uri is chagned)

        This implementation is based on the assumption creating a client is fast
        """
        # 首次连接需要初始化 SQLite schema，不计入重复创建 client 的耗时。
        warmup_client = mlflow.tracking.MlflowClient(tracking_uri=self.tracking_uri)
        self.clients.append(warmup_client)
        warmup_client.search_experiments()

        start = time.time()
        for i in range(10):
            client = mlflow.tracking.MlflowClient(tracking_uri=self.tracking_uri)
            self.clients.append(client)
        end = time.time()
        elapsed = end - start
        if platform.system() == "Linux":
            self.assertLess(elapsed, 1e-2)  # it can be done in less than 10ms
        else:
            self.assertLess(elapsed, 2e-2)
        print(elapsed)

    def test_sqlite_artifact_location_uses_experiment_name(self):
        """验证 SQLite 元数据与按实验名称隔离的 artifact 路由。"""
        manager = MLflowExpManager(
            uri=self.tracking_uri,
            artifact_root=self.artifact_root,
            default_exp_name="Experiment",
        )
        experiment_name = "实验/Alpha:Test"
        experiment = manager.create_exp(experiment_name)
        client = manager.client
        self.clients.append(client)

        mlflow_experiment = client.get_experiment(experiment.id)
        safe_experiment_name = quote(experiment_name, safe="")
        expected_location = f"{self.artifact_root}/{safe_experiment_name}"
        self.assertEqual(mlflow_experiment.artifact_location, expected_location)

        run = client.create_run(experiment.id)
        self.assertTrue(run.info.artifact_uri.startswith(f"{expected_location}/"))
        self.assertTrue(run.info.artifact_uri.endswith("/artifacts"))


if __name__ == "__main__":
    unittest.main()
