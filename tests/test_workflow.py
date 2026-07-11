# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import unittest
from pathlib import Path
import shutil

from mlflow.tracking._tracking_service import utils as tracking_utils

from qlib.config import get_default_mlflow_storage_uris
from qlib.tests import TestAutoData
from qlib.workflow import R


class WorkflowTest(TestAutoData):
    TMP_PATH = Path("./.mlruns_tmp/workflow").resolve()
    TRACKING_URI, ARTIFACT_ROOT = get_default_mlflow_storage_uris(TMP_PATH)
    _setup_kwargs = {
        "exp_manager": {
            "class": "MLflowExpManager",
            "module_path": "qlib.workflow.expm",
            "kwargs": {
                "uri": TRACKING_URI,
                "artifact_root": ARTIFACT_ROOT,
                "default_exp_name": "Experiment",
            },
        }
    }

    def tearDown(self) -> None:
        """释放 SQLite 连接后清理测试产物。"""
        client = R.exp_manager.client
        store = client._tracking_client.store
        engine = getattr(store, "engine", None)
        if engine is not None:
            engine.dispose()
        tracking_utils._tracking_store_registry._get_store_with_resolved_uri.cache_clear()
        if self.TMP_PATH.exists():
            shutil.rmtree(self.TMP_PATH)

    def test_get_local_dir(self):
        """验证 qlib 对象保存、读取及本地 artifact 目录定位。"""
        self.TMP_PATH.mkdir(parents=True, exist_ok=True)

        with R.start(experiment_name="workflow_test"):
            R.save_objects(**{"result.pkl": {"value": 1}})
            recorder_id = R.get_recorder().id

        resume_recorder = R.get_recorder(
            experiment_name="workflow_test",
            recorder_id=recorder_id,
        )
        self.assertEqual(resume_recorder.load_object("result.pkl"), {"value": 1})

        local_dir = Path(resume_recorder.get_local_dir())
        expected_experiment_dir = self.TMP_PATH / "artifacts" / "workflow_test"
        self.assertTrue(local_dir.is_relative_to(expected_experiment_dir))


if __name__ == "__main__":
    unittest.main()
