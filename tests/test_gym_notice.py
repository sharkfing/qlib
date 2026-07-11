# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import subprocess
import sys
import unittest


class GymNoticeTest(unittest.TestCase):
    """验证 Qlib 对 Gym 维护公告的定向屏蔽。"""

    def test_qlib_suppresses_gym_maintenance_notice(self):
        """Qlib 先导入时不应输出 Gym 的停止维护公告。"""
        completed_process = subprocess.run(
            [sys.executable, "-c", "import qlib; import gym"],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed_process.returncode, 0, completed_process.stderr)
        self.assertNotIn("Gym has been unmaintained since 2022", completed_process.stderr)
