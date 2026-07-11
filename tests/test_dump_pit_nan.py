# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import struct
import tempfile
import unittest
from pathlib import Path

from scripts.dump_pit import DumpPitData


class DumpPitNaNTest(unittest.TestCase):
    """验证 dump 层跳过 NaN，同时保留合法数值零。"""

    def test_nan_revision_is_skipped_and_zero_is_written(self):
        """NaN 不进入修订链，后续合法零值仍应正常写入。"""
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            source_file = temporary_path / "sh600519.csv"
            qlib_dir = temporary_path / "qlib_data"
            source_file.write_text(
                "date,period,value,field\n"
                "2020-01-01,201904,1.5,roe\n"
                "2020-02-01,201904,,roe\n"
                "2020-03-01,201904,0,roe\n",
                encoding="utf-8",
            )

            dumper = DumpPitData(str(source_file), str(qlib_dir), max_workers=1)
            dumper._dump_pit(source_file, interval="quarterly", overwrite=True)

            data_file = qlib_dir / "financial" / "sh600519" / "roe_q.data"
            raw_records = data_file.read_bytes()
            self.assertEqual(len(raw_records), dumper.DATA_DTYPE_SIZE * 2)

            first_record = struct.unpack(dumper.DATA_DTYPE, raw_records[: dumper.DATA_DTYPE_SIZE])
            second_record = struct.unpack(dumper.DATA_DTYPE, raw_records[dumper.DATA_DTYPE_SIZE :])
            self.assertAlmostEqual(first_record[2], 1.5)
            self.assertEqual(first_record[3], dumper.DATA_DTYPE_SIZE)
            self.assertEqual(second_record[2], 0.0)
            self.assertEqual(second_record[3], dumper.NA_INDEX)


if __name__ == "__main__":
    unittest.main()
