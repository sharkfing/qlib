# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import math
import unittest
from unittest.mock import patch

from scripts.data_collector.pit.collector import PitCollector


class FakeBaostockResponse:
    """模拟 Baostock 的逐行响应对象。"""

    error_code = "0"

    def __init__(self, fields, rows):
        """保存字段、响应行并初始化游标。"""
        self.fields = fields
        self._rows = rows
        self._index = -1

    def next(self):
        """移动到下一行并返回是否存在数据。"""
        self._index += 1
        return self._index < len(self._rows)

    def get_row_data(self):
        """返回当前响应行。"""
        return self._rows[self._index]


class PitCollectorSymbolTest(unittest.TestCase):
    """验证 PIT collector 的可选自定义股票参数。"""

    def test_custom_symbols_are_normalized(self):
        """六位代码应自动补充市场后缀，并去重排序。"""
        symbols = PitCollector.parse_symbols("600519,000725.sz,600519.ss")
        self.assertEqual(symbols, ["000725.sz", "600519.ss"])

    def test_missing_custom_symbols_keeps_original_mode(self):
        """未传入自定义股票时返回 None，由原东方财富股票池逻辑接管。"""
        self.assertIsNone(PitCollector.parse_symbols(None))

    def test_invalid_custom_symbol_is_rejected(self):
        """格式无效的股票代码应明确报错。"""
        with self.assertRaises(ValueError):
            PitCollector.parse_symbols("invalid")


class PitCollectorPlaceholderTest(unittest.TestCase):
    """验证 Baostock 来源特有的占位零过滤。"""

    def test_performance_express_placeholder_roe_becomes_nan(self):
        """快报核心资产字段全零时标记占位 ROE，同时保留合法 ROE 零值。"""
        fields = [
            "performanceExpPubDate",
            "performanceExpStatDate",
            "performanceExpressROEWa",
            "performanceExpressTotalAsset",
            "performanceExpressNetAsset",
        ]
        response = FakeBaostockResponse(
            fields,
            [
                ["2019-07-13", "2019-06-30", "0", "0", "0"],
                ["2019-07-14", "2019-06-30", "0", "100", "50"],
            ],
        )
        with patch(
            "scripts.data_collector.pit.collector.bs.query_performance_express_report",
            return_value=response,
        ):
            result = PitCollector.get_performance_express_report_df("sh.600519", "2019-07-01", "2019-07-31")

        self.assertEqual(len(result), 2)
        self.assertTrue(math.isnan(result.iloc[0]["value"]))
        self.assertEqual(result.iloc[1]["date"], "2019-07-14")
        self.assertEqual(result.iloc[1]["value"], 0.0)

    def test_forecast_double_zero_bounds_become_nan(self):
        """业绩预告上下限双零应标记为百分比不适用。"""
        fields = [
            "profitForcastExpPubDate",
            "profitForcastExpStatDate",
            "profitForcastChgPctUp",
            "profitForcastChgPctDwn",
        ]
        response = FakeBaostockResponse(
            fields,
            [
                ["2007-07-05", "2007-06-30", "0", "0"],
                ["2007-07-06", "2007-06-30", "20", "-10"],
            ],
        )
        with patch("scripts.data_collector.pit.collector.bs.query_forecast_report", return_value=response):
            result = PitCollector.get_forecast_report_df("sz.000725", "2007-07-01", "2007-07-31")

        self.assertEqual(len(result), 2)
        self.assertTrue(math.isnan(result.iloc[0]["value"]))
        self.assertAlmostEqual(result.iloc[1]["value"], 0.05)


if __name__ == "__main__":
    unittest.main()
