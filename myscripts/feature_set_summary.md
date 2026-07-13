# 行情特征集合并与计算说明

## 1. 合并结果

本文将两组 SQL 风格的行情特征按经济含义和计算方法合并，并仅删除“输入字段、窗口和公式完全一致”的特征。

| 项目 | 数量 |
|---|---:|
| 第一组特征 | 91 |
| 第二组特征 | 46 |
| 完全重复并删除 | 23 |
| 合并后保留 | **114** |

合并后的114个特征分布如下：

| 类型 | 数量 |
|---|---:|
| 滞后行情与价格比值 | 28 |
| 5日均值及相对均值 | 11 |
| 5日最大值 | 7 |
| 5日最小值 | 7 |
| 5日标准差 | 7 |
| 5日滚动排名 | 7 |
| 5日线性衰减 | 7 |
| 5日时序相关系数 | 21 |
| 当日截面排名 | 19 |
| 合计 | **114** |

## 2. 计算约定

为了便于解释，下文采用以下记号：

- `x_t`：当前股票在当前交易日的字段值。
- `m_LAG(x, n)`：同一股票向前第 `n` 个观测值，即 `x_(t-n)`。
- `m_AVG(x, 5)`：同一股票最近5个观测值的移动平均。
- `m_MAX(x, 5)` / `m_MIN(x, 5)`：同一股票最近5个观测值的最大值和最小值。
- `m_STDDEV(x, 5)`：同一股票最近5个观测值的标准差。总体标准差还是样本标准差取决于计算引擎的具体定义。
- `m_rolling_rank(x, 5)`：当前值在该股票最近5个观测值中的时序排名。
- `m_decay_linear(x, 5)`：对最近5个观测值按时间进行线性加权。通常越接近当前日期权重越高，但具体权重方向应以计算引擎实现为准。
- `m_CORR(x, y, 5)`：同一股票最近5个观测值中 `x` 与 `y` 的时序 Pearson 相关系数。
- `c_pct_rank(x)`：同一交易日所有入选股票之间的截面百分位排名。
- `change_ratio + 1`：将收益率转换为增长倍数；这一解释要求 `change_ratio` 使用小数形式，例如2%记为0.02。

所有5日时序运算原则上使用 `[t-4, t]` 五个观测值。停牌、缺失值和并列排名的处理方式由实际计算引擎决定。

## 3. 滞后行情与价格比值（28个）

对 `n = 1, 2, 3, 4`，每个滞后期生成7个特征：

| 统一公式 | 别名 | 计算含义 |
|---|---|---|
| `m_LAG(close, n)` | `close_n` | 第n个滞后期的收盘价 |
| `m_LAG(open, n)` | `open_n` | 第n个滞后期的开盘价 |
| `m_LAG(high, n)` | `high_n` | 第n个滞后期的最高价 |
| `m_LAG(low, n)` | `low_n` | 第n个滞后期的最低价 |
| `m_LAG(amount, n)` | `amount_n` | 第n个滞后期的成交额 |
| `m_LAG(turn * 100, n)` | `turn_n` | 第n个滞后期的换手率百分数 |
| `close / m_LAG(close, n + 1)` | `return_n` | 当前收盘价相对第 `n+1` 个滞后期收盘价的比值 |

按照原始公式：

```text
return_1 = close_t / close_(t-2)
return_2 = close_t / close_(t-3)
return_3 = close_t / close_(t-4)
return_4 = close_t / close_(t-5)
```

这些字段没有减1，因此是“价格比值”，不是通常意义上的收益率；其跨度分别为2、3、4、5个观测期，也不是别名看起来所暗示的1、2、3、4期收益率。

如果原意是计算过去 `n` 期收益率，更常见的公式是：

```sql
close / m_LAG(close, n) - 1
```

本文为忠实合并原始特征，不自动修改这4个公式。

## 4. 5日均值及相对均值（11个）

### 4.1 原始5日均值（7个）

| 公式 | 别名 | 计算含义 |
|---|---|---|
| `m_AVG(close, 5)` | `ma_close_5` | 5日平均收盘价 |
| `m_AVG(low, 5)` | `ma_low_5` | 5日平均最低价 |
| `m_AVG(open, 5)` | `ma_open_5` | 5日平均开盘价 |
| `m_AVG(high, 5)` | `ma_high_5` | 5日平均最高价 |
| `m_AVG(turn * 100, 5)` | `ma_turn_5` | 5日平均换手率百分数 |
| `m_AVG(amount, 5)` | `ma_amount_5` | 5日平均成交额 |
| `m_AVG(change_ratio + 1, 5)` | `ma_cr_5` | 5日平均增长倍数 |

### 4.2 相对当前值的5日均值（4个）

第二组中的4个均值与上述原始均值同名但公式不同，不能作为重复项删除。为避免别名冲突，统一增加 `_ratio_5` 后缀。

| 原始公式 | 统一别名 | 计算含义 |
|---|---|---|
| `m_AVG(close, 5) / close` | `ma_close_ratio_5` | 5日平均收盘价相对当前收盘价的倍数 |
| `m_AVG(turn * 100, 5) / turn` | `ma_turn_ratio_5` | 原公式下，5日平均换手率百分数相对当前原始换手率的倍数 |
| `m_AVG(amount, 5) / amount` | `ma_amount_ratio_5` | 5日平均成交额相对当前成交额的倍数 |
| `m_AVG(change_ratio + 1, 5) / (change_ratio + 1)` | `ma_cr_ratio_5` | 5日平均增长倍数相对当前增长倍数的比值 |

`ma_turn_ratio_5` 的分子使用 `turn * 100`，分母却使用 `turn`，因此结果会比通常的相对倍数放大100倍。如果目标是同单位比较，建议后续确认是否应改为以下任一等价形式：

```sql
m_AVG(turn * 100, 5) / (turn * 100)
m_AVG(turn, 5) / turn
```

本文仍保留用户给出的原公式，不做静默修正。

## 5. 5日最大值（7个）

| 公式 | 统一别名 | 计算含义 |
|---|---|---|
| `m_MAX(close, 5)` | `max_close_5` | 最近5日最高收盘价 |
| `m_MAX(low, 5)` | `max_low_5` | 最近5日最高的最低价 |
| `m_MAX(open, 5)` | `max_open_5` | 最近5日最高开盘价 |
| `m_MAX(high, 5)` | `max_high_5` | 最近5日最高价的最大值 |
| `m_MAX(turn * 100, 5)` | `max_turn_5` | 最近5日最大换手率百分数 |
| `m_MAX(amount, 5)` | `max_amount_5` | 最近5日最大成交额 |
| `m_MAX(change_ratio + 1, 5)` | `max_cr_5` | 最近5日最大增长倍数 |

原始别名中的 `mx_low_5`、`mx_open_5`、`mx_high_5` 和 `max_cr_r` 在本文统一为 `max_*_5`，仅统一命名，不改变公式。

## 6. 5日最小值（7个）

| 公式 | 别名 | 计算含义 |
|---|---|---|
| `m_MIN(close, 5)` | `min_close_5` | 最近5日最低收盘价 |
| `m_MIN(low, 5)` | `min_low_5` | 最近5日最低价的最小值 |
| `m_MIN(open, 5)` | `min_open_5` | 最近5日最低开盘价 |
| `m_MIN(high, 5)` | `min_high_5` | 最近5日最低的最高价 |
| `m_MIN(turn * 100, 5)` | `min_turn_5` | 最近5日最小换手率百分数 |
| `m_MIN(amount, 5)` | `min_amount_5` | 最近5日最小成交额 |
| `m_MIN(change_ratio + 1, 5)` | `min_cr_5` | 最近5日最小增长倍数 |

## 7. 5日标准差（7个）

第二组中的 `std_close_5`、`std_turn_5`、`std_amount_5` 和 `std_cr_5` 与第一组完全重复，因此只保留一次。

| 公式 | 别名 | 计算含义 |
|---|---|---|
| `m_STDDEV(close, 5)` | `std_close_5` | 5日收盘价波动 |
| `m_STDDEV(low, 5)` | `std_low_5` | 5日最低价波动 |
| `m_STDDEV(open, 5)` | `std_open_5` | 5日开盘价波动 |
| `m_STDDEV(high, 5)` | `std_high_5` | 5日最高价波动 |
| `m_STDDEV(turn * 100, 5)` | `std_turn_5` | 5日换手率百分数波动 |
| `m_STDDEV(amount, 5)` | `std_amount_5` | 5日成交额波动 |
| `m_STDDEV(change_ratio + 1, 5)` | `std_cr_5` | 5日增长倍数波动；加1不改变标准差 |

原始价格和成交额的标准差带有量纲，跨股票比较时会受价格水平和公司规模影响。

## 8. 5日滚动排名（7个）

第二组的7个滚动排名与第一组完全重复，因此只保留一次。

| 公式 | 别名 | 计算含义 |
|---|---|---|
| `m_rolling_rank(close, 5) / 5` | `rank_close_5` | 当前收盘价在自身最近5日中的相对排名 |
| `m_rolling_rank(low, 5) / 5` | `rank_low_5` | 当前最低价在自身最近5日中的相对排名 |
| `m_rolling_rank(open, 5) / 5` | `rank_open_5` | 当前开盘价在自身最近5日中的相对排名 |
| `m_rolling_rank(high, 5) / 5` | `rank_high_5` | 当前最高价在自身最近5日中的相对排名 |
| `m_rolling_rank(turn * 100, 5) / 5` | `rank_turn_5` | 当前换手率在自身最近5日中的相对排名 |
| `m_rolling_rank(amount, 5) / 5` | `rank_amount_5` | 当前成交额在自身最近5日中的相对排名 |
| `m_rolling_rank(change_ratio + 1, 5) / 5` | `rank_cr_5` | 当前收益表现在自身最近5日中的相对排名 |

除以5隐含假设 `m_rolling_rank` 返回1至5的名次。如果引擎已经返回0至1百分位，或者返回0至4的下标，则不能再次直接除以5，应先核对函数定义。

## 9. 5日线性衰减（7个）

| 公式 | 别名 | 计算含义 |
|---|---|---|
| `m_decay_linear(close, 5)` | `dl_close_5` | 5日线性加权收盘价 |
| `m_decay_linear(low, 5)` | `dl_low_5` | 5日线性加权最低价 |
| `m_decay_linear(open, 5)` | `dl_open_5` | 5日线性加权开盘价 |
| `m_decay_linear(high, 5)` | `dl_high_5` | 5日线性加权最高价 |
| `m_decay_linear(turn * 100, 5)` | `dl_turn_5` | 5日线性加权换手率百分数 |
| `m_decay_linear(amount, 5)` | `dl_amount_5` | 5日线性加权成交额 |
| `m_decay_linear(change_ratio + 1, 5)` | `dl_cr_5` | 5日线性加权增长倍数 |

如果采用常见的线性衰减定义，5个观测值的权重与 `1, 2, 3, 4, 5` 成比例，最近值权重最高；实际权重顺序仍需以引擎实现为准。

## 10. 5日时序相关系数（21个）

第二组中的12个相关系数均已包含在第一组中，因此只保留第一组完整的21个相关系数。

| 变量组 | 公式及别名 | 数量 |
|---|---|---:|
| 成交量与其他变量 | `m_CORR(volume, change_ratio + 1, 5)` → `corr_vcr`；`volume` 分别与 `high`、`low`、`close`、`open`、`turn * 100` 相关，别名为 `corr_vh`、`corr_vl`、`corr_vc`、`corr_vo`、`corr_vt` | 6 |
| 收益表现与其他变量 | `change_ratio + 1` 分别与 `high`、`low`、`close`、`open`、`turn` 相关，别名为 `corr_crh`、`corr_crl`、`corr_crc`、`corr_cro`、`corr_crt` | 5 |
| 最高价与其他变量 | `high` 分别与 `low`、`close`、`open`、`turn * 100` 相关，别名为 `corr_hl`、`corr_hc`、`corr_ho`、`corr_ht` | 4 |
| 最低价与其他变量 | `low` 分别与 `close`、`open`、`turn * 100` 相关，别名为 `corr_lc`、`corr_lo`、`corr_lt` | 3 |
| 收盘价与其他变量 | `close` 分别与 `open`、`turn * 100` 相关，别名为 `corr_co`、`corr_ct` | 2 |
| 开盘价与换手率 | `m_CORR(open, turn * 100, 5)` → `corr_ot` | 1 |

相关系数不受正比例缩放影响，因此在没有缺失值和数值误差的情况下，使用 `turn` 或 `turn * 100` 得到的相关系数相同。

## 11. 当日截面排名（19个）

截面排名不是原时序特征的完全重复。它在每个交易日对所有入选股票排序，表达的是“该股票当天相对其他股票的位置”。

### 11.1 当前基础字段（2个）

| 公式 | 别名 |
|---|---|
| `c_pct_rank(turn)` | `cross_turn` |
| `c_pct_rank(change_ratio + 1)` | `cross_change_ratio` |

### 11.2 相对5日均值（4个）

第二组中的 `ma_close_5`、`ma_turn_5`、`ma_amount_5` 和 `ma_cr_5` 指的是相对当前值的均值。为与原始均值区分，这里同步使用新的统一别名。

| 统一公式 | 统一别名 |
|---|---|
| `c_pct_rank(ma_close_ratio_5)` | `cross_ma_close_ratio_5` |
| `c_pct_rank(ma_turn_ratio_5)` | `cross_ma_turn_ratio_5` |
| `c_pct_rank(ma_amount_ratio_5)` | `cross_ma_amount_ratio_5` |
| `c_pct_rank(ma_cr_ratio_5)` | `cross_ma_cr_ratio_5` |

### 11.3 标准差（4个）

| 公式 | 别名 |
|---|---|
| `c_pct_rank(std_close_5)` | `cross_std_close_5` |
| `c_pct_rank(std_turn_5)` | `cross_std_turn_5` |
| `c_pct_rank(std_amount_5)` | `cross_std_amount_5` |
| `c_pct_rank(std_cr_5)` | `cross_std_cr_5` |

### 11.4 时序排名（4个）

| 公式 | 别名 |
|---|---|
| `c_pct_rank(rank_close_5)` | `cross_rank_close_5` |
| `c_pct_rank(rank_turn_5)` | `cross_rank_turn_5` |
| `c_pct_rank(rank_amount_5)` | `cross_rank_amount_5` |
| `c_pct_rank(rank_cr_5)` | `cross_rank_cr_5` |

### 11.5 相关系数（5个）

| 公式 | 别名 |
|---|---|
| `c_pct_rank(corr_vcr)` | `cross_corr_vcr` |
| `c_pct_rank(corr_vc)` | `cross_corr_vc` |
| `c_pct_rank(corr_vt)` | `cross_corr_vt` |
| `c_pct_rank(corr_crc)` | `cross_corr_crc` |
| `c_pct_rank(corr_crt)` | `cross_corr_crt` |

截面排名必须明确当天参与排名的股票池。股票池变化、停牌和缺失值都会改变排名，因此训练、验证和回测必须使用当时真实可投资股票池，不能使用未来成分股回填历史。

## 12. 删除的23个完全重复特征

### 12.1 标准差（4个）

```text
std_close_5
std_turn_5
std_amount_5
std_cr_5
```

### 12.2 5日滚动排名（7个）

```text
rank_close_5
rank_low_5
rank_open_5
rank_high_5
rank_turn_5
rank_amount_5
rank_cr_5
```

### 12.3 5日相关系数（12个）

```text
corr_vcr
corr_vc
corr_vt
corr_crc
corr_crt
corr_hl
corr_hc
corr_ho
corr_lc
corr_lo
corr_co
corr_ct
```

## 13. 使用前需要确认的问题

1. `return_1` 至 `return_4` 当前是2至5期价格比值，并且没有减1，应确认是否符合原意。
2. `ma_turn_ratio_5` 的分子和分母相差100倍，应确认换手率单位。
3. `change_ratio` 必须确认是小数还是百分数；只有小数收益率才通常使用 `change_ratio + 1` 表示增长倍数。
4. `m_STDDEV` 使用总体标准差还是样本标准差，需要核对计算引擎。
5. `m_rolling_rank` 的返回范围需要核对，否则 `/ 5` 可能产生错误缩放。
6. `m_decay_linear` 的权重方向需要核对，避免把最久远日期设为最高权重。
7. `c_pct_rank` 必须按交易日和当时真实股票池计算，不能跨日期排名，也不能使用未来股票池。
8. 原始价格、成交额及其标准差带有量纲；如果模型跨股票训练，应评估是否需要复权、标准化或改成相对比例。

