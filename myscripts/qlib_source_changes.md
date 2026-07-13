# Qlib 源码修改记录（2026-07-11）

记录日期：2026-07-11

## 1. 修改概览

今天主要解决了 Qlib 在新版 MLflow 下的实验存储兼容问题，以及 Data Handler、DDG-DA、TRA、HighFreq 等模块的缓存目录分散问题。

修改后的项目级目录约定如下：

```text
C:\Users\wangyc\wqlib\
├── mlruns\
│   ├── mlflow.db
│   └── artifacts\
│       └── <experiment_name>\
│           └── <run_id>\
│               └── artifacts\
└── datacache\
    ├── handler_cache\
    ├── DDG-DA\
    ├── TRA\
    └── highfreq\
```

截至本文档生成时：

- 当前分支：`main`
- 相对 `origin/main`：领先 4 个提交
- 源码工作树：干净，无未提交源码修改
- 本文档是新增文件，存放于 `myscripts/`

## 2. MLflow 实验存储改为 SQLite

### 2.1 问题

Qlib 原来的默认配置使用本地 FileStore：

```text
file:<运行目录>/mlruns
```

当前环境安装的是 MLflow 3.14.0。新版 MLflow 已不再允许使用旧 FileStore 创建新的实验，因此 Qlib 启动实验时会失败。

### 2.2 修改方案

将默认 tracking store 改为项目级 SQLite 数据库：

```text
sqlite:///C:/Users/wangyc/wqlib/mlruns/mlflow.db
```

将实验 artifacts 根目录统一设为：

```text
file:///C:/Users/wangyc/wqlib/mlruns/artifacts
```

实际 artifact 目录保留 experiment 和 run 两层标识：

```text
mlruns/artifacts/<编码后的实验名称>/<run_id>/artifacts/
```

其中 `<run_id>` 上层确实存在实验层。这里使用经过 URL 编码的实验名称，而不是数据库中的数字 experiment ID，便于直接从文件夹名称识别实验。

### 2.3 涉及文件

- `qlib/config.py`
  - 增加默认 MLflow SQLite URI 和 artifact URI 的生成逻辑。
  - 在 `MLflowSettings` 中增加 `artifact_root`。
  - 将 tracking URI 和 artifact 根目录传入实验管理器。
- `qlib/workflow/expm.py`
  - `MLflowExpManager` 支持接收 `artifact_root`。
  - 创建实验时显式设置 `artifact_location`。
  - 对实验名称进行 URL 编码后用于 artifact 子目录。
- `qlib/cli/run.py`
  - `qrun` 不再强制使用旧的本地 FileStore。
  - 显式传入 `uri_folder` 时，也会生成对应的 SQLite 数据库和 artifacts 目录。
- `docs/component/recorder.rst`
- `docs/start/initialization.rst`
  - 更新默认实验存储方式的说明。
- `tests/dependency_tests/test_mlflow.py`
- `tests/test_workflow.py`
  - 增加 SQLite tracking store 和 artifact 路径相关测试。

### 2.4 结果

- 实验元数据写入 `mlruns/mlflow.db`。
- `params.pkl`、`pred.pkl`、`label.pkl` 等 Recorder artifacts 写入 `mlruns/artifacts/` 下对应的实验和 run 目录。
- `mlruns/` 已被 `.gitignore` 忽略，不进入 Git。

## 3. 统一项目级 Data Cache

### 3.1 问题

原实现中的 Handler 缓存和部分示例模型中间文件依赖运行目录或示例目录。以不同工作目录启动程序时，会产生多份缓存，路径也不稳定。

### 3.2 新的统一配置

在 `qlib/config.py` 中增加项目级默认缓存根目录：

```text
C:\Users\wangyc\wqlib\datacache
```

也可通过环境变量覆盖：

```text
QLIB_DATACACHE_ROOT
```

Qlib 默认配置中新增：

```python
C["datacache_path"]
```

`.gitignore` 已增加 `datacache/`，缓存文件不会进入 Git。

### 3.3 公共路径函数

`qlib/config.py` 中整理为两个职责不同的函数：

- `get_datacache_dir(directory_name, root_path=...)`
  - 创建并返回 `datacache` 下的一级公共目录。
  - 用于 `handler_cache`、`DDG-DA`、`highfreq` 等目录。
- `get_model_cache_path(model_name, configured_path, root_path=...)`
  - 将模型配置中的相对输出路径解析到 `datacache/<model_name>/`。
  - 保留用户给出的绝对路径。
  - 兼容历史配置里的 `output/...` 和 `outputs/...` 前缀，避免生成多余的 `output` 层。
  - 拒绝含 `..`、可能逃逸出缓存根目录的相对路径。

此前重复创建模型目录和 Handler 目录的 `get_model_cache_dirs` 已删除。

## 4. Handler Cache 修改

### 4.1 默认位置

所有通过 `replace_task_handler_with_cache` 生成的 Handler 缓存，默认写入：

```text
datacache/handler_cache/
```

涉及文件：

- `qlib/workflow/task/utils.py`
  - 默认缓存目录改为 `C["datacache_path"]/handler_cache`。
  - 路径统一解析为绝对路径，并转换为标准 file URI。
- `qlib/contrib/rolling/base.py`
  - Rolling 流程不再用配置文件所在目录作为 Handler 缓存目录。
  - 用户显式传入的 Handler 路径仍然保留并规范化。

### 4.2 Rolling Process Data

手工生成的 `pre_handler.pkl` 改为：

```text
datacache/handler_cache/rolling_process_data.pre_handler.pkl
```

涉及文件：

- `examples/rolling_process_data/workflow.py`
- `examples/rolling_process_data/README.md`

## 5. DDG-DA 缓存修改

DDG-DA 的缓存按类型分开存放：

```text
datacache/
├── handler_cache/
│   ├── DDG-DA.handler_proxy.pkl
│   └── <DDG-DA 生成的其他 Handler 缓存>
└── DDG-DA/
    ├── fea_label_df.pkl
    ├── internal_data_s20.pkl
    └── tasks_s20.pkl
```

也就是说，DDG-DA 的 Handler 与其他模型共享 `handler_cache` 一级目录；DDG-DA 自己的特征、内部数据和任务中间文件进入 `datacache/DDG-DA/`。

涉及文件：

- `qlib/contrib/rolling/ddgda.py`
  - DDG-DA 中间文件改用项目级缓存目录。
  - proxy Handler 使用唯一名称 `DDG-DA.handler_proxy.pkl`。
  - 哈希命名的 Handler 也进入共享 `handler_cache`。
- `examples/benchmarks_dynamic/DDG-DA/workflow.py`
  - 不再强制把示例目录作为 `working_dir`。
- `examples/benchmarks_dynamic/DDG-DA/vis_data.py`
  - 从 `datacache/DDG-DA/` 读取可视化所需缓存。
- `examples/benchmarks_dynamic/DDG-DA/Makefile`
  - `clean` 删除 `datacache/DDG-DA` 和 DDG-DA 的 proxy Handler。
  - 不会清空整个共享 `handler_cache`，避免误删其他模型的缓存。
- `examples/benchmarks_dynamic/DDG-DA/README.md`
  - 更新缓存路径和清理说明。

DDG-DA 训练产生的 MLflow 模型 artifacts 仍属于实验结果，继续保存在 `mlruns/`，不会放入 `datacache/`。

## 6. TRA 缓存修改

TRA 配置中的相对 `logdir` 现在统一解析到：

```text
datacache/TRA/
```

例如历史配置：

```text
output/Alpha158
```

实际会映射为：

```text
datacache/TRA/Alpha158
```

其中包括 TRA 的 `model.bin`、训练日志以及 train、valid、test 阶段生成的预测、概率和 P 矩阵等 PKL 文件。用户显式指定的绝对 `logdir` 不会被改写。

涉及文件：

- `qlib/contrib/model/pytorch_tra.py`
- `examples/benchmarks/TRA/src/model.py`
- `examples/benchmarks/TRA/README.md`

## 7. HighFreq 缓存修改

HighFreq 示例生成的数据集缓存改为：

```text
datacache/highfreq/dataset.pkl
datacache/highfreq/dataset_backtest.pkl
```

保存和读取路径已同步修改。

涉及文件：

- `examples/highfreq/workflow.py`
- `examples/highfreq/README.md`

## 8. 未统一迁移的文件

以下内容本次没有迁入 `datacache`：

- Portfolio 风险数据：属于业务输入数据，不是可随时重建的运行缓存。
- RL orders：属于训练输入或业务数据，不按模型中间缓存处理。
- 普通 PyTorch 模型默认写入 `~/tmp` 的 checkpoint：本次只处理已确认会在 Qlib 示例目录产生专用 PKL 或模型缓存的路径。
- MLflow 中的 params、pred、label 和模型 artifacts：属于实验记录，继续保存在 `mlruns/`。

## 9. 测试与验证

本次已完成以下验证：

- MLflow SQLite 和 artifact 路径的 3 个针对性测试通过。
- `R.start -> save_objects -> load_object -> get_local_dir` 实际流程通过。
- Handler、Rolling、DDG-DA、TRA 路径相关的 4 个测试通过，无跳过。
- 修改模块和示例的 `compileall` 通过。
- `git diff --check` 通过。
- `pip check` 通过，未发现损坏的依赖关系。
- 已确认 Rolling Process Data 的 Handler 路径为：

  ```text
  C:\Users\wangyc\wqlib\datacache\handler_cache\rolling_process_data.pre_handler.pkl
  ```

- 已确认 HighFreq 的两个数据集路径位于 `datacache/highfreq/`。

### 尚未执行的完整验证

当前虚拟环境尚未安装 PyTorch，因此没有执行完整的 DDG-DA 和 TRA 模型训练。现有验证覆盖了路径解析、配置传递、模块编译和不依赖完整训练的单元测试。后续安装与当前 Python 环境兼容的 PyTorch 后，应分别运行一次 DDG-DA 和 TRA 的最小训练流程，检查缓存文件与 MLflow artifacts 的最终落盘位置。

Rolling 模块导入测试期间出现过 Gym 的维护状态警告，但没有影响测试通过，与本次缓存路径修改无直接关系。

## 10. 项目规则修改

新增 `.agents/AGENTS.md`，约定：

- 用户要求新建或起草的脚本，默认存入项目根目录 `myscripts/`。
- 用户明确指定其他路径时，以用户指定路径为准。
- 修改已有脚本时，保持原文件位置。

## 11. 今日提交记录

| 提交 | 时间 | 内容 |
| --- | --- | --- |
| `a1e2bdfe` | 17:32 | 修改实验结果输出目录，切换到 MLflow SQLite 和项目级 artifacts |
| `375964f7` | 17:36 | 新建项目规则，约定用户脚本存入 `myscripts` |
| `7cd881b4` | 17:57 | 解决 Handler cache 存储位置问题，迁移 DDG-DA 缓存 |
| `647d242e` | 18:22 | 进一步统一 TRA、HighFreq、Rolling Process Data 等模型缓存目录 |
| `bf641d1b` | 18:35 | 定向屏蔽 Qlib 导入链触发的 Gym 停止维护公告 |
| `1cb5e8eb` | 18:37 | 将源码修改记录改为固定文件名，日期改在文档内部维护 |

## 12. 主要源码文件汇总

核心源码：

- `qlib/config.py`
- `qlib/cli/run.py`
- `qlib/workflow/expm.py`
- `qlib/workflow/task/utils.py`
- `qlib/contrib/rolling/base.py`
- `qlib/contrib/rolling/ddgda.py`
- `qlib/contrib/model/pytorch_tra.py`

示例与说明：

- `examples/benchmarks_dynamic/DDG-DA/`
- `examples/benchmarks/TRA/`
- `examples/highfreq/`
- `examples/rolling_process_data/`
- `docs/component/recorder.rst`
- `docs/start/initialization.rst`

测试：

- `tests/test_workflow.py`
- `tests/dependency_tests/test_mlflow.py`
- `tests/test_gym_notice.py`

## 13. Gym 维护公告定向屏蔽

旧版 Gym 的停止维护信息是在导入时直接写入 `stderr`，并非标准 Python warning。为避免 Qlib 的非 RL 模型导入时反复显示该公告，在 `qlib/__init__.py` 中增加定向处理：

- 仅移除当前 Gym 版本在 `gym-notices` 中对应的维护公告。
- 不重定向全局 `stderr`，保留其他警告与真实异常。
- 用户在 Qlib 之前直接导入 Gym 时，公告仍会正常出现。
- `tests/test_gym_notice.py` 使用独立 Python 子进程验证屏蔽行为。

## 14. CatBoost 日志统一存储

项目新增独立的运行日志根目录：

```text
C:\Users\wangyc\wqlib\logs\
```

`qlib/config.py` 新增 `DEFAULT_LOG_ROOT`、`C["logs_path"]` 和 `get_model_log_path()`。默认路径可通过环境变量 `QLIB_LOG_ROOT` 覆盖。

CatBoost 不再把 `catboost_info` 写入启动目录，默认使用独立运行目录：

```text
logs/CatBoost/<YYYYMMDD_HHMMSS_微秒_随机后缀>/
```

路径规则：

- 每个 `CatBoostModel` 实例使用独立日志目录，避免不同训练互相覆盖。
- 用户显式传入 `train_dir` 时保持用户配置。
- 用户设置 `allow_writing_files=False` 时不创建默认日志目录。
- `logs/` 已加入 `.gitignore`。

验证结果：

- 3 个日志路径单元测试通过。
- CatBoost RTX 5080 GPU 最小训练通过。
- 训练指标、耗时和 TensorBoard events 均写入新的 `logs/CatBoost/` 路径。

## 15. MLflow 测试临时目录清理

测试不再在项目根目录使用 `.mlruns_tmp`：

- `tests/dependency_tests/test_mlflow.py` 为每个测试创建独立的系统 `TemporaryDirectory`。
- `tests/test_workflow.py` 在 `setUpClass` 中创建系统临时目录，避免模块导入阶段产生文件。
- 测试结束时先释放 Windows SQLite engine 和 MLflow store cache，再清理临时目录。
- 项目根目录下遗留的空 `.mlruns_tmp` 可以永久删除，后续导入测试模块也不会重新创建。

## 16. PIT 小样本下载与上游占位零复现

PIT collector 增加显式股票参数：

```text
--symbols 600519.ss,000725.sz
```

涉及修改：

- `scripts/data_collector/pit/collector.py`
  - 新增逗号分隔的 `symbols` 参数。
  - 指定股票时跳过东方财富全市场股票池请求，直接进入 Baostock 下载。
  - 支持六位代码自动补充 `.ss` 或 `.sz` 后缀，并校验代码格式。
  - 未指定 `symbols` 时保留原东方财富股票池和 `symbol_regex` 行为。

公共 `get_hs_stock_symbols()` 未修改，因为 Yahoo 全市场采集器仍在使用。

PIT 清洗采用两层规则：

- 采集层按 Baostock 来源字段识别占位零：
  - 业绩预告上下限同时为零时，视为“百分比不适用”，将该条百分比修订标记为 `NaN`。
  - 业绩快报的总资产、净资产和 `ROEWa` 同时为零时，将 `ROEWa=0` 标记为 `NaN`；资产数据非零时，合法的 `ROEWa=0` 仍保留。
  - 采集层只负责识别来源语义，不直接删除记录。
- `scripts/dump_pit.py` 在写入修订链前跳过真正的 `NaN` 数值，避免空修订覆盖上一有效值；合法数值 `0` 仍正常写入。

诊断阶段曾删除以下两只股票的 raw/normalized 小样本后重新下载（仅为本次复现，不作为后续运行步骤）：

```text
~/.qlib/stock_data/source/pit/{sh600519,sz000725}.csv
~/.qlib/stock_data/source/pit_normalized/{sh600519,sz000725}.csv
```

保留现有 `~/.qlib/qlib_data/cn_data/financial/` 作为修正结果对照。本次只重新下载 raw CSV，没有执行 normalize 或 dump。

后续修改与测试不得再删除或重新下载这批数据；除非用户另行明确要求，只做只读核验。

Baostock 重新下载结果：

- 两只股票均下载成功，错误股票数为 0。
- `sh600519` 返回 1 条零值，5 天后同字段、同报告期更新为非零。
- `sz000725` 返回 22 条零值，其中 14 条随后被同字段、同报告期的非零值修订。
- 该结果确认占位零来自 Baostock 当前返回，不是旧 CSV 或 `dump_pit.py` 生成。

## 17. 统一记录实验模型与数据集元数据（2026-07-12）

为避免 notebook、Trainer 等不同运行入口遗漏关键实验配置，在
`qlib/workflow/record_temp.py` 的 `SignalRecord.generate()` 公共入口增加自动记录：

```text
qlib.model.class
qlib.model.module
qlib.dataset.instruments
qlib.dataset.test_start
qlib.dataset.test_end
```

数据直接取自本次预测实际使用的 `model`、`dataset.handler.instruments` 和
`dataset.segments["test"]`，并写入当前 recorder 的 MLflow params。无法提取时输出明确 warning，
不使用默认股票池或根据源码猜测。

新增 `myscripts/summ.py`，按 run 打印 MLflow 实验摘要：

- 支持输入 N，仅打印按开始时间倒序排列的最新 N 条 run。
- 打印模型、instrument、配置测试区间、Rank IC、Rank ICIR，以及扣费后超额收益指标。
- 优先读取新的 `qlib.*` 统一字段。
- 兼容标准 Trainer 已有的 `model.class`、`dataset.kwargs.handler.kwargs.instruments` 和
  `dataset.kwargs.segments.test` 字段。
- 旧 run 缺少元数据时显示 `-`，不回填、不读取 `pred.pkl` 猜测配置。
- 每次执行同时刷新 `myscripts/exp_summaries.md`；Markdown 中每个 run 占一条记录，
  `experiment_name` 与 `run_id` 使用独立列。

验证结果：

- `SignalRecord` 元数据写入测试通过。
- tuple 与 slice 两种测试区间表示均通过。
- `summ.py` 新统一字段与标准 Trainer 旧字段兼容读取测试通过。

### 18. `myscripts` 项目级命令路径

- 在 `.vscode/settings.json` 和 `.vscode/wqlib.code-workspace` 中将
  `C:\Users\wangyc\wqlib\myscripts` 加入 VS Code 集成终端的 `PATH`。
- 将命令入口安装到 `C:\Users\wangyc\.venvs\qlib\Scripts\summ.cmd`，
  固定使用 qlib 虚拟环境执行项目内的 `myscripts/summ.py`。
- 项目内不再保留重复的 `myscripts/summ.cmd`；激活 qlib 虚拟环境后，
  无需依赖 VS Code 的项目 PATH 设置。
- 新建 VS Code 终端后，可直接运行 `summ` 或 `summ <N>`。

## 19. GRU Alpha158 DataLoader worker 调整（2026-07-12）

- `examples/benchmarks/GRU/workflow_config_gru_Alpha158.yaml` 将 `n_jobs` 从 20 调整为 4，
  降低 Windows 创建大量 DataLoader worker 的开销。
- 不修改 `persistent_workers`，`qlib/contrib/model/pytorch_gru_ts.py` 保持原有实现。

## 20. qrun 可配置 Data Handler 缓存（2026-07-12）

- `qlib/cli/run.py` 支持读取 YAML 顶层的 `handler_cache.enabled`。
- 启用时，qrun 在调用 `task_train()` 前执行 `replace_task_handler_with_cache()`；未配置时保持
  原有不缓存行为。
- `replace_task_handler_with_cache()` 明确记录 cache hit、cache miss 或已有 Handler URI 的跳过状态。
- `examples/benchmarks/GRU/workflow_config_gru_Alpha158.yaml` 已设置：

```yaml
handler_cache:
    enabled: true
```

- 原始 YAML 继续保存为 MLflow 的 `config` artifact，实际使用缓存 URI 的 task 保存为 `task` artifact。
- 新增 qrun 缓存开关测试，覆盖启用、默认关闭和非法类型三种情况。

## 21. GRU 可关闭训练集逐轮评估（2026-07-12）

- `qlib/contrib/model/pytorch_gru_ts.py` 新增 `eval_train` 布尔参数，默认 `True`，保持其他配置
  的原有行为。
- `eval_train=False` 时，每个 epoch 只评估验证集；验证指标、early stop 和最佳权重选择逻辑不变，
  `evals_result["train"]` 保持空列表。
- `examples/benchmarks/GRU/workflow_config_gru_Alpha158.yaml` 已设置 `eval_train: false`，跳过每轮
  第二次完整遍历训练集。
- 新增测试，覆盖关闭、默认开启和非法参数类型三种情况。

## 22. GRU Alpha158 复现与 batch 调整（2026-07-12）

- `examples/benchmarks/GRU/workflow_config_gru_Alpha158.yaml` 设置 `seed: 42`，固定模型初始化及
  训练随机状态。
- 将 `batch_size` 从 800 调整为 2048，减少每个 epoch 的 batch 数量，提高 RTX 5080 的利用率。

## 23. Windows Handler cache file URI 修复（2026-07-12）

- `qlib/utils/mod.py` 使用 `url2pathname()` 将 `file:///C:/...` 转换为 Windows 本地路径，避免
  原逻辑生成非法的 `\\C:\\...` 路径。
- 保留 `file://data/...` 相对路径的历史解析行为。
- 新增带空格文件名的绝对 file URI 加载测试，覆盖 Windows 盘符和 POSIX 路径。
- 本项目的 Handler cache 均由用户自己的 Qlib 代码生成，因此安全反序列化器将 `qlib.*` 加入
  受信任模块前缀，允许加载 Alpha158、Processor 和 QlibDataLoader 等内部对象。
- 保持原有 `Alpha158.<hash>.pkl` 缓存格式，现有 `Alpha158.ebe4fea9c7.pkl` 可以直接复用。

## 24. Rolling 最终拼接 run 元数据（2026-07-12）

- `qlib/contrib/rolling/base.py` 在最终拼接 recorder 中记录模型类型、模型模块、股票池和完整测试区间。
- 模型与股票池优先读取实际滚动子 run 的 `SignalRecord` 元数据；测试区间读取滚动前的有效任务配置，
  并兼容 `train_start`、`test_end` 和 `task_ext_conf` 覆盖。
- 最终 run 使用统一的 `qlib.model.*` 和 `qlib.dataset.*` 参数，`myscripts/summ.py` 无需解析模型 artifact
  即可显示 `model`、`instrument` 和 `test_period`。
- 已为现有 `rolling_LGBM_h20_s120` 最终 run 补写元数据，并新增最终拼接元数据测试。

## 25. 实验摘要默认隐藏 Rolling 子 run（2026-07-12）

- `myscripts/summ.py` 默认排除 `rolling_models_*` 中间实验，以及被最终 run 的 `exp_name` 参数
  引用的自定义 Rolling 子实验。
- 新增 `--include-child-runs` 开关；只有用户明确指定时才打印中间子 run。
- 最终 Rolling run 的 `model` 展示增加 `Roll ` 前缀，例如 `Roll LGBModel`；数据库中的真实
  `qlib.model.class` 仍保持 `LGBModel`。
- `summ <N>` 仍先选择最新 N 条符合过滤条件的记录，再按 experiment_id 和开始时间正序展示。

## 26. Rolling 支持选择训练窗口类型（2026-07-12）

- `qlib/contrib/rolling/base.py` 为 `Rolling` 新增 `rtype` 参数，默认值仍为 `expanding`，保持原有行为。
- 可将 `rtype` 设置为 `sliding`，使训练集保持初始窗口长度并随滚动步骤整体向前移动。
- `Rolling.get_task_list()` 将窗口类型传入 `RollingGen`；非法值会立即抛出 `ValueError`，不会静默回退。
- `examples/benchmarks_dynamic/baseline/rolling_benchmark.py` 已通过 `**kwargs` 透传命令行参数，无需重复修改。
- 新增测试，覆盖默认 expanding、显式 sliding 及非法类型三种情况。

## 27. RollingDataHandler 统一管理原始 Handler cache（2026-07-12）

- `examples/rolling_process_data/rolling_handler.py` 新增 `handler_config`、`cache_raw_handler` 和
  `cache_dir` 参数，通过唯一的 `handler_config` 入口接收原始 Handler 配置、实例或已有缓存 URI。
- 字典形式的原始 Handler 复用 `replace_task_handler_with_cache()`，按配置 hash 写入
  `datacache/handler_cache`；已有 URI 或 Handler 实例不会重复缓存。
- 原始 Handler 的 `infer_processors` 和 `learn_processors` 由内部统一设置为空，外层
  RollingDataHandler 的 Processor 仍按当前窗口拟合，不会写入共享缓存。
- `examples/rolling_process_data/workflow.py` 删除固定文件名的手工 dump/load 流程，直接将 Alpha158
  参数交给新增的 `rolling_handler.Alpha158` 统一入口；不同运行可以复用同一 hash cache。
- `rolling_handler.Alpha158` 内部构造无 Processor 的原生 Alpha158，并将外层 Processor 交给
  RollingDataHandler；`label` 参数继续兼容 Rolling 的 horizon 动态覆盖。
- 原始 cache 使用固定的 `start_time/end_time`；外层使用 `window_start_time/window_end_time`
  表示当前滚动任务的读取范围，避免缓存范围与窗口范围混用。
- 更新示例 README，并新增测试覆盖缓存路由、URI 复用、Alpha158 包装和 workflow 配置。
- 真实数据端到端测试通过：首次运行生成 `Alpha158.698fb38724.pkl`（约 445 MB），5 个滚动窗口均完成；
  第二次运行命中同一缓存，并再次逐窗口执行 RobustZScoreNorm、DropnaLabel 和 CSZScoreNorm。
- 时间参数重命名后再次完成 5 个窗口的真实数据测试，仍命中同一 Alpha158 cache，未重新生成原始特征。

## 28. 用户滚动训练脚本 rolling_method（2026-07-13）

- 新增 `myscripts/rolling_method.py`，保留 benchmark 的 Rolling 训练、拼接和回测流程，不修改原
  `examples/benchmarks_dynamic/baseline/rolling_benchmark.py`。
- 外层 `rolling_handler.Alpha158` 始终保留为配置字典，不缓存已经拟合的 Processor；原始 Alpha158
  cache 由 Handler 内部按配置 hash 创建或复用。
- YAML 只配置固定全样本范围 `start_time/end_time`；每个 RollingGen 子任务根据自身 segments
  自动注入 `window_start_time/window_end_time` 和 `fit_start_time/fit_end_time`。
- 任务拆分后会恢复 YAML 中的固定全样本范围，并将窗口范围限制在全样本内，避免
  `RollingGen.handler_mod` 为末尾子任务扩展原始 Handler cache 的 `end_time`。
- 时间同步直接复用标准 `train` 和 `test` segments，不再重复实现通用 segment 范围推导。
- 新增 `myscripts/rolling_method_lgbm_Alpha158.yaml`，配置逐窗口执行 ProcessInf、RobustZScoreNorm、
  Fillna、DropnaLabel 和 CSZScoreNorm。
- qlib 虚拟环境通过 `site-packages/wqlib-project-root.pth` 固定加入项目根目录；所有 `myscripts`
  脚本无需 `_bootstrap.py` 或 `PYTHONPATH` 即可导入 `examples` 等本地模块。
- 减少滚动实验的非异常 warning：已有 Qlib 数据不再重复调用下载器；滚动 Handler 可记录 instrument
  元数据；Git 换行符提示写入代码快照 artifact 而非终端；全 NaN 均值显式返回 NaN。
- 新增测试覆盖 horizon 标签、外层缓存跳过、任务时间同步和默认 YAML Processor 配置。
- 使用真实交易日历预检 `horizon=10、step=240、rtype=sliding`，成功生成 4 个任务，各任务的
  `fit_start_time/fit_end_time` 均与滑动 train segment 一致；预检未启动模型训练。

## 29. Rolling dataset artifact 不再复制原始 Alpha158（2026-07-13）

- `examples/rolling_process_data/rolling_handler.py` 新增 `RollingDataLoader`，运行时延迟读取公共
  Handler cache，序列化时只保留 cache URI 和读取参数。
- 已加载的原始 Handler 保存在私有 `_handler` 属性；Trainer 设置 `dump_all=False` 后不会把
  Alpha158 DataFrame 写入 MLflow dataset artifact。
- 外层 RollingDataHandler、Processor 拟合状态和 Dataset segments 继续保留，可用于在线推理和
  实验审计；公共 cache 缺失时明确抛出包含 URI 的 `FileNotFoundError`，不做静默回退。
- 新增小型数据序列化往返测试，确认 artifact 中不存在 pandas DataFrame，恢复后仍可从 cache
  重新生成相同数据。
- 使用真实 `Alpha158.6531823050.pkl` 验证：单个 dataset artifact 从约 560.35 MiB 降至
  4,814 bytes（4.701 KiB）；反序列化后生成的 LightGBM 预测与原 `pred.pkl` 逐元素完全一致。

## 30. RollingDataHandler 支持任意底层特征 Handler（2026-07-13）

- 删除 `rolling_handler.Alpha158` 专用包装类，统一使用 `RollingDataHandler` 作为 DatasetH 的外层
  Handler；Alpha158、Alpha360 或自定义特征集合由嵌套的 `handler_config` 选择。
- `start_time/end_time/instruments/label/freq` 保留在外层配置，兼容 Rolling 的 horizon 覆盖、
  固定全样本范围和 run 元数据；初始化时再统一合并到底层 Handler kwargs。
- `myscripts/rolling_method_lgbm_Alpha158.yaml` 和 `examples/rolling_process_data/workflow.py` 已迁移到
  通用配置格式；Alpha158 cache hash 仍为 `6531823050`，继续复用现有公共 cache。
- 自定义 DataHandlerLP 保留自身类型写入公共 cache；后续增加具体 `AlphaXXX` 时，将其精确类加入
  安全反序列化白名单，不对自定义 Handler 做转换或使用特殊 cache 文件名。
- 新增 Alpha360 参数合并、输入配置不变性和通用 workflow 配置测试。
