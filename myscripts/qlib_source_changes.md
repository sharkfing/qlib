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
