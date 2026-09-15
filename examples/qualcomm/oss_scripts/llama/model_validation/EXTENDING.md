# 添加模型、参数组合与新流程

## 配置约定

`schema_version: 1` 配置只包含 `description`（可选）、`variables`（可选）、`output_root` 和有序 `steps`。

变量使用 `${name}`，不是 shell 展开。内置 `${repo}`、`${validation}`、`${config_dir}`、`${output}`；矩阵任务还有 `${job}`。普通变量按声明顺序展开，只允许标量，不可覆盖内置名称。需要字面量 `$` 时写 `$$`。命令通过参数数组调用，不通过 shell，不会执行反引号、管道或 `$(...)`。

`job` 是保留名称，普通变量和 matrix 均不能重新定义它。

相对路径以仓库根目录为基准；需要配置文件旁的资源时显式写 `${config_dir}/...`。运行器始终以仓库根目录为工作目录。

每个步骤的字段：

| 字段 | 用法 |
|---|---|
| `id` | 唯一阶段名，由字母、数字、`_`、`-` 组成 |
| `tool` | `run.py list` 中已注册的工具名 |
| `args` | 原样传给该 CLI 的参数数组，工具自己的子命令也写在这里 |
| `inputs` | 需要哈希校验的输入文件/模型目录；不要漏掉模型、题集、tokenizer、驱动 |
| `outputs` | 必填，完成后必须存在的文件列表，全部位于 `output_root` 内；建议 manifest/summary 等稳定产物 |
| `depends_on` | 必须先完成的阶段 ID，只能引用列表中先出现的阶段；依赖一个矩阵阶段表示依赖其全部组合 |
| `matrix` | 参数名→候选值数组，展开笛卡尔积；没有矩阵就执行一次 |
| `append_when` | 根据矩阵值追加旗标的规则，结构见示例 |
| `resume_args` | 恢复中断任务时追加的参数，例如 `["--resume"]`；只能填写该工具真实支持的参数 |
| `resource` | 可选共享锁名，例如 `gpu:cuda:0`、`adb:SERIAL` |
| `description` | 可选说明，不执行 |

未知字段、变量拼写错误、重复阶段/输出、依赖缺失会报错。矩阵输出必须包含 `${job}` 或其他足以区分全部组合的字段；1.05 不会被四舍五入成 1.1。

模型目录哈希包含 `.safetensors/.json/.jinja/.pt2/.pte/.txt/.jsonl/.u64` 文件；特殊权重格式应通过 `inputs` 显式列出文件。大文件首次和恢复时的哈希需要时间，这是避免误用不同模型结果的成本。

## 示例：增加 PT2 对话消融

在 `variables` 增加 `pt2_model`，然后在 `inputs` 阶段之后添加：

```json
{
  "id": "pt2-ablation",
  "tool": "pt2",
  "depends_on": ["inputs"],
  "resource": "gpu:${gpu}",
  "matrix": {"think": ["on", "off"], "rp": [1.0, 1.1]},
  "args": [
    "--model", "${pt2_model}", "--tokenizer", "${tokenizer}",
    "--pt2-layout", "lincal-full-sequence", "--device", "${gpu}",
    "--cases", "${cases}", "--seq-len", "${seq_len}",
    "--repetition-penalty", "${rp}", "--repeat-last-n", "${repeat_last_n}",
    "--output", "${output}/generation/${job}"
  ],
  "append_when": [{"parameter": "think", "equals": "off", "args": ["--disable-thinking"]}],
  "inputs": ["${pt2_model}", "${tokenizer}", "${cases}"],
  "outputs": ["${output}/generation/${job}/summary.json"],
  "resume_args": ["--resume"]
}
```

先在 args 增加 `--ids I01` 小试并使用专门的输出根目录。该适配器不改量化图，不是旧 08/09/10 修补图的自动替代。

## 示例：给 PTE 增加双保护消融

复制 PTE 阶段，改 ID 与输出根目录，在 matrix 添加 `"guard": ["off", "on"]`。从固定 args 移除双保护开关，添加以下条件（原 think 条件保留）：

```json
[
  {"parameter": "guard", "equals": "off", "args": ["--no-repair-think-eos", "--no-enable-think-budget"]},
  {"parameter": "guard", "equals": "on", "args": ["--repair-think-eos", "--enable-think-budget", "--think-budget", "512"]}
]
```

单独比较 EOS 修复与预算截断，可以拆成两个矩阵变量，分别追加各自开关。当前 HF/PT2 CLI 没有这些保护参数，不能把 PTE 开关直接复制过去；若要三端对齐，需要先实现且验证等价的生成处理。

## 新增任务工具

1. 将实现放到 `calibration/`、`ppl/`、`benchmarks/`、`reporting/` 或新的合适子目录。脚本提供 `--help`，依赖尽量在执行时导入。
2. 在 `common/registry.py` 的 `TOOLS` 加一个 `Tool(ROOT / "子目录/script.py", "说明")`，即可 `run.py 工具名 ...` 和配置 `tool` 调用。不需要改调度循环。
3. 实现参数校验、非零失败退出、独立输出目录、实际参数/输入/产物哈希记录。不要静默修改问题、tokenizer、模型图或丢弃失败样本。
4. 为新的 CLI、结果格式、恢复逻辑添加小型测试，再在 README 写清输入、输出、支持架构和完整命令。

新增后端时先做固定 token 的最小等价测试；不要把 CUDA、QNN、KV/hybrid、AR 长度等不同执行语义藏在同一个未经验证的默认值中。

## 结果与恢复

工作流不自动发现并混入目录里的历史实验；报告只汇总显式提供的结果。标准 PPL 需要固定语料、词表与 token 序列、计分目标以及上下文策略；对话生成另外记录 think、RP、窗口、保护和温度。

`--resume` 校验配置、工具源码、声明的输入和完成标记，不跳过参数不同的任务。修改模型、代码、配置、复核后会影响已声明产物的哈希；需要新输出目录或独立报告目录，不能在完成标记后原地篡改文件再续跑。半个对话会话由底层工具归档并整组重跑，避免拼接不一致历史。

工作流按顺序执行且遇错停止。先独立解决失败原因；不要自动回退到另一个模型来填分数。`--with-dependencies` 才会补跑前置阶段，`--all` 才会启动配置中全部步骤，包括你自行添加的量化/导出步骤。

取消会清理本机子进程组；设备任务异常时不能据此断言远端 driver 已停止。恢复前确认手机空闲，再加 `--device-idle-confirmed`。这一确认不发送设备停止命令，也不会终止其他人的进程。
