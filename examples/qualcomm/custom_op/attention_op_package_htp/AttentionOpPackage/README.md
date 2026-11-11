# Valid-length causal Attention

实现位于 `src/ops/Attention.cpp`，online softmax 位于
`src/ops/attention/OnlineAttention.h`。这是计算算子；模型拆图、专家管理、
FastRPC 服务仍由现有导出器和 Runner 负责。

源码、算子定义和 C++ 测试由 `attention_op_package_htp/AttentionOpPackage` 独立维护，
与 `deltanet_op_package_htp/DeltaNetOpPackage` 平级。共用的包入口、编译规则与
HVX 工具在 `custom_op/htp_op_package`；Attention 不依赖 DeltaNet 的源码或架构头。

两个算子仍打入同一个 `libet_qnnop.so`。QNN 包标识保留 `DeltaNetOpPackage`，
这是兼容已有 PTE 的序列化名称，不代表 Attention 源码归属于 DeltaNet。

## 接口与参数

Python 注册：`examples/qualcomm/custom_op/attention.py`。
QNN 公共名称：`DeltaNetOpPackage::AttentionValid`，七输入、一输出。
`AttentionInitialize` 和 `AttentionExecute` 是包内部重写产生的执行阶段，
不是按有效 token 长度生成的多套图。

| 输入 | 类型 / 布局 |
| --- | --- |
| query | QUInt16 `[B,Hq,Q,D]` |
| key_descriptor | Int32 `[1,1,1,3]` |
| new_key | QUInt8 或 QUInt16 `[B,Hkv,D,Q]` |
| value_descriptor | Int32 `[1,1,1,3]` |
| new_value | 与 new_key 相同位宽 `[B,Hkv,Q,D]` |
| lengths | Int32 `[1,1,1,3]`，值为 `[valid,first,history]` |
| grids | Float32 `[1,1,1,13]` |

输出与 query 同形状，使用输出张量自己的 U16 量化参数。无效 query 行写量化零。
每个有效 query 行只看 `history + row + 1` 个因果可见位置。

`B/Hq/Hkv/Q/D` 来自张量形状，要求正数、`Hq % Hkv == 0`、`D <= 4096`，
query 元素总数不超过 INT32_MAX。支持 `1 <= valid <= Q`，不限定尾块长度；
`first >= 0` 且 `first + history <= C`。lengths 在 batch 内共用。
`C` 是历史缓存容量，不包含本次独立传入的 Q 个新位置。

缓存描述符是 `[fd,bytes_and_flags,epoch]`。低 30 位表示历史缓存字节数，
bit30 是 exact-append 标志。K 描述符的 bit31 表示 FD 中附带 packed KV
工作区，V 描述符不设置此位。K/V 使用不同 FD，逻辑字节数、exact-append
标志和 epoch 一致。历史缓存与新 K/V 使用相同位宽，按新 K/V 张量类型
确定元素字节数 E，根据 `B*Hkv*D*E` 推导历史容量 C，
K 的物理布局为 `[B,Hkv,D,C]`，V 为 `[B,Hkv,C,D]`，两者均需持久映射。
调用者必须提供覆盖声明字节数的实际映射。

设置 bit31 时，K FD 的实际分配还必须包含工作区：从逻辑字节数向上
按 4096 对齐的位置开始，预留 K 和 V 的 FP16 packed 容量；整数路径在同一
工作区存放 U8 packed 数据及 K 的整数和元数据。共享协议与
大小计算在 `attention_support/cache.h`；行数为
`min(16640, align_up(C+256,256))`，AR128/AR1 共用相同大小。
Runner 在加载阶段预分配、映射这部分内存，避免第一次 FA 执行时
逐层申请大块 DSP 堆内存。未设置 bit31 的调用者仍使用算子自己的分配。

只有历史前缀持续保留且量化参数兼容时才能设置 exact-append。reset、
历史重写、布局或缓存量化参数改变时需更新 epoch，防止复用旧 packed KV。
算子读取缓存，不替 Runner 将本次 K/V 回写历史缓存。

grids 沿用已有量化图的 13 项约定：

```
[qk_scale, qk_zero, attention_scale,
 scaled_logits_scale, scaled_logits_zero,
 logits_scale, logits_zero,
 probability_scale, probability_zero,
 historical_key_scale, historical_key_zero,
 historical_value_scale, historical_value_zero]
```

所有 scale 必须有限且为正，zero 必须在对应位宽的整数范围内。
query、新 K/V 和输出的量化参数来自各自 QHPI 张量。

## 执行路径

V79/V81/V85 通过公共 `htp_arch.h` 的编译定义维护。

V81 HMX 路径支持 `B=1,D=256,Hq/Hkv∈{4,8},1<=Hkv<=8,1<=Q<=256`，
且 `C+Q<=16640`。头数、容量及有效长度从输入获得。
原 16Q/2KV/Q128 和 Q1 形状保留原来的快速路径：
有效长度达到 8192 时使用 online512；较短长度使用 DMA 流水路径；
16Q/2KV 的 Q1、有效 KV 不超过 8 时使用 HVX 短序列路径。
8Q/2KV 的 FP16 分支使用分组比 4 的 HMX 路径；decode 保留硬件的 8 行 tile，
只填充并输出 4 个有效 query 行。16Q/2KV 专用的 HVX 短序列路径
不会用于 16Q/4KV。Prefill 的 Q、新 K/V 和输出可位于 DDR 或 TCM。
HVX worker 数由执行时获得，最多 8 个。

分组比 4 的 FP16 长 prefill 使用两头一组的 GQA 复用路径：每个 32 行 query
块只打包一次，K/V 的 DMA 数据供两个 query head 共用，HMX 请求一次提交
两个 32 行矩阵。保留 512-token online softmax，不生成全长 score 中间量。
8Q/2KV、Q128、有效 KV 为 8192 时，HMX 请求从 2048 次降到 1024 次，
K/V DMA 字节数减半；HMX 的矩阵乘法数量不变。

该路径要求至少 64 个有效 query、1024 个有效 KV，并且有足够任务使用
当前所有 HVX worker。每个 worker 使用 386 KiB TCM，并保持 HMX activation
要求的 2 KiB 对齐；8 worker 的工作区约 3.02 MiB，由各层复用。导出时检查目标的 TCM tiling 上限，
不足时保留原路径。旧 context 的 scratch 较小，运行时也会保留原路径；
要使用新路径需重新导出 PTE。Decode 和分组比 8 的原路径不变。
不足一个 query 块且有效 KV 小于 8192 的尾块保留原路径，保持其概率 U16 中间量化。
Decode 的 scratch 占位初始化注册为单线程 HVX；实际初始化和计算仍在原 exclusive kernel 中执行。

原 FP16 packed 路径在加载 U8 KV 时用 HVX 扩展并反量化为 FP16。
原始历史缓存减半，HMX 内部仍使用 FP16 packed KV。Q 和输出保持 U16。
8K、2 KV heads、D256 的六层 FA 共分配约 99 MiB packed 工作区；不再按
16640 行固定分配 195 MiB。
U8 decode 复用增量打包；16Q/2KV 的直接读取 U16 新 KV 的短路径仅用于 U16。

V81、分组比 4、U8 KV 的 prefill 另有整数路径，实现在
`GroupedPrefillU16.h`。它使用 HVX 装填和重排，HMX 计算 U16 activation × S8
weight，随后用 I32 临时量做零点补偿和定点缩放，饱和输出 U16；QK/PV
矩阵和 online accumulator 不重建成 FP32。Softmax 使用单调整数指数查表，
U32 normalizer 与 I32 accumulator 使用 Q15 online rescale。浮点运算仅用于
标量量化系数设置。两次 HMX U16 输出共同保留缩放后的 I32 部分和，
不会将完整点积直接截断到 U16。

该路径直接在 HMX probability 布局内处理 logits。1024 token 是 online
块的上限，实际宽度受有效 KV 和每个 32-query tile 的因果边界限制，按
32 个 key 对齐；完全不可见的块不提交 HMX 任务，也不更新 online 状态。
history=0、valid=128 时，四个 query tile 分别只计算 N=32/64/96/128。
仅保留对角 32×32 硬件 tile 内的 mask，不固定补算 1024 个 key。

`StreamKv.h` 按 KV head 顺序扫描一次 K/V，HMX 服务线程通过 8 个 64 KiB
slot 预取，每份数据在 TCM 内供该 KV head 的全部 query tile 和 head group
复用。slot 在实际需要它的读者完成 HMX 后才复用；暂不能处理的请求保留在
队列中，服务线程继续处理其他请求。AR128 及其部分尾块使用同一共享流水。
全局共享要求 8 个 HVX worker，且每个 KV head 不超过 8 个 query/head-group
任务；其他情况使用独立搬运，同时保留因果裁剪。

8Q/2KV、D256、history=8064、valid=128 时，K payload DMA 为 4 MiB，V 为
4 MiB，合计 8 MiB；此前成对共享的整数版为 32 MiB。history=0、valid=128
时合计仅为 128 KiB。这是 payload 描述符字节数，不包含其他元数据、输入输出
及片内 HMX 读取，也不代表外部 DRAM 总线计数。该满 8K 样本的 QK matrix
调用为 8144，PV 为 2048；512-token 旧路径的 PV 为 4096。

整数和 FP16 HMX 服务在处理请求时先取得 job 参数的局部快照，再执行矩阵循环。
这使编译器能将地址和步长保留在寄存器中，避免每个 32×32 tile 都从共享
TCM job 重新加载参数；仅减少 payload DMA 不能消除这部分调度开销。
HVX QK 转换也在进入列循环前缓存量化零点和两条 query 行的修正值，
避免在每组列上重复读取和计算这些不变量。

PV 的行修正值和 rescale 在每个 online 分块计算一次，供四组输出列复用。
首块直接写入 accumulator，并跳过旧状态的指数缩放和 accumulator 清零；
后续块沿用原有 online 更新。奇数尾行只读取实际存在的行参数。

整数路径与原 grouped 路径均使用每 worker 386 KiB scratch，可复用已经
导出这部分工作区的 context。新 K/V 的量化网格必须与缓存一致，原始/缩放后
logits 网格及 I32 范围必须兼容；不满足条件时保留原路径。packed 格式由缓存
状态显式记录，切回 FP16 时完整重打包，防止复用不同格式的数据。
该近似计算不要求与 SDK 或原 FA bit-exact，性能和质量需分别验证。

V81 的 `8Q/2KV,D256,Q1,U8 KV` 在量化网格及 I32 范围兼容、HVX worker
为 2/4/8 时使用 `IntegerDecode.h`。QK/PV 用 HVX 做整数点积、I32 零点补偿
和定点缩放，输出 U16；不把 KV 展开成 FP16。四个 GQA query head 共用一份
K/V，每个 KV head 最多再按有效历史分为四段，使用 2/4/8 个活跃 worker。
分段数取 2 的幂以保持 DMA bank 对齐。每个 worker 使用两个 32 KiB bank
预取，复用原 decode scratch；QK 最大值和 PV 部分和分别在段间归并。
KV 只按有效长度补齐到 64，DMA 每块至多 128 个 key。

Query 和 softmax 权重用 HVX 分离高低字节，标量乘法操作数留在可缓存栈上，
避免逐元素读取 VTCM。AR1 的新 K 用 HVX 批量读取、求和后按 U32 写入，
新 V 用字节 mask 向量写入 packed 布局。Inline decode 将已校验的配置和
缓存映射传给初始化及执行函数，避免重复解析元数据和扫描 FD 链表。
初始化阶段将整数格式选择写入共享头，控制线程与 HVX worker 使用同一个选择。AR1 只追加新 token 的 packed KV 和 K sum；
64-token padding 边界改变时重排尾块，读取原始缓存前失效完整尾部，避免
直接追加后复用上一轮请求的缓存行。原始缓存失效完成后，所有初始化 worker
同步再开始打包，防止其他 worker 提前读取旧行。兼容的整数 prefill/decode
之间不切换格式。
其他 decode 形状或量化网格继续使用原分支。

兼容的 prefill/decode 固定共用 KV8 packed 缓存、I32/U32 临时计算和 U16
输出。`PackedKv8.h` 统一 K、V、K sum 的地址及尾块步长；初始化、HMX
prefill、HVX decode 和共享 DMA 均使用该布局。`Slot::can_append` 统一检查
格式、epoch、有效历史和追加标记，阶段切换本身不触发整段 FP16 转换或重打包。
新增 KV 仍需增量装填，64/256-token 边界只重排必要尾块；量化参数必须与
缓存网格一致。prefill/decode 分别保留 HMX/HVX 调度。

整数 decode 的 packed 分配保留原容量及 FP16 分支的预留空间，但有效 K/V
数据和 DMA payload 均为 8 bit；预留空间大小不等于实际搬运量。

其他合法形状、更大容量及 V79/V85 使用通用 FP32 online 路径。
其工作区只随 D 和 worker 上限增长，不随缓存容量增长。
它保证参数适配，不代表这些形状具有相同的 HMX 性能。

HMX 的 tile 和 QNN scratch 保留固定上限，packed KV 按缓存容量分配。
它的搬运与计算只覆盖有效 KV；整数 prefill 按 32 个 key 对齐，整数 decode
和原 FP16 路径按 64 个 key 对齐。
通用路径不遍历无效 KV。原 FP16 HMX 路径的 FP32 online 累加取消了旧归一化概率 U16 中间量化；
不能用 bit-exact 作为它与原 SDK 分解图一致的要求。

执行分支决定 PTE 中的工作区分配。将分组比 4 从 generic 切换到 HMX 后，
需重新导出 PTE，并部署匹配的算子库。
调整 packed KV 的存储方式不改变 QNN scratch，可以复用原 context；
预分配模式要求 runner 与支持 bit31 的算子库配套更新。

## 导出与生命周期

导出器通过 `attention_support.package.attention_package(host, device)` 注册该算子。
同时使用 GDN 时，将返回的 `op_package_infos` 加入已有导出选项；Attention 的
注册不依赖或修改 `deltanet_package`。
FD 版本仅可在 QNN 上执行；主机校准/验证应先用真实缓存张量调用
`attention_reference`，再在设备图边界替换为描述符。
`attention_quant_annotations` 仅标注 Q、新 K/V 与输出，描述符、长度和 grids 不量化。
该标注器默认统一使用 U16；混合 U16 Q/U8 KV 从已校准的 Q/DQ 图替换获得。

MoE 导出 profile 设置 `attention_core: "valid_online"` 后，由
`attention_support/graph.py` 替换已校准的 Attention 核心，同时在 `GRAPHS.json`
记录缓存布局及量化参数。Runner 据此共享 Prefill/Decode 的原始 KV 缓冲，
传入实际有效长度。默认 `sdk` 及已有 PTE 保留原来的图。

V81 构建会直接编译 SDK `libs/qhl_hmx` 中用到的三个 HMX 函数并链接进
算子库，不额外分发 HMX 动态库。

算子缓存 FD 映射和 packed KV。op package terminate 时清理。
共享 DSP 进程若在 package terminate 前释放模型 FD，需要在所有相关图
结束执行后调用 `release_attention_cache_maps()`，再销毁 FD；该入口释放
此包的全部 Attention 映射，不能与执行或其他共享模型的调用并发。

## 离线检查

```bash
source env.sh
make -C examples/qualcomm/custom_op/attention_op_package_htp/AttentionOpPackage \
  WORK=/tmp/et-attention-op-build htp_x86 htp_aarch64 htp_v79 htp_v81 htp_v85 test-attention test-attention-u16 test-attention-stream

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 \
python -m pytest -o addopts= examples/qualcomm/custom_op/tests/test_attention.py -q
```

C++ 用例比较独立 double softmax 参考，并污染无效缓存和尾部新 KV 验证读取边界；
Python 用例与 PyTorch SDPA 对照并验证 descriptor 图可导出。
`test-attention-u16` 通过 SDK libnative 验证 HVX 装填、定点缩放、655360 个
指数输入的单调性及误差，并用两个线程验证共享 K/V slot 的生命周期。
`test-attention-stream` 用异步 DMA 模拟和真实 stream 调度代码验证多读者生命周期、
因果块跳过、跨 KV head 的加载顺序、payload 单次读取，并验证被延后的 HMX
请求能继续完成；该测试不依赖 SDK libnative。
编译和离线数值检查不能替代新增形状的真机性能、PPL 或端到端质量验证。
