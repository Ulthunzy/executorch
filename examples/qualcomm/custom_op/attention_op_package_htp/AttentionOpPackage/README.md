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
| new_key | QUInt16 `[B,Hkv,D,Q]` |
| value_descriptor | Int32 `[1,1,1,3]` |
| new_value | QUInt16 `[B,Hkv,Q,D]` |
| lengths | Int32 `[1,1,1,3]`，值为 `[valid,first,history]` |
| grids | Float32 `[1,1,1,13]` |

输出与 query 同形状，使用输出张量自己的 U16 量化参数。无效 query 行写量化零。
每个有效 query 行只看 `history + row + 1` 个因果可见位置。

`B/Hq/Hkv/Q/D` 来自张量形状，要求正数、`Hq % Hkv == 0`、`D <= 4096`，
query 元素总数不超过 INT32_MAX。支持 `1 <= valid <= Q`，不限定尾块长度；
`first >= 0` 且 `first + history <= C`。lengths 在 batch 内共用。
`C` 是历史缓存容量，不包含本次独立传入的 Q 个新位置。

缓存描述符是 `[fd,bytes_and_flags,epoch]`。低 30 位表示历史缓存字节数，
bit30 是 exact-append 标志，bit31 保留为零。K/V 使用不同 FD，字节数、
标志和 epoch 一致。根据 `B*Hkv*D*sizeof(uint16_t)` 推导历史容量 C，
K 的物理布局为 `[B,Hkv,D,C]`，V 为 `[B,Hkv,C,D]`，两者均需持久映射。
调用者必须提供覆盖声明字节数的实际映射。

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

所有 scale 必须有限且为正，zero 必须是 0～65535 内的整数。
query、新 K/V 和输出的量化参数来自各自 QHPI 张量。

## 执行路径

V79/V81/V85 通过公共 `htp_arch.h` 的编译定义维护。

V81 HMX 路径支持 `B=1,D=256,Hq/Hkv=8,1<=Hkv<=8,1<=Q<=256`，
且 `C+Q<=16640`。头数、容量及有效长度从输入获得。
原 16Q/2KV/Q128 和 Q1 形状保留原来的快速路径：
有效长度达到 8192 时使用 online512；较短长度使用 DMA 流水路径；
16Q/2KV 的 Q1、有效 KV 不超过 8 时使用 HVX 短序列路径。
HVX worker 数由执行时获得，最多 8 个。

其他合法形状、更大容量及 V79/V85 使用通用 FP32 online 路径。
其工作区只随 D 和 worker 上限增长，不随缓存容量增长。
它保证参数适配，不代表这些形状具有相同的 HMX 性能。

HMX 快速路径仍保留固定 tile/workspace 上限，未把所有内部工作区改为
按上下文动态分配。它的搬运与计算只覆盖有效 KV（按 64 元素硬件边界补齐）。
通用路径不遍历无效 KV。online 的 FP32 累加取消了旧归一化概率 U16 中间量化；
不能用 bit-exact 作为它与原 SDK 分解图一致的要求。

## 导出与生命周期

导出器通过 `attention_support.package.attention_package(host, device)` 注册该算子。
同时使用 GDN 时，将返回的 `op_package_infos` 加入已有导出选项；Attention 的
注册不依赖或修改 `deltanet_package`。
FD 版本仅可在 QNN 上执行；主机校准/验证应先用真实缓存张量调用
`attention_reference`，再在设备图边界替换为描述符。
`attention_quant_annotations` 仅标注 Q、新 K/V 与输出，描述符、长度和 grids 不量化。

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
  WORK=/tmp/et-attention-op-build htp_x86 htp_aarch64 htp_v79 htp_v81 htp_v85 test-attention

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 \
python -m pytest -o addopts= examples/qualcomm/custom_op/tests/test_attention.py -q
```

C++ 用例比较独立 double softmax 参考，并污染无效缓存和尾部新 KV 验证读取边界；
Python 用例与 PyTorch SDPA 对照并验证 descriptor 图可导出。
编译和离线数值检查不能替代新增形状的真机性能、PPL 或端到端质量验证。
