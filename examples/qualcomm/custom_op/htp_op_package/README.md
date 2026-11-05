# Shared HTP operator bundle

`attention_op_package_htp/AttentionOpPackage` 和
`deltanet_op_package_htp/DeltaNetOpPackage` 分别维护 Attention 和 GDN 的
计算源码、参数定义与测试。本目录只负责共同构建、QNN 包入口、Python 注册
辅助及通用 HVX 工具，不实现 Attention 或 GDN 计算。

两个算法的 Python 导出映射分别位于 `attention_support/package.py` 和
`deltanet_support/package.py`。两者使用相同包入口，分别注册各自的算子。

QNN 序列化包名 `DeltaNetOpPackage`、接口符号
`DeltaNetOpPackageInterfaceProvider` 和已有导出侧 SO 文件名保持兼容。
这里的名字是已有 PTE 的协议标识，源代码不再嵌套放置。
设备仍交付 `libet_infer.so` 与合并后的 `libet_qnnop.so`。

```bash
source env.sh
bash examples/qualcomm/custom_op/htp_op_package/build.sh \
  --arch 81 --output /tmp/et-htp-ops
```

两个算子目录中的 Makefile/build.sh 均转到此处同一套构建规则。
`QCOM_HTP_ARCH=79/81/85` 由公共 `htp_arch.h` 管理。
