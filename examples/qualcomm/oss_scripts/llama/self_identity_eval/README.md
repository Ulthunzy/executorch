# 工具已迁移

正文与最新使用说明见 [benchmarks/README.md](../model_validation/benchmarks/README.md)。

旧 Python 路径保留 CLI 和函数导入兼容入口；请在新目录修改实现。
测试仅在新目录保留，避免重复收集。
迁移后脚本哈希可能变化；旧结果不能在哈希不匹配时原地 resume，请保留原结果并使用新输出目录，不绕过校验。
本目录的小型 JSON 文件仅作兼容副本，canonical 文件位于 `../model_validation/benchmarks/`。
消融工具见 [model_validation/ablation](../model_validation/ablation/README.md)。
