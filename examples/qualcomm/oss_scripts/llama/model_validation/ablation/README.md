# 消融实验

统一使用 `../run.py workflow` 和 `../configs/generation.example.json`。
think 开/关 × RP 1.0/1.1、窗口 128 是示例四组；可在配置中增加 1.05 或双保护组合。
完整命令见 [主 README](../README.md)，添加阶段见 [扩展指南](../EXTENDING.md)。

标准 PPL 使用 [独立 PPL 流程](../ppl/README.md)，不施加采样、重复惩罚或 think 修复。

## 历史脚本清理

硬编码 08/09/10 路径的旧队列、旧 PPL driver、候选图加速探针及其专用测试，
以及 `self_identity_eval/ablation/` 的重复转发入口，已在 2026-09-15 归档移出。
通用入口没有对它们的运行时依赖。常用的旧对话/校准命令仍保留轻量转发。

本次归档位置（本机，不随源码发布）：

`llama-output/model-validation-cleanup.woKNbV/legacy-ablation.tar.gz`

归档保留旧脚本和原 README。需要查看时先解压到新的临时目录，不要直接覆盖当前源码。
历史输出 `llama-output/ablation-08-09-10-20260914`、权重和数据集均未修改。
旧 PT2 mask 改图只是诊断，不会自动用于新模型；原图因果性检查失败时应拒绝计分，不能沿用旧图修补结果冒充标准 PPL。
