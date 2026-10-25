# LIBERO source-SPI support replay 可行性审计（2026-10-25）

## 目的与边界

输入替换敏感性已经证明，历史 HDF5 causal observation 和当前 runtime observation 会改变 Policy 输出。下一步需要确认：能否用当前 runtime 沿着 source-SPI 的完整 meta-support 示教重新生成训练输入。如果当前环境经常在 HDF5 action 序列结束前判定任务成功，当前 runtime 只能生成前缀，不能直接替代完整离线训练数据。

本次访问 source-SPI 的 5 个任务（0、1、3、4、7）和每个任务的全部 30 条 `meta_support` 示教，共 150 条。每条只读取 HDF5 action 和 `init_state`，在当前 runtime 中执行固定 action；不训练 Policy、不保存 observation、不访问 validation/held-out/qualification、不选择 checkpoint、不更新 SPT/Knowledge，也不注册 Module。当前 replay 的 `success` 只表示环境边界诊断，不是 Policy 结果或 SPI 资格。

- 配置：[source support replay feasibility audit](../configs/libero_source_support_replay_feasibility_audit_v1.json)，SHA-256 `f0da50bf4109143df1f26153974b8e1d5a4acb47549b3b6451914c8e5ba68778`。
- 结果：`/tmp/libero-source-support-replay-feasibility-20261025-v1.json`，SHA-256 `0a52e6ac68dec233bc3f73646f2e4b62c9b080e4a4f69f2cdc22c1cc7c7dccf8`。
- 运行时记录提交：`8185f10b67f34268c3794adca54d646b0955673b`；代码和本报告随后提交到 GitHub。
- seed `20261021`，`stabilization_steps=0`，`max_episode_steps=300`。

## 结果

| task | 示教数 | 完整执行 action 序列 | 完整覆盖率 | 成功判定数 | 平均前缀覆盖率 | 最小覆盖率 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 30 | 1 | 3.33% | 29 | 83.33% | 54.48% |
| 1 | 30 | 5 | 16.67% | 25 | 86.29% | 58.73% |
| 3 | 30 | 4 | 13.33% | 26 | 86.78% | 68.47% |
| 4 | 30 | 3 | 10.00% | 27 | 89.41% | 76.76% |
| 7 | 30 | 4 | 13.33% | 26 | 90.93% | 81.71% |
| **合计** | **150** | **17** | **11.33%** | **133** | **87.35%** | **54.48%** |

当前 runtime 在 133 条示教中提前判定 success；这不是 replay 崩溃，也不是 action 被拒绝，而是当前环境的成功边界早于 HDF5 action 序列的末端。其余示教没有在当前 adapter 中提前成功，因而可以执行到序列末端。HDF5 的动作长度均被完整记录，因此当前 runtime 不能为大多数示教提供同长度的 pre-action observation 序列。

## 诊断结论

1. **当前 runtime regenerated observation 不能直接作为完整 source-support 训练视图。** 只有 11.33% 的示教具备完整 action-prefix 覆盖；把剩余示教截断后训练会改变示教长度分布和尾部动作覆盖。
2. **这解释了为什么“直接用当前 runtime 重建全部训练输入”不是一个无代价修复。** 它同时改变 observation 语义、有效步数和成功终止边界，无法把后续收益归因到视觉输入对齐或 Policy 学习。
3. **观察协议现在冻结为双协议。** HDF5 causal observation 作为完整离线 BC 数据协议；当前 runtime regenerated observation 只用于闭环、replay 和输入/状态对齐诊断。两者不能在同一训练结果中混用而不标注。
4. **这不是正式训练结论。** replay 成功数只是环境语义边界的证据，不能作为 Policy 成功率、SPI 资格或 Module 资格。

## 对后续实验的约束

后续阶段平衡、gripper-loss 和 CNN-only/CNN+GRU 对照先在固定的 HDF5 causal 离线协议下做，闭环评估继续使用当前 runtime，并同时报告输入替换敏感性和 replay 边界。若以后要研究 runtime-native 训练，必须另立协议：明确如何处理提前 success、是否截断示教、如何保持动作阶段覆盖，并用独立诊断预算验证其可比性。它不能直接替换当前离线协议。

本审计没有更新 SPT/Knowledge，也没有注册 Module；它只把 observation 协议和训练数据覆盖边界冻结下来。
