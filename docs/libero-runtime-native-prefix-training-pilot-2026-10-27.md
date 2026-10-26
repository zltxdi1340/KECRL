# LIBERO historical HDF5 vs runtime-native 前缀训练 pilot（2026-10-27）

## 目的与边界

runtime-native 前缀采集已经把 150 条 source-support 示教裁成可执行的真实 runtime 前缀。本 pilot 检查：在完全相同的 action target、前缀长度、Policy 初始化、batch 顺序和优化预算下，使用 runtime-native pre-action observation 训练是否比历史 HDF5 causal observation 更接近当前 runtime 闭环。

这不是正式 SPI、SPT 更新或 Module 资格实验：

- 两臂都是 CNN-only、七维等权 action MSE；每个 task 100 次 Adam update，只有 observation 来源不同。
- 任务为 source-SPI `0/1/3/4/7`，每个 task 使用相同的 30 条 `meta_support` 可执行前缀；seed 为 `20261021`。
- 训练后两臂都只在同一 task 的 candidate-nonregression 初始状态 `0/1/2` 上运行，共 15 个 episode/arm。
- 没有读取 meta-query、validation、held-out 或 qualification，不选择 checkpoint，不更新 SPT/Knowledge，不注册 Module。

冻结协议：[runtime-native prefix protocol](../configs/libero_runtime_native_prefix_protocol_pilot_v1.json)，SHA-256 `56b4aab18f23e7c9123f6313d0e1775dc7a30ca3db6826a0a72215d102acba08`。采集输入报告见 [runtime-native prefix collection](libero-runtime-native-prefix-collection-2026-10-27.md)。

## 运行与结果

结果 JSON 在实验主机 `/tmp/libero-runtime-native-prefix-training-pilot-20261027-v1.json`，SHA-256 `557725c53082cb689c5bcf5721f6174811414f07a5747f5c8e4eefeb8fc8d9a7`。运行时为 Python `3.10.22`、PyTorch `2.4.1+cu124`、`cuda:0` / RTX 4090，耗时约 240 秒。

每个 task 内两臂初始模型 SHA 完全相同，均执行 100 updates；所有 30 个 candidate episode 都到达 300 步 `time_limit`。

| 观测臂 | offline prefix MSE | translation MSE | rotation MSE | gripper MSE | gripper sign accuracy | candidate 成功 |
|---|---:|---:|---:|---:|---:|---:|
| historical HDF5 causal | 0.09031 | 0.13683 | 0.00254 | 0.21403 | 0.9382 | 0/15 |
| runtime-native | 0.09091 | 0.13585 | 0.00252 | 0.22122 | 0.9373 | 0/15 |

两臂合计使用 `16,464` 个有效前缀 step。task 级别的 offline 方向不一致：runtime-native 在 task 0、7 的总 MSE 略低，在 task 3 明显更高；没有形成跨任务一致收益。translation 和 rotation 的小幅下降没有转化为闭环成功，gripper 拟合略差。

## 诊断判断

1. **本 pilot 没有发现 runtime-native 观测在 100-update 单 seed 下改善闭环。** 两臂 candidate 成功率均为 `0/15`，不能支持 runtime-native 已解决基础 Policy 执行瓶颈。
2. **也不能据此否定 runtime-native 协议。** 100 updates 是校准 pilot，不是与既有 500-update 闭环结果同预算的正式比较；两臂都可能受训练预算、动作 head 和状态覆盖限制。
3. **输入语义差异确实进入了拟合结果，但方向混杂。** 两臂初始参数、action target 和前缀窗口相同，offline 误差只在 task/动作分项上局部变化；这与此前 HDF5/runtime observation 替换会改变 Policy 输出的审计一致，但尚未给出闭环因果收益。

## 500-update 预算复核

按独立冻结配置 [500-update training pilot](../configs/libero_runtime_native_prefix_training_pilot_500_v1.json) 完成同一前缀对照。运行结果位于实验主机 `/tmp/libero-runtime-native-prefix-training-pilot-20261027-500-v1.json`，SHA-256 `fe0c28451045ac74c5ef541b85d2264f1a4c75adff5cce0ca0530f450ece903e`。配置 SHA-256 为 `6b397719c737e36c5aaebae97ecc18f3d0ed1e5228c9eb5df7563d4ce7979efa`；运行提交 `e855e03`，Python `3.10.22`、PyTorch `2.4.1+cu124`、`cuda:1` / RTX 4090，耗时 `248.54s`。每个 task 两臂均共享初始模型 SHA；每臂 500 optimizer updates、共消费 16,464 个有效前缀 step。

| 更新预算 | 观测臂 | 加权离线 MSE | translation MSE | rotation MSE | gripper MSE | gripper sign accuracy | candidate 成功 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 100 | historical HDF5 causal | 0.09031 | 0.13683 | 0.00254 | 0.21403 | 0.9382 | 0/15 |
| 100 | runtime-native | 0.09091 | 0.13585 | 0.00252 | 0.22122 | 0.9373 | 0/15 |
| 500 | historical HDF5 causal | 0.03687 | 0.04639 | 0.00200 | 0.11292 | 0.9646 | 2/15 |
| 500 | runtime-native | 0.03766 | 0.04622 | 0.00197 | 0.11905 | 0.9628 | 3/15 |

500-update 成功发生在 task 1 和 task 3：task 1 两臂各成功 1/3；task 3 的 HDF5 臂成功 1/3，runtime-native 臂成功 2/3；task 0、4、7 两臂均为 0/3。预算增加后，两臂拟合误差都明显下降，且各有少量闭环成功。runtime-native 相比 HDF5 只多 1/15 次成功，并在加权总 MSE、gripper MSE 上略差；该差距不足以证明 observation source 的优势。

因此，500-update 结果说明 100-update 预算确实偏低，但当前 Policy 仍未稳定完成这些 source-SPI。单 seed、每 task 三个固定初态只能提供 pilot 证据；不能把少量成功解释为 Module 资格或 runtime-native 的可重复收益。

## 固定评估环境的多 seed 复核

为分离训练随机性和模拟器评估随机性，配置 [500-update replication v2](../configs/libero_runtime_native_prefix_training_replication_500_v2.json) 将训练 seed 设为 `20261022/20261023`，并把 candidate 环境 seed 固定为 `20261021`。配置 SHA-256 为 `4f9218239d3cd45b9d819c6bff902bb5cc94a1ed96b5c87a33e35744d7918793`；结果 JSON 位于实验主机 `/tmp/libero-runtime-native-prefix-training-replication-20261027-v2.json`，SHA-256 `61b332a5b533f4aa56342eb59456ce4aab7f43a5db9182f45271198821c01f33`。运行提交 `d0caa68`，使用 `cuda:1` / RTX 4090，耗时 `535.30s`。每个训练 seed、task 内两臂初始模型 SHA 均相同，没有配对哈希不匹配。

| 训练 seed | 评估环境 seed | HDF5 加权 MSE | HDF5 成功 | runtime-native 加权 MSE | runtime-native 成功 |
|---:|---:|---:|---:|---:|---:|
| 20261021 | 20261021 | 0.03687 | 2/15 | 0.03766 | 3/15 |
| 20261022 | 20261021 | 0.03412 | 0/15 | 0.03538 | 2/15 |
| 20261023 | 20261021 | 0.03460 | 3/15 | 0.03562 | 3/15 |
| 合计 | 固定 | 0.03520 | 5/45 | 0.03622 | 8/45 |

45 对相同 task/初态 episode 中，两臂共同成功 4 对、仅 runtime-native 成功 4 对、仅 HDF5 成功 1 对、共同失败 36 对。runtime-native 的成功数在三个 seed 中没有反向劣势，但只净增 3/45；其加权 action MSE 均略高于 HDF5，主要差异在 gripper 拟合。故只能说出现弱的闭环候选信号，尚不能判断观测来源带来稳定收益。

曾运行的 replication v1 使用每个训练 seed 同时作为评估环境 seed。由于两种随机性共变，其输出 `/tmp/libero-runtime-native-prefix-training-replication-20261027-v1.json`（SHA-256 `46db23c5b35ce663a2c56bb21135d43c494c5978f259479f98b25ec77bd9e5f9`）保留作审计，但不并入上述结论；v2 是固定评估 seed 的有效复核。

当前仍只读取每个 task 的 candidate 初态 `0,1,2`。传递划分中 `candidate_nonregression` 有 `0..19` 共 20 个初态，qualification 是独立的 `20..39`，尚未触碰。当前低成功率下，下一步扩展到完整 20 个 candidate 初态，按 task、初态和训练 seed 分层检查成功是否集中于少数状态；不据此选 checkpoint 或做资格判定。

## 下一步

先不把这两臂注册为 baseline 或 Module，也不启动 A/B/C。扩大 candidate 初态覆盖后，再依据 task×初态×seed 失败分层决定是否需要 RGB/action-state 轨迹诊断。只有出现可复现、跨初态的闭环差异，才评估是否调整后续 SPI 训练输入协议；否则保留 HDF5 causal 离线协议，并把 runtime-native 限于闭环对齐诊断。

