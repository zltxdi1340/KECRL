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

## 下一步

先不把这两臂注册为 baseline 或 Module，也不启动 A/B/C。下一项保持同一前缀协议，增加一个明确的预算轴：在同一 seed、同一 task 和同一 candidate 状态上，把 100 update 扩到 500 update，只比较 historical/runtime-native 两臂的闭环曲线和 action/state drift；如果两臂仍都失败，主瓶颈继续指向动作链执行/状态覆盖，而不是观测语义。只有出现可重复的 runtime-native 闭环改善，才冻结 runtime-native observation 作为后续 SPI 训练协议；否则保留 HDF5 causal 离线协议，并把 runtime-native 只用于闭环对齐诊断。

