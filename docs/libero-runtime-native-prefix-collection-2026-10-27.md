# LIBERO runtime-native 前缀观测采集（2026-10-27）

## 目的与边界

此前的 replay 可行性审计发现，source-SPI 的 150 条 `meta_support` 示教中，当前 runtime 只有 17 条可以执行到 HDF5 action 序列末端；其余 133 条会在环境提前判定 success。为了判断历史 HDF5 观测语义是否值得单独建立 runtime-native 训练协议，本次只采集这些**实际可执行前缀**的当前 runtime pre-action observation。

本次没有训练 Policy、执行 Policy 动作、访问 candidate/validation/held-out/qualification、选择 checkpoint、更新 SPT/Knowledge 或注册 Module。私有 simulator state 只用于按示教 `init_state` reset；Policy-facing 数据只包含两路 RGB 和公开 joint/gripper observation。

协议：[runtime-native prefix protocol](../configs/libero_runtime_native_prefix_protocol_pilot_v1.json)，SHA-256 `56b4aab18f23e7c9123f6313d0e1775dc7a30ca3db6826a0a72215d102acba08`。

## 固定采集规则

- 任务：source-SPI task `0/1/3/4/7`。
- 示教：每个任务的 `meta_support` demonstration `0..29`，共 150 条。
- 每条前缀长度：`L = min(HDF5 action count, 当前 runtime 实际执行 action count)`；不在 early success 后补观测或动作。
- 每一步先记录当前公开 observation，再执行对应 HDF5 action；动作目标因此与 historical HDF5 causal view 相同。
- 最短固定前缀阈值为 32 步；预检得到实际范围 63–197 步，中位数 110 步。

采集器：[collect_libero_runtime_native_prefix.py](../experiments/collect_libero_runtime_native_prefix.py)。第一次运行在 task 0 / demo 15 暴露了“完整前缀”与“前缀末端必须终止”的错误混淆；复现确认该轨迹稳定执行完 84 个 action 且不终止，随后修正为按封存 replay 的 `terminal_reason` 校验。修复提交为 `21ec39c`，定向测试为 `7 passed`。

## 结果

结果 JSON 位于实验主机 `/tmp/libero-runtime-native-prefix-20261027-v2.json`，SHA-256 `b447373621906af25b60acf718652f714c1f1a08fac8e4380d4959999b1d302f`；150 个压缩 NPZ 位于 `/tmp/libero-runtime-native-prefix-20261027-v2/`，总大小 `692,762,894` bytes。

| 项目 | 结果 |
|---|---:|
| 任务数 | 5 |
| 示教数 | 150 |
| 完整 HDF5 action 前缀 | 17/150 |
| runtime 提前 success 的前缀 | 133/150 |
| 前缀长度范围 | 63–197 |
| 前缀长度中位数 | 110 |
| 实际磁盘 NPZ 文件 | 150/150 |
| 终止状态与封存 replay 一致 | 150/150 |

133 条 runtime success 前缀在采集末端记录 `terminal_reason=success`；17 条完整执行 HDF5 action 的前缀记录 `terminal_reason=null`、`success=false`。这与封存 replay 结果一致：前缀可用于同一 action 目标下的 observation 对照，但不能伪装成 150 条完整 runtime episode。

每个 NPZ 同时保存 `agentview_rgb`、`eye_in_hand_rgb`、`joint_states`、`gripper_states` 和前缀 actions；结果 JSON 保存文件 SHA-256、数组 digest、长度、reset 索引、终止原因和数据集 provenance。重新统计确认每个 task 30 条、路径唯一且全部存在。

## 诊断意义与限制

这一步只解决了“哪些 runtime observation 可以与同一 HDF5 action 目标配对”的数据边界问题。它还没有回答 runtime-native 观测训练是否能提高闭环成功率，也不能把 early success 解释为示教完成、Policy 成功或 SPI 资格。后续 pilot 必须在相同前缀、相同 action target、相同窗口长度、相同初始化和 100-update 预算下，比较 historical HDF5 causal observation 与 runtime-native observation 两臂；两臂只能共享可配对前缀，不得把短前缀拼接或补齐为完整示教。

