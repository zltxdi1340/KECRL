# LIBERO 候选轨迹阶段命令诊断（2026-10-28）

## 目的与边界

eye-in-hand 平移增强已被停止扩大：它降低了固定状态敏感性，却使离线 MSE 和 candidate 成功率下降。本项回到不改 Policy 的机制诊断，检查失败是否表现为任务阶段命令分布断裂，尤其比较 task 4 的 open-container 条件和 task 7 的 fixture-surface 条件。

本项将两类数据分开处理：

- **source 阶段分布**：读取 150 条 runtime-native source-support 前缀中的 HDF5 action，按每条示教自己的 gripper action 符号划分 `approach / transport / release`；
- **candidate 阶段命令**：读取 failure-trace 中固定的 18 条 candidate 轨迹，按 Policy 实际输出的 gripper action 符号划分同样的命令阶段。

candidate 轨迹没有对应的专家逐步动作，因此不能计算 supervised action error；candidate 阶段也只是 Policy 的夹爪命令 schedule，不是真实环境 phase 标签。RGB 只用于计算相对 candidate 初始 observation 的变化，不作为 target-relative geometry 标签，也不读取 simulator 私有物体坐标。

协议：[phase command diagnostic](../configs/libero-runtime-native-prefix-phase-command-diagnostic_v1.json)，SHA-256 `a922a256047c3bbe3fb607a4c9eff9a953ee435fd346149a8069554a4a387f57`。结果文件位于实验主机 `/tmp/libero-runtime-native-prefix-phase-command-20261028-v1.json`，SHA-256 `da0680c6b62fcf5662e9ebb4454d5a303f4905b12adaaf11f11733def0373860`。source cross-task result SHA-256 为 `c734966c4d84eb615e2b408df67303bc01ec9b06273410dbb70f69b06ac37c9e`，failure trace SHA-256 为 `4b453f994df9a5a59889b0cf3d94fd6ecc4a5d65756ccc235c7f485381631085`，observation archive SHA-256 为 `226c85c406e0d2f237e37afc3171a1421de2755fff9bb2ba83396860aa0dd9b3`。

本项是离线归档分析，没有训练、rollout、checkpoint 选择、validation/held-out/qualification 访问、SPT/Knowledge 更新或 Module 注册。

## Source 阶段分布

| task | source steps | approach | transport | release | 首次 close 中位步 |
|---:|---:|---:|---:|---:|---:|
| 1 | 3513 | 44.6% | 45.8% | 9.7% | 52 |
| 3 | 2688 | 48.2% | 44.9% | 6.9% | 43 |
| 4 | 4001 | 37.7% | 61.1% | 1.1% | 51 |
| 7 | 3844 | 25.3% | 65.4% | 9.3% | 31 |

这是所有 30 条 source-support prefix 的汇总；release 比例受当前 runtime prefix 提前 success 和示教终止影响，不能解释为任务没有 release 标签。

## Candidate 命令分布

下表是固定候选 trace、两个 observation arm 的合并结果。它描述 Policy 命令 schedule，不是专家动作误差。

| task | arm | trace 数 | success | approach | transport | release | 首次 close 中位步 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | HDF5 | 2 | 0/2 | 21.8% | 78.2% | 0.0% | 66 |
| 1 | runtime | 2 | 1/2 | 26.7% | 73.3% | 0.0% | 58 |
| 3 | HDF5 | 2 | 0/2 | 20.7% | 79.3% | 0.0% | 62 |
| 3 | runtime | 2 | 1/2 | 26.4% | 20.8% | 52.8% | 52 |
| 4 | HDF5 | 3 | 1/3 | 29.4% | 15.6% | 55.0% | 79 |
| 4 | runtime | 3 | 1/3 | 23.3% | 76.7% | 0.0% | 60 |
| 7 | HDF5 | 2 | 0/2 | 100.0% | 0.0% | 0.0% | 无 |
| 7 | runtime | 2 | 1/2 | 22.9% | 23.3% | 53.8% | 50 |

### 关键差异

1. **task 7 的 HDF5 臂仍是最清楚的阶段断裂。** 两条 candidate 轨迹 600 步全部保持在 `approach`，没有正向 close；source 中 close 占多数，首次 close 中位步为 31。这个结果复现了既有 failure trace 的观察，但仍不能区分视觉读出、action head 或状态覆盖。
2. **task 4 的 HDF5 臂不是“没有 close”，而是命令时序异常。** 两个失败状态很早进入 release（首次 close 后约 1/22 步出现 release），一个成功状态则一直保持 transport；runtime 臂三条轨迹都保持 transport，其中一条成功。这说明固定提前 close、延长 close 或统一 release 规则都不能作为全局修复。
3. **task 3 runtime 臂出现状态依赖的 release。** state 0 成功且没有 release，state 1 失败并在约第 94 步开始 release；同一 arm 的阶段 schedule 不一致，说明夹爪命令不是唯一充分条件。
4. **task 1 两臂都没有 release。** runtime state 17 成功但仍没有 release，说明 LIBERO 的 success 判定在这些 trace 中不必然对应可观测的开合序列完成；candidate schedule 只能作为执行线索。

## 早期公开观测漂移

第 20 个 pre-action observation 相对初始 observation 的 task-level 平均如下；RGB 是像素 MAE，joint 是公开 joint-state L2。

| task | arm | joint L2 | agentview RGB MAE | eye-in-hand RGB MAE |
|---:|---|---:|---:|---:|
| 1 | HDF5 | 0.464 | 2.903 | 36.581 |
| 1 | runtime | 0.452 | 2.980 | 37.130 |
| 3 | HDF5 | 0.450 | 3.246 | 40.508 |
| 3 | runtime | 0.757 | 4.752 | 42.416 |
| 4 | HDF5 | 0.729 | 2.038 | 47.788 |
| 4 | runtime | 0.677 | 2.284 | 49.143 |
| 7 | HDF5 | 0.296 | 3.736 | 21.117 |
| 7 | runtime | 0.335 | 3.916 | 22.462 |

task 4 的 eye-in-hand 漂移最大，与既有分层诊断的 open-container 困难一致；task 7 的视觉漂移较小，但 HDF5 臂仍完全没有 close。这使“视觉状态漂移是唯一根因”的解释不成立：task 7 还需要检查早期 Policy mapping/action-head 读出和候选状态覆盖。

## 诊断判断

- 当前证据支持两个并存机制：task 4 的几何条件会放大早期状态/视觉漂移，task 7 的 HDF5 Policy 存在明显的候选 close 命令缺失。
- 但 candidate 没有 expert action 对齐，不能把 phase schedule 差异直接称为 action error；release/transport 也不是 simulator 提供的真实任务阶段。
- 因此不能直接增加 gripper loss、提前 close、强制 release，或再次扩大 eye-in-hand 平移增强。现有 CNN-only + uniform action MSE 仍是基线，SPI/SPT/Knowledge/Module 边界不变。

## 下一步

下一项固定已有 Policy，不改训练数据和 loss，针对 task 4 与 task 7 做 **action-head / visual representation 对照**：在相同 source-support 与 candidate 初始 observation 上，分别测量冻结视觉表示下 action head 的读出误差，以及完整 Policy 的读出误差；candidate 只作为初态/短窗口机制输入，不作为监督标签。重点检验 task 7 的 close 缺失是否已经出现在初始状态映射中，以及 task 4 的漂移是否主要伴随视觉表征变化。若 action-head-only 与 full Policy 差距显著，再转向数据覆盖和表示学习；否则检查候选状态到动作映射的条件化边界。

