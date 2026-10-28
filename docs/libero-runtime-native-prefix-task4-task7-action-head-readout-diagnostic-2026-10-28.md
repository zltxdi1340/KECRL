# LIBERO task 4/7 action-head 与表示读出诊断（2026-10-28）

## 目的

阶段命令诊断显示 task 7 的 HDF5 Policy 在候选轨迹中长期没有 close，task 4 则出现状态依赖的提前 release。此前的观测替换和 eye-in-hand 敏感性结果无法区分视觉表示、fusion/recurrent 映射和 action head 的贡献。本项固定同一训练初始化、数据顺序、500 次 Adam 更新和 CNN-only + uniform seven-dimensional action MSE，比较：

1. **full Policy**：visual encoder、fusion、recurrent 和 action head 全部更新；
2. **action-head-only**：冻结 visual encoder、fusion、recurrent，只更新 1,799 个 action-head 参数。

实验覆盖 task 4、task 7，各使用 historical HDF5 与 runtime-native 两个 observation arm。source-support 前缀用于监督误差；candidate 只使用已经归档的 20 步 policy-facing observation，报告输出动作和 gripper schedule，不把 candidate 输出差异当作专家动作误差。

## 可复现性与边界

- 配置：[action-head readout diagnostic](../configs/libero-runtime-native-prefix-task4-task7-action-head-readout-diagnostic_v1.json)，SHA-256 `641838c52fcf2fae0aa3be6eb650845bdba114b72962a39fee87686c0b453bb6`。
- 结果：`/tmp/libero-runtime-native-prefix-task4-task7-action-head-readout-20261028-v1.json`，SHA-256 `fa106679671d6fe09030a01f1bfa23c7a0281d3248b3f548bc3c8be3a489a9cf`。
- 输入使用既有 cross-task observation archive，先校验 9 个固定案例、72 个数组和每一步 observation digest，再开始训练。
- 运行环境：Python `3.10.22`、PyTorch `2.4.1+cu124`、`cuda:1` / NVIDIA GeForce RTX 4090；8 个模型、每个 500 updates，耗时约 `45.35 s`。
- full Policy 重建的最终参数 SHA 与既有 coverage 结果一致；20 步 candidate action 与既有 failure trace 一致，误差容限为 `1e-6`。
- 没有访问 validation、held-out 或 qualification，没有选择 checkpoint，没有执行 cross-policy action，没有更新 SPT/Knowledge，也没有注册 Module。

## source-support 监督误差

| task | observation arm | full Policy MSE | action-head-only MSE | head/full | full gripper MSE | head-only gripper MSE |
|---:|---|---:|---:|---:|---:|---:|
| 4 | historical HDF5 | 0.02656 | 0.21104 | 7.95× | 0.07600 | 0.92798 |
| 4 | runtime-native | 0.03132 | 0.21084 | 6.73× | 0.10147 | 0.92700 |
| 7 | historical HDF5 | 0.03300 | 0.18288 | 5.54× | 0.11196 | 0.80618 |
| 7 | runtime-native | 0.03290 | 0.18304 | 5.56× | 0.11633 | 0.80669 |

冻结表示后，action head 无法把 source-support 动作拟合到 full Policy 的误差水平。差距在两个任务、两个 observation arm 上方向一致；runtime-native 与 HDF5 的 head-only 误差几乎相同，说明仅比较两种 observation 来源不能解释这个读出瓶颈。该结果支持“端到端表示/融合/时序映射学习是必要的”，但不能进一步区分 visual encoder、fusion 和 recurrent 哪一个单独负责。

## candidate 短窗口输出

candidate 只用于固定输入上的输出对比。full Policy 的第一步 gripper 命令保持与既有 trace 一致；action-head-only 的四个 task/observation 组合在所有 10 个固定案例中都在第 1 步输出正向 close。相对于 full Policy，前 20 步 action MAE 为：

| task | observation arm | candidate 数 | action MAE（head-only − full） | gripper MAE | head-only 首次 close |
|---:|---|---:|---:|---:|---:|
| 4 | historical HDF5 | 3 | 0.3512 | 1.1306 | 1 |
| 4 | runtime-native | 3 | 0.3382 | 1.0885 | 1 |
| 7 | historical HDF5 | 2 | 0.3049 | 1.2474 | 1 |
| 7 | runtime-native | 2 | 0.3261 | 1.2860 | 1 |

这些数值是两个读出模型在同一 candidate observation 上的输出差异。candidate 没有逐步专家动作标签，因此它们不是 candidate action error，也不能据此判断哪个模型在环境中会成功。action-head-only 的“第 1 步 close”表明冻结的初始表示和下游状态映射无法提供正确的开合读出；它不能证明 full Policy 的失败一定由视觉造成。

## 诊断结论

1. **只训练 action head 不够。** 任务 4/7 的 source-support MSE 高出 full Policy 5.5–7.9 倍，gripper MSE 高出约 7–12 倍，排除了“只需重新拟合最后一层”这一简单解释。
2. **表示/融合/时序映射确实参与瓶颈。** full Policy 的收益来自至少一个被冻结模块的学习，但当前对照没有把 visual encoder、fusion、recurrent 分开，因此不能写成“CNN 已被证明是根因”。
3. **task 7 的 close 缺失不是单纯标签缺失。** 在相同 candidate observation 上，full Policy 复现既有长期 open trace，而 action-head-only 反而立即 close；这说明 close schedule 对 Policy 映射和表示共同敏感，不能用固定提前 close 或增加 gripper loss 作为全局修复。
4. **两种 observation arm 的 head-only 误差接近。** 该结果不支持把当前失败主要归因于 HDF5/runtime-native 的 observation source；前期 eye-in-hand 敏感性仍然是局部输入机制证据，不能替代本项的读出分解。

## 限制与下一项

本项只有 task 4/7、一个训练 seed 和 500 updates，candidate 结果是离线短窗口读出，不是闭环成功率或 SPI qualification。下一项保持数据、seed、预算和 CNN-only uniform loss 不变，做冻结层级的三臂分解：

- 冻结 visual encoder，只训练 fusion + recurrent + action head；
- 冻结 visual encoder + fusion，只训练 recurrent + action head；
- 保留本项 full Policy 与 action-head-only 作为上下界。

在 source-support 监督误差上先判断表示、融合和时序映射的边际贡献；只有某一层级在多个任务和 seed 上显示稳定差距，才决定是否进入更大预算或闭环复核。期间不更新 SPT/Knowledge，不注册 Module，也不把诊断结果当作正式 SPI 结果。
