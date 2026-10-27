# LIBERO task 7 交叉输入动作差异诊断（2026-10-28）

## 目的与边界

上一项初始观察诊断发现：task 7、training seed `20261022` 的两个 candidate 状态中，historical HDF5 与 runtime-native 两臂的公开初始 proprioception 相同，但首步动作已经不同。本项固定两臂的 500-update CNN-only Policy 和各自已有的 20 步 candidate free-run observation，构造四种输入组合：

| 模型 | 输入观察 | 含义 |
|---|---|---|
| HDF5 Policy | HDF5 轨迹 | 原始配对 |
| runtime-native Policy | HDF5 轨迹 | 固定观察时的模型映射差异 |
| HDF5 Policy | runtime-native 轨迹 | 固定模型时的观察敏感性 |
| runtime-native Policy | runtime-native 轨迹 | 原始配对 |

协议：[cross-input diagnostic](../configs/libero_runtime_native_prefix_task7_cross_input_diagnostic_v1.json)，SHA-256 `1f053d71925c3dfa49b5762951fd2addfda6d3852492a23eae9ad314c3160ac4`。结果文件位于实验主机 `/tmp/libero-runtime-native-prefix-task7-cross-input-20261028-v1.json`，SHA-256 `88af3f10e5b9d11c51551d82b63a6fa3af8abf682ca0983447b10749f3a1ba25`。运行使用提交 `e027d7f0d3d3e95b4c166c97f30ad04f2e68c3b5`、Python `3.10.22`、PyTorch `2.4.1+cu124`、`cuda:1` / RTX 4090，耗时约 `208.88s`。

每个模型仍按固定顺序遍历 candidate 初态 `0..19`，只详细保留 task 7 的 state `0/8` 前 20 步。四条原始组合的动作逐步复现既有 failure-trace；交叉组合只在内存中读取同一时刻的 observation 并推理，没有把交叉动作执行到环境中。没有访问 validation、held-out 或 qualification，没有选择 checkpoint、更新 SPT/Knowledge 或注册 Module。

## 结果

下表的“固定观察模型差异”比较两个模型接收同一条观察轨迹时的动作差异；“固定模型观察差异”比较同一模型接收两条观察轨迹时的动作差异。数值是截至该 horizon 的逐步平均 action L2。

| 初态 | 步数 | 固定 HDF5 观察的模型差异 | 固定 runtime 观察的模型差异 | 固定 HDF5 模型的观察差异 | 固定 runtime 模型的观察差异 |
|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 0.2105 | 0.2105 | 0 | 0 |
| 0 | 5 | 0.1858 | 0.1895 | 0.0148 | 0.0166 |
| 0 | 10 | 0.1366 | 0.1353 | 0.0206 | 0.0213 |
| 0 | 20 | 0.1090 | 0.1007 | 0.0281 | 0.0224 |
| 8 | 1 | 0.2609 | 0.2609 | 0 | 0 |
| 8 | 5 | 0.2151 | 0.2171 | 0.0395 | 0.0370 |
| 8 | 10 | 0.1486 | 0.1509 | 0.0281 | 0.0240 |
| 8 | 20 | 0.0977 | 0.1044 | 0.0273 | 0.0216 |

第 1 步两条 observation 完全相同，所以固定模型观察差异为零，而模型差异分别为 `0.2105` 和 `0.2609`，与初始观察诊断一致。到第 20 步，两条自由运行轨迹的平均 joint L2 差为 state 0 `0.0326`、state 8 `0.0385`；eye-in-hand RGB MAE 分别为 `7.73` 和 `8.71`，agentview RGB MAE 分别为 `1.26` 和 `1.11`。观察差异确实随运行累积，但在同一时间点的动作影响仍小于模型映射差异。

夹爪分量也呈相同排序：state 0 第 20 步前的逐步平均模型差异为 `0.0421`（HDF5 observation）和 `0.0397`（runtime observation），固定模型观察差异为 `0.0064` 和 `0.0033`；state 8 对应为 `0.0426`、`0.0426`、`0.0066`、`0.0069`。因此不能把 HDF5 臂前 80 步没有 close 简化成“RGB 轨迹差异直接导致夹爪没有学会”。

## 诊断判断

1. 当前两个 candidate 状态的首步运动分叉首先由两次训练得到的 Policy 输出映射差异产生；首步输入观察相同，因而不是两条 free-run 轨迹已经不同造成的。
2. 后续观察状态差异会累积，并对同一 Policy 的动作产生可测影响，但在前 20 步的平均量级低于固定观察下的跨模型动作差异。
3. 这项结果把“模型/动作 head 条件化差异”和“观察状态漂移敏感性”分开了，但没有进一步区分视觉 encoder、fusion、action head、控制器误差或数据覆盖。它也没有证明哪一臂的 observation source 更优。
4. 结果仅来自 task 7、一个 training seed 和两个 candidate 初态，仍是机制诊断；不构成 SPI 学习结果、SPT 更新证据或 Module qualification。

## 下一步

下一项应在不改变现有训练权重的前提下，扩展到 task 1/3/4/7 的共同失败与少数成功状态，并只保留相同的四格输入分解。若模型差异在不同 task/seed 上稳定占主导，再做 action-head 与视觉 encoder 的冻结/替换敏感性对照；若观察差异在特定任务显著放大，再回到任务条件覆盖和 runtime/HDF5 观测边界。只有完成跨 task、跨 seed 的机制复核后，才决定是否修订基础 Policy 训练协议。

