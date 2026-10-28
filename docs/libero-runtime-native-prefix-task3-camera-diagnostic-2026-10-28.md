# LIBERO task 3 两路相机敏感性诊断（2026-10-28）

## 目的与边界

task 3 的 RGB/本体分解显示 RGB 输入是局部动作敏感性的主要部分。本项继续固定相同的两个 500-update CNN-only Policy、两个 candidate 状态 `0/1` 和前 20 步 observation，只替换两路 RGB：

- `agentview_only`：替换 runtime-native agentview，保留 historical eye-in-hand 与 joint/gripper；
- `eye_in_hand_only`：替换 runtime-native eye-in-hand，保留 historical agentview 与 joint/gripper；
- `both_cameras`：同时替换两路 RGB，保留 historical joint/gripper；
- `interaction`：`both - agentview_only - eye_in_hand_only + baseline` 的 signed action 向量，再报告绝对距离。

协议：[task-3 camera diagnostic](../configs/libero-runtime-native-prefix-task3-camera-diagnostic_v1.json)，SHA-256 `bbc439f402246f3b906b03728ea9870b3f46c10d87a2219af755d85917283729`。结果文件位于实验主机 `/tmp/libero-runtime-native-prefix-task3-camera-20261028-v1.json`，SHA-256 `e4d1b2152bf061ffd30ec49ef1502b3b96c001c87e004be3b758e370d700ca23`。输入 observation archive SHA-256 为 `226c85c406e0d2f237e37afc3171a1421de2755fff9bb2ba83396860aa0dd9b3`。

本项只在已保存 observation 上离线推理，没有把替换后的动作送回环境，没有访问 validation、held-out 或 qualification，没有选择 checkpoint、更新 SPT/Knowledge 或注册 Module。运行使用 Python `3.10.22`、PyTorch `2.4.1+cu124`、`cuda:1` / RTX 4090；两个 Policy 各重建 500 updates，耗时约 `12.95s`。

## 结果

以下是四个 Policy×状态组合的逐步平均七维 action MAE：

| Policy | 状态 | 3 步 agentview | 3 步 eye-in-hand | 5 步 agentview | 5 步 eye-in-hand | 20 步 agentview | 20 步 eye-in-hand | 20 步联合 | 20 步交互 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HDF5 | 0 | 0.00108 | 0.01317 | 0.00119 | 0.02802 | 0.00356 | 0.05023 | 0.05149 | 0.00153 |
| HDF5 | 1 | 0.00052 | 0.01256 | 0.00056 | 0.02884 | 0.00296 | 0.04110 | 0.04201 | 0.00127 |
| runtime | 0 | 0.00120 | 0.01172 | 0.00132 | 0.02486 | 0.00416 | 0.03727 | 0.03812 | 0.00166 |
| runtime | 1 | 0.00053 | 0.01067 | 0.00071 | 0.02433 | 0.00338 | 0.03164 | 0.03217 | 0.00117 |

四个组合平均到第 20 步：agentview MAE `0.00351`，eye-in-hand MAE `0.04006`，双相机联合 `0.04095`，交互项绝对 MAE `0.00141`。eye-in-hand 项约为 agentview 项的 `11.4` 倍。第 3 步对应均值为 `0.00083`、`0.01203`、`0.01200`、`0.00011`；第 5 步为 `0.00095`、`0.02651`、`0.02666`、`0.00017`。四个组合的排序一致。

## 诊断判断

1. 在 task 3 的两个 candidate 状态、两个 observation arm 训练出的 Policy 和前 20 步范围内，eye-in-hand 替换产生的动作变化稳定大于 agentview 替换；这与 task 3 的 RGB 总体敏感性结果一致。
2. 双相机联合项几乎由 eye-in-hand 项决定，交互项较小但非零；局部输入替换中，末端相机是当前可测 RGB 敏感性的主要来源。
3. 这不是 eye-in-hand 相机或视觉 encoder 的长程因果证明。替换效应仍经过共享 CNN、fusion 和 action head，不能区分相机内容、目标几何覆盖、表示质量或动作映射。
4. 结果只来自 task 3、一个训练 seed、两个候选状态；不构成 observation-source 优势、SPI qualification、SPT 更新或 Module 资格结果。

## 下一步

下一项把相同相机分解扩展到已采集的 task 1/3/4/7、9 个固定状态，检查 eye-in-hand 主导是否跨任务保持。只有跨任务复核稳定后，才评估单变量 eye-in-hand 数据增强或目标几何覆盖诊断；当前不因本项局部敏感性直接修改 Policy 训练协议。

