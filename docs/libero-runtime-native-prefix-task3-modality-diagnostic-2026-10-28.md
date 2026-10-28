# LIBERO task 3 RGB/本体输入模态分解诊断（2026-10-28）

## 目的与边界

跨任务四格诊断显示 task 3 的 observation 替换效应大于跨模型动作差，因此本项进一步固定 task 3、training seed `20261023` 的两个 Policy 和两个 candidate 状态 `0/1`，拆分两路 RGB 与 joint/gripper 输入的局部动作敏感性：

- `rgb_only`：替换 runtime-native 两路 RGB，保留 historical HDF5 joint/gripper；
- `proprio_only`：替换 runtime-native joint/gripper，保留 historical HDF5 两路 RGB；
- `both_modalities`：同时替换两类输入；
- `interaction`：`both - rgb_only - proprio_only + baseline` 的 signed action 向量，再报告其绝对距离。

协议：[task-3 modality diagnostic](../configs/libero_runtime_native_prefix_task3_modality_diagnostic_v1.json)，SHA-256 `083d201600f2a429bb2dac56d538f3806ca5c2f73ef8d7f985a600ed39d76d61`。结果文件位于实验主机 `/tmp/libero-runtime-native-prefix-task3-modality-20261028-v1.json`，SHA-256 `6d9f885d27a2e735a6cc249c831bf22e9897bd64e849b30bb4ba1636caa7b928`。结果使用跨任务诊断已核验的 observation archive：其 SHA-256 为 `226c85c406e0d2f237e37afc3171a1421de2755fff9bb2ba83396860aa0dd9b`。

重建两个 500-update CNN-only Policy，模型哈希分别为 historical HDF5 `c8eca2cedcf0ae4626fada2a20adc9e7ef2934283c34d5b364ca1c8f30e1a02d`、runtime-native `c310c4c37c9d8502f9f25ae4bc86e5a96d15e2ad53ae38e2cdd8b901d9aba136`。整个过程只在已保存 observation 上推理，没有执行交叉动作，没有访问 validation、held-out 或 qualification，没有选择 checkpoint、更新 SPT/Knowledge 或注册 Module。运行使用 Python `3.10.22`、PyTorch `2.4.1+cu124`、`cuda:1` / RTX 4090，耗时约 `12.95s`。

## 结果

下表为截至 horizon 的逐步平均七维 action MAE；四行分别对应两个 Policy 和两个 candidate 状态。

| Policy | 状态 | 3 步 RGB | 3 步本体 | 5 步 RGB | 5 步本体 | 20 步 RGB | 20 步本体 | 20 步联合 | 20 步交互 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HDF5 | 0 | 0.01319 | 0.00124 | 0.02851 | 0.00298 | 0.05149 | 0.01884 | 0.05121 | 0.00435 |
| HDF5 | 1 | 0.01225 | 0.00117 | 0.02867 | 0.00286 | 0.04201 | 0.01678 | 0.04273 | 0.00318 |
| runtime | 0 | 0.01204 | 0.00106 | 0.02552 | 0.00265 | 0.03812 | 0.01565 | 0.03773 | 0.00330 |
| runtime | 1 | 0.01054 | 0.00097 | 0.02393 | 0.00257 | 0.03217 | 0.01356 | 0.03170 | 0.00239 |

四个 Policy×状态组合平均到第 20 步：RGB action MAE `0.04095`，joint/gripper action MAE `0.01621`，两类输入联合 `0.04084`，交互项绝对 action MAE `0.00330`。RGB 项约为本体项的 `2.53` 倍；联合项与 RGB 项接近，交互项较小但非零。第 1 步所有替换项均为零，因为两臂初始 observation 完全相同。

## 诊断判断

1. 在 task 3 的这两个 candidate 状态和前 20 步局部范围内，Policy 输出对 RGB 轨迹差异更敏感，对 joint/gripper 差异较不敏感；这与跨任务诊断中 task 3 的 observation 主导结果一致。
2. RGB 项的主导不等于“CNN 已被证明是根因”。当前替换同时经过视觉 encoder、fusion 和 action head，不能仅凭该结果区分表示质量、相机内容、状态覆盖或动作映射。
3. 交互项相对较小，说明在这组局部输入替换中两类模态的非加性贡献有限，但它不是零，也不能用加法分解解释长程成功率。
4. 结果只来自 task 3、一个训练 seed、两个候选状态和固定 Policy；不是 RGB 修复效果、SPI qualification、SPT 更新或 Module 资格结果。

## 下一步

下一项在同一 task 3 observation archive 上分别只替换 agentview、只替换 eye-in-hand，并保留 historical joint/gripper 与固定 Policy，计算两路相机及其交互项。若 eye-in-hand 在两个 Policy、两个状态中稳定占主导，再决定是否需要单变量相机增强或目标几何覆盖诊断；不在本项结果上直接改训练协议。

