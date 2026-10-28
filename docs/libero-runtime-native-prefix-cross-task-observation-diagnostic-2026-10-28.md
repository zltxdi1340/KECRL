# LIBERO 跨任务早期观察与交叉输入诊断（2026-10-28）

## 目的与边界

task 7 的交叉输入诊断只覆盖一个任务，不能说明模型映射差异是否是所有任务的首要因素。本项沿用同一四格输入分解，扩展到既有 failure-trace 中固定的 task 1、3、4、7 状态：

| 模型 | 输入观察 | 解释 |
|---|---|---|
| historical HDF5 Policy | historical HDF5 轨迹 | 原始配对 |
| runtime-native Policy | historical HDF5 轨迹 | 固定观察下的跨模型映射差异 |
| historical HDF5 Policy | runtime-native 轨迹 | 固定模型下的观察敏感性 |
| runtime-native Policy | runtime-native 轨迹 | 原始配对 |

协议：[cross-task observation diagnostic](../configs/libero_runtime_native_prefix_cross_task_observation_diagnostic_v1.json)，SHA-256 `afab0ca6c1b63b9a2f7cfa5d6587113e1106ccbcc3be2aa4e732678bef26a926`。结果文件位于实验主机 `/tmp/libero-runtime-native-prefix-cross-task-observation-diagnostic-20261028-v1.json`，SHA-256 `c734966c4d84eb615e2b408df67303bc01ec9b06273410dbb70f69b06ac37c9e`；逐步 observation 归档位于 `/tmp/libero-runtime-native-prefix-cross-task-observations-20261028-v1.npz`，SHA-256 `226c85c406e0d2f237e37afc3171a1421de2755fff9bb2ba83396860aa0dd9b`。

本项重建 8 个固定 500-update CNN-only Policy，使用 task 1/3/4/7 的训练 seed `20261023/20261023/20261022/20261022` 和评估 seed `20261021`。每个模型按候选初态 `0..19` 顺序复用同一 adapter；9 个预先固定状态（task 1: `0,17`；task 3: `0,1`；task 4: `0,2,7`；task 7: `0,8`）各保存两臂前 20 步的完整 Policy observation，共 72 个压缩数组。所有候选状态的 success/终止步数，以及选中状态前 20 步原始 Policy action，均与已有 coverage/failure-trace 逐项核验通过。运行提交为 `37dc609`，Python `3.10.22`、PyTorch `2.4.1+cu124`、`cuda:1` / RTX 4090，耗时约 `883.78s`。

交叉组合只在内存中对已采集 observation 推理；交叉模型的 action 没有送回环境。因此本项是机制诊断，不是干预实验、SPI qualification、SPT 更新或 Module 资格检查。没有访问 validation、held-out 或 qualification，没有选择 checkpoint，也没有更新 SPT/Knowledge。

## 固定状态与闭环结果

这些状态来自已经冻结的 failure-trace 分层，不是随机抽样，也不能用来估计任务成功率。

| task | training seed | 初态 | HDF5 arm | runtime-native arm |
|---:|---:|---:|---|---|
| 1 | 20261023 | 0 | 失败/300 | 失败/300 |
| 1 | 20261023 | 17 | 失败/300 | 成功/131 |
| 3 | 20261023 | 0 | 失败/300 | 成功/90 |
| 3 | 20261023 | 1 | 失败/300 | 失败/300 |
| 4 | 20261022 | 0 | 失败/300 | 失败/300 |
| 4 | 20261022 | 2 | 失败/300 | 成功/159 |
| 4 | 20261022 | 7 | 成功/156 | 失败/300 |
| 7 | 20261022 | 0 | 失败/300 | 失败/300 |
| 7 | 20261022 | 8 | 失败/300 | 成功/137 |

## 交叉输入结果

下表是截至第 20 步的逐步平均 action L2。`模型差异`固定 HDF5 observation，比较两臂 Policy；`观察差异`固定 HDF5 Policy，比较两条 free-run observation 轨迹。runtime-native observation 作 reference 的对应结果保存在结果 JSON 中，个别状态的大小关系会略有变化。

| task | 状态 | 模型差异 | 观察差异 | 20 步观察 joint L2 | 20 步 eye-in-hand RGB MAE |
|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 0.0589 | 0.0217 | 0.0150 | 6.43 |
| 1 | 17 | 0.0455 | 0.0117 | 0.0219 | 5.40 |
| 3 | 0 | 0.1966 | 0.2184 | 0.1816 | 18.31 |
| 3 | 1 | 0.1767 | 0.1835 | 0.1633 | 16.73 |
| 4 | 0 | 0.0862 | 0.0215 | 0.0420 | 5.03 |
| 4 | 2 | 0.0993 | 0.0153 | 0.0408 | 4.84 |
| 4 | 7 | 0.0835 | 0.0244 | 0.0423 | 9.46 |
| 7 | 0 | 0.1090 | 0.0281 | 0.0326 | 7.73 |
| 7 | 8 | 0.0977 | 0.0273 | 0.0385 | 8.71 |

按 task 求均值后，第 20 步的 HDF5-observation 模型差异/观察差异分别为：task 1 `0.0522/0.0167`，task 3 `0.1866/0.2010`，task 4 `0.0897/0.0204`，task 7 `0.1034/0.0277`。第 1 步两臂初始 observation 完全相同，9/9 状态的观察差异为零；到第 20 步，7/9 状态模型差异仍大于观察差异，task 3 的两个状态则均由观察差异占主导。

## 诊断判断

1. task 7 的“早期模型映射差异优先”只在 task 1、4、7 的这些固定样本中大体成立，不能推广到 task 3。
2. task 3 的两条轨迹在第 5 步已经出现约 `0.024` joint L2 和约 `5.5` eye-in-hand RGB MAE，到第 20 步放大到约 `0.16--0.18` joint L2、`16.7--18.3` RGB MAE；固定模型的动作变化超过跨模型变化。这使 task 3 成为下一项拆分 RGB 与本体输入贡献的优先对象。
3. task 1、4、7 的观察替换效应较小，但仍随闭环累积；不能把它们归因为观察完全无关。task 3 的结果也不能单独证明 RGB 是根因，因为当前指标把两路 RGB、joint 和 gripper 的变化合在一起。
4. 选中状态包含 runtime-only、HDF5-only 和共同失败，但没有被选中的成功/失败比例估计；observation-source 优势、SPI 学习效果和 Module 资格均未得到证明。

## 下一步

固定本项已采集的 task 3 observation 与两臂 Policy，分别只替换 RGB、只替换 joint/gripper、同时替换两类输入，计算与本项相同的 action L2 及七维分量。该拆分沿用已有模态分解定义，不执行交叉动作，不重新选择模型。若 task 3 的 RGB 项明显占主导，再做相机分解；若本体项或交互项显著，则回到控制状态漂移和 action head 条件化诊断。结果仍只用于机制定位，不修改 SPI/SPT 协议。

