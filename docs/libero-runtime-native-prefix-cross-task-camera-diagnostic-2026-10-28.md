# LIBERO 跨任务两路相机敏感性诊断（2026-10-28）

## 目的与边界

task 3 的相机拆分显示，固定 Policy 的动作输出对 eye-in-hand 替换比对 agentview 替换更敏感。本项把同一协议扩展到已经冻结的 task `1/3/4/7` 分层，检验这个排序是否只属于 task 3。

本项复用既有的 9 个 candidate 状态、8 个 500-update CNN-only uniform Policy 和前 20 步完整 observation archive，不重新 rollout 环境。每个任务使用一组历史 HDF5 observation Policy 和一组 runtime-native observation Policy；两臂分别在相同状态上做四种输入替换：

- `agentview_only`：替换 runtime-native agentview，保留 historical eye-in-hand 与 joint/gripper；
- `eye_in_hand_only`：替换 runtime-native eye-in-hand，保留 historical agentview 与 joint/gripper；
- `both_cameras`：同时替换两路 RGB，保留 historical joint/gripper；
- `interaction`：`both - agentview_only - eye_in_hand_only + baseline` 的 signed action 向量，再报告绝对距离。

协议：[cross-task camera diagnostic](../configs/libero-runtime-native-prefix-cross-task-camera-diagnostic_v1.json)，SHA-256 `de06227e81799561725c2ed5cffcb9c71540c0a439fb32dc25f41cef23e6f99e`。结果文件位于实验主机 `/tmp/libero-runtime-native-prefix-cross-task-camera-20261028-v1.json`，SHA-256 `d6f9359981115998f42695644ad609f8767e53955c27a629dae904e2fda47a72`。输入 observation archive SHA-256 为 `226c85c406e0d2f237e37afc3171a1421de2755fff9bb2ba83396860aa0dd9b3`，source cross-task result SHA-256 为 `c734966c4d84eb615e2b408df67303bc01ec9b06273410dbb70f69b06ac37c9e`，prefix result SHA-256 为 `b447373621906af25b60acf718652f714c1f1a08fac8e4380d4959999b1d302f`。

本项是离线前向诊断：没有把替换后的动作送回环境，没有访问 validation、held-out 或 qualification，没有选择 checkpoint，没有更新 SPT/Knowledge，也没有注册 Module。运行使用 Python `3.10.22`、PyTorch `2.4.1+cu124`、`cuda:1` / NVIDIA GeForce RTX 4090；重建 8 个 Policy，耗时约 `50.65s`。

## 第 20 步结果

表中每个数是该任务选定状态和两种 Policy 臂的 task-level 平均七维 action MAE；interaction 是 signed interaction 向量的绝对 MAE。

| task | agentview only | eye-in-hand only | both cameras | interaction | eye/agentview |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.000113 | 0.003958 | 0.003956 | 0.000010 | 35.0× |
| 3 | 0.003513 | 0.040062 | 0.040946 | 0.001408 | 11.4× |
| 4 | 0.000392 | 0.005266 | 0.005346 | 0.000036 | 13.4× |
| 7 | 0.001014 | 0.006391 | 0.006447 | 0.000058 | 6.3× |

两种 observation arm 分开看，eye-in-hand 仍在每个任务中大于 agentview。第 20 步的 eye/agentview 比值范围为：

- historical HDF5 arm：task 1 `30.7×`、task 3 `14.0×`、task 4 `12.2×`、task 7 `9.1×`；
- runtime-native arm：task 1 `41.8×`、task 3 `9.2×`、task 4 `14.8×`、task 7 `4.9×`。

在所有任务中，双相机替换与 eye-in-hand-only 接近；interaction 相对较小，但 task 3 的 interaction 明显高于其他任务，说明两路输入在该任务上的非加性更强。效应量不能跨任务直接比较为成功率或性能提升，因为本项只测固定 observation 上的局部动作输出变化。

## 诊断判断

1. task 3 中观察到的 eye-in-hand 主导排序在当前 task `1/3/4/7`、9 个预先固定状态、8 个固定 Policy 上均复现。它不再是单任务、单状态的孤立现象。
2. 这项证据把下一步问题收窄到 eye-in-hand 输入通道及其下游映射：需要继续区分相机内容/几何覆盖、CNN 表示、fusion 和 action head，不能把结果直接写成“相机硬件是根因”。
3. task 3 的绝对敏感性最大，task 1 的 agentview 基线极小而 eye-in-hand 效应仍明显；因此跨任务复现的是相对排序，不是统一的误差规模或统一失败机制。
4. 结果仍不构成 observation-source 优势、SPI qualification、SPT 更新、Knowledge 更新或 Module 资格结论。固定 Policy 的敏感性也不能代替闭环干预结果。

## 下一步

先保持 CNN-only + uniform action MSE 和现有 HDF5 causal 离线训练边界不变，进行一个单变量的 eye-in-hand 诊断。优先比较同一任务/状态/Policy 上的 eye-in-hand 几何覆盖与输入增强，要求保留 agentview、joint/gripper、训练预算、seed 和评价分层；只在独立诊断中测 action sensitivity、短窗口 action error 和闭环候选结果。若该单变量干预在多个任务和 seed 上方向一致，再把它写入新的 candidate protocol；在此之前不把局部敏感性当作视觉修复收益，也不启动 SPI qualification 或 SPT 更新。

