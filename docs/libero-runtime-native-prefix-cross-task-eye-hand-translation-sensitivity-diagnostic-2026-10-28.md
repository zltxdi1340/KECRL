# LIBERO 跨任务 eye-in-hand 平移敏感性诊断（2026-10-28）

## 目的与边界

跨任务相机拆分已经显示 eye-in-hand 替换对动作输出的影响大于 agentview。本项进一步固定当前 8 个 500-update CNN-only uniform Policy、9 个候选状态和每个状态前 20 步 observation，只对 Policy 自己的 eye-in-hand 图像施加可控的整数平移，测量局部动作敏感性是否随几何扰动半径稳定增加。

每个 Policy 使用自己的 observation arm：historical HDF5 Policy 使用 historical observation，runtime-native Policy 使用 runtime-native observation。对 eye-in-hand RGB 施加反射填充的 `±1/±2/±4` 像素左右/上下平移；agentview、joint 和 gripper 保持原值。每一步都用未平移图像的 action 作为基线，报告平移后的 action 差异及 translation、rotation、gripper 分量。

协议：[cross-task eye-hand translation sensitivity](../configs/libero-runtime-native-prefix-cross-task-eye-hand-translation-sensitivity-diagnostic_v1.json)，SHA-256 `20547f586ab93d9ceea4537f5540d6ab738e5453419127a9bf3a77f1cc6bb08f`。结果文件位于实验主机 `/tmp/libero-runtime-native-prefix-cross-task-eye-hand-translation-sensitivity-20261028-v1.json`，SHA-256 `dab39a916d43f2eb2d8dbfad6690504367131983efbdb4afc88f9e71bb892ec7`。source camera result SHA-256 为 `d6f9359981115998f42695644ad609f8767e53955c27a629dae904e2fda47a72`，observation archive SHA-256 为 `226c85c406e0d2f237e37afc3171a1421de2755fff9bb2ba83396860aa0dd9b3`。

本项只做离线前向推理和固定 Policy 重建，没有将平移后的动作送入环境，没有训练增强臂，没有访问 validation、held-out 或 qualification，没有选择 checkpoint，没有更新 SPT/Knowledge，也没有注册 Module。运行使用 Python `3.10.22`、PyTorch `2.4.1+cu124`、`cuda:1` / NVIDIA GeForce RTX 4090；重建 8 个 Policy，耗时约 `57.11s`。

## 20 步结果

下表对 task 1/3/4/7、两种 observation arm 和四个方向取平均；action MAE 是七维 action 差异的逐元素绝对均值。

| 平移半径 | action MAE | translation L2 | rotation L2 | gripper absolute |
|---:|---:|---:|---:|---:|
| 1 px | 0.010919 | 0.040113 | 0.003704 | 0.011592 |
| 2 px | 0.021758 | 0.081013 | 0.007226 | 0.022650 |
| 4 px | 0.025939 | 0.095008 | 0.008452 | 0.028255 |

各任务在 20 步、两臂和四方向平均的 action MAE 为：

| task | 1 px | 2 px | 4 px |
|---:|---:|---:|---:|
| 1 | 0.010261 | 0.021671 | 0.028328 |
| 3 | 0.009256 | 0.016796 | 0.020687 |
| 4 | 0.006122 | 0.010927 | 0.011722 |
| 7 | 0.018038 | 0.037640 | 0.043019 |

方向并不对称。跨任务、两臂在 20 步的 action MAE 为：半径 1 时 left/right/up/down `0.00499/0.00464/0.01333/0.02072`，半径 4 时 `0.01477/0.01384/0.04127/0.03388`。task 7 的 gripper 分量最大，半径 4 的两臂平均 gripper absolute 为 `0.06111`；task 4 对应为 `0.00792`。

## 诊断判断

1. 当前固定 Policy 对 eye-in-hand 的小几何平移存在可测敏感性，并且总体随平移半径增加；这为“输入几何扰动可能影响动作读出”提供了跨任务机制证据。
2. 效应量强烈依赖任务和方向。task 7 的敏感性明显高于 task 4，说明后续增强不能只看全局平均，也要保留 task/phase/动作分量分层。
3. 这项结果只测固定 Policy 的局部输入—动作映射，不代表真实相机位姿误差、目标几何覆盖或闭环失败的因果效应。尤其没有说明训练增强一定有效。
4. 该诊断没有改变当前 CNN-only + uniform action MSE 的训练边界，也没有产生 SPI、SPT、Knowledge 或 Module 结论。

## 下一步

在相同任务、seed、数据、500-update 预算和 observation arm 下，新增唯一变量：训练阶段对 eye-in-hand 序列施加最大 4 像素、窗口内共享的随机反射平移；评估使用未增强图像。比较 baseline 与增强臂的离线 action error、上述平移敏感性以及 candidate 非资格闭环结果。只有多个任务/seed 同时改善，才考虑冻结视觉增强 pilot；若只降低固定状态敏感性而不改善闭环，则转向 action head 与状态覆盖诊断。

