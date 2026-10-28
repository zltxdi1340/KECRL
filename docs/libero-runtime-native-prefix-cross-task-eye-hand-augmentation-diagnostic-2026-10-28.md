# LIBERO 跨任务 eye-in-hand 平移增强诊断（2026-10-28）

## 目的与协议

前一项固定 Policy 敏感性诊断发现，eye-in-hand 的小幅图像平移会明显改变动作输出。本项检查训练时对 eye-in-hand 做随机平移，是否能在当前 task 1/3/4/7 candidate 分层上同时改善动作拟合、降低局部敏感性并保持闭环候选表现。

按 task 和 observation arm 分别训练 baseline 与增强臂。两个臂使用相同初始化、batch 顺序、数据前缀和 500 optimizer updates；增强臂只在训练时对 eye-in-hand 序列施加窗口内共享、反射填充、最大 4 像素随机平移。离线评估和 candidate rollout 均使用未增强图像。每个 task 使用一个既有 training seed、2–3 个预先固定 candidate 初始状态。闭环 episode 只属于 `candidate_nonregression`，不属于 qualification。

协议：[cross-task eye-hand augmentation diagnostic](../configs/libero-runtime-native-prefix-cross-task-eye-hand-augmentation-diagnostic_v1.json)，SHA-256 `58ac4a5802dd47c8a8e513b352bf5952765b4de6674999afa019666582634039`。结果文件位于实验主机 `/tmp/libero-runtime-native-prefix-cross-task-eye-hand-augmentation-20261028-v1.json`，SHA-256 `06e309d7f22bb04dd0b18a5759388989fa56591af015fece3ad4de11eb9f1ecc`。前置敏感性结果 SHA-256 为 `dab39a916d43f2eb2d8dbfad6690504367131983efbdb4afc88f9e71bb892ec7`；observation archive SHA-256 为 `226c85c406e0d2f237e37afc3171a1421de2755fff9bb2ba83396860aa0dd9b3`；prefix result SHA-256 为 `b447373621906af25b60acf718652f714c1f1a08fac8e4380d4959999b1d302f`。

运行使用 Python `3.10.22`、PyTorch `2.4.1+cu124`、`cuda:1` / NVIDIA GeForce RTX 4090。共重建 16 个模型，每个 500 updates；candidate 共运行 36 个 episode，运行时间约 `322.21s`。没有访问 validation、held-out 或 qualification，没有选择 checkpoint，没有更新 SPT/Knowledge，也没有注册 Module。

## 结果

### 离线拟合

未增强图像上的动作 MSE 在四个 task 都变差：

| task | baseline MSE | augmentation MSE | 相对变化 |
|---:|---:|---:|---:|
| 1 | 0.043059 | 0.045764 | +6.3% |
| 3 | 0.034470 | 0.038981 | +13.1% |
| 4 | 0.028940 | 0.031109 | +7.5% |
| 7 | 0.032949 | 0.035956 | +9.1% |
| 按有效步数汇总 | 0.034627 | 0.037607 | +8.6% |

### 固定状态敏感性

在固定候选 observation 上，增强 Policy 对眼在手输入平移的 action MAE 大多明显下降。第 20 步、半径 4、两 observation arm 与四方向平均：

| task | baseline | augmentation | 相对变化 |
|---:|---:|---:|---:|
| 1 | 0.028328 | 0.006648 | -76.5% |
| 3 | 0.020687 | 0.017501 | -15.4% |
| 4 | 0.011722 | 0.008398 | -28.4% |
| 7 | 0.043019 | 0.010861 | -74.8% |
| 四任务平均 | 0.024359 | 0.010580 | -56.6% |

在半径 1/2/4 上总体都下降。例如第 20 步总体 action MAE 分别从 `0.010386/0.020555/0.024359` 降至 `0.003918/0.006262/0.010580`。这是输入—动作局部敏感性变化，不是闭环控制能力提升。

### Candidate 非资格闭环

| task | baseline 成功 | augmentation 成功 |
|---:|---:|---:|
| 1 | 1/4 | 1/4 |
| 3 | 1/4 | 1/4 |
| 4 | 3/6 | 1/6 |
| 7 | 1/4 | 0/4 |
| 合计 | 6/18 | 3/18 |

18 个配对 episode 中，两臂都成功 `2` 个，只有 baseline 成功 `4` 个，只有 augmentation 成功 `1` 个，两臂都失败 `11` 个。该小样本只作诊断性描述，不用于统计显著性或 qualification。

## 诊断判断

1. 训练增强强烈降低了固定状态下对眼在手平移的动作敏感性，但这种不变性没有转化为当前未增强专家分布上的拟合或 candidate 闭环收益。
2. 离线 MSE 在四个任务一致恶化，candidate 成功总数从 `6/18` 降至 `3/18`，下降集中在 task 4 和 task 7。当前 4 像素随机平移增强不应直接扩大为正式训练或 SPI qualification 设置。
3. 更合理的暂定解释是：敏感性并非越低越好；平移扰动也可能削弱任务所需的局部视觉定位或和 action head 的细粒度映射。本项没有直接证明是哪一种机制，也没有识别相机硬件、CNN 表示、状态覆盖或闭环反馈的根因。
4. 当前数据只有每个任务一个 training seed，且 candidate episode 数少；结果足以停止无条件扩大这一增强臂，但不足以判断其他增强强度、方向分布或更大数据覆盖是否有效。

## 下一步

暂停扩大 eye-in-hand 随机平移增强。下一项回到不改 Policy 的机制诊断，利用已有 RGB/action/joint/gripper trace，检查任务阶段与可观测目标接近线索能否预测早期平移 action error；重点比较 task 7 与 task 4，并保持不使用 simulator 私有目标坐标。若发现误差集中在特定接近几何或动作阶段，再设计不抹除几何信息的覆盖补充或 action-head 诊断。现有 SPT/SPI 资格边界和 CNN-only + uniform action MSE 基线均不变。

