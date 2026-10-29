# LIBERO task 4/7 冻结层级读出诊断（2026-10-29）

## 问题与设计

前一轮发现只训练 action head 时 source-support action MSE 远高于完整 Policy，但无法判断损失来自视觉 encoder、fusion/recurrent 映射，还是 action head。此项在 task 4 和 task 7、historical HDF5 与 runtime-native 两个 observation arm 上，固定初始化、batch 顺序、500 次 Adam 更新和 CNN-only uniform seven-dimensional action MSE，比较四种可训练范围：

| 读出臂 | 更新参数 | 参数量 |
|---|---|---:|
| `full_policy` | visual encoder、fusion、recurrent、action head | 821,704 |
| `freeze_visual` | fusion、recurrent、action head | 464,648 |
| `freeze_visual_fusion` | recurrent、action head | 396,552 |
| `action_head_only` | action head | 1,799 |

指标分两类：source-support prefix 上的动作监督误差；已归档 candidate observation 前 20 步上的模型输出差异和 gripper schedule。candidate observation 没有逐步专家 action 标签，因此不计算 candidate action error，也不执行冻结策略的环境动作。

## 协议与复现信息

- 配置：[module-freeze readout diagnostic](../configs/libero-runtime-native-prefix-task4-task7-module-freeze-readout-diagnostic_v1.json)，SHA-256 `be13ce1211050edbfbdbd157033bbae6384f25573ea4df81a3f3c14d64c2f3b3`。
- 结果：`/tmp/libero-runtime-native-prefix-task4-task7-module-freeze-readout-20261029-v3.json`，SHA-256 `870b7a25d5538c4b269bb16e582b9631bff9d1d62aa1df4a399716e8925fe7db`。
- 结果中记录的诊断脚本 SHA-256：`2b03f2fcc9955f6c6e8e104551368cc71694def1d6d8668f99b904abdb4c8240`；共享训练器 SHA-256：`780e26d1a50d452609fb24b2a7d4740351f4f044dadc03a286e8523bfbe22e22`。
- 运行环境：Python `3.10.22`、PyTorch `2.4.1+cu124`、`cuda:1` / NVIDIA GeForce RTX 4090；16 个模型各 500 updates，约 `58.42 s`。
- 归档 observation 在训练前按每个时间步 digest 校验；full Policy 参数 SHA 与先前 candidate coverage 结果相符，candidate 前 20 步动作与 pinned trace 在 `1e-6` 容差内一致。
- 没有访问 validation、held-out 或 qualification，没有执行冻结策略 rollout、选择 checkpoint、更新 SPT/Knowledge 或注册 Module。

## Source-support 动作监督误差

下表先在每个 task 内对两个 observation arm 取均值；任务和 arm 的原始值保存在结果 JSON。

| task | full Policy | 冻结 visual | 冻结 visual + fusion | 仅 action head |
|---:|---:|---:|---:|---:|
| 4 | 0.02894 | 0.04703 | 0.07046 | 0.21094 |
| 7 | 0.03295 | 0.05101 | 0.07588 | 0.18296 |

相对 full Policy，冻结 visual 的 MSE 分别为 `1.62×` 和 `1.55×`；进一步冻结 fusion 后为 `2.43×` 和 `2.30×`；只训练 action head 则为 `7.29×` 和 `5.55×`。gripper MSE 呈同样递进：

| task | full Policy | 冻结 visual | 冻结 visual + fusion | 仅 action head |
|---:|---:|---:|---:|---:|
| 4 | 0.08874 | 0.13530 | 0.18705 | 0.92749 |
| 7 | 0.11415 | 0.18470 | 0.28288 | 0.80643 |

因此，在这两个 task 和该训练预算下，端到端更新的 source-support 拟合最好；冻结 visual encoder 会损失一部分拟合能力，再冻结 fusion 后继续损失，冻结到只剩 action head 时误差最大。这个递进在两个 observation arm 上方向一致。它说明视觉特征与融合/状态映射都参与了拟合，但不构成模块的因果效应估计：冻结改变了可训练参数数量、梯度路径和优化问题；单 seed、source-support 指标也不能说明 query 泛化。

## Candidate observation 输出

`full_policy` 在 10 个固定 candidate 窗口上均复现旧 trace；旧 trace 中 3/10 成功。冻结策略没有执行环境 rollout，以下只描述相同 observation 输入下的输出：

| task | 读出臂 | 对 full Policy 的前 20 步 action MAE 均值 | 第 1–20 步出现 close 的案例 |
|---:|---|---:|---:|
| 4 | 冻结 visual | 0.06880 | 0/6 |
| 4 | 冻结 visual + fusion | 0.06875 | 0/6 |
| 4 | 仅 action head | 0.34468 | 6/6 |
| 7 | 冻结 visual | 0.08595 | 0/4 |
| 7 | 冻结 visual + fusion | 0.14273 | 0/4 |
| 7 | 仅 action head | 0.31552 | 4/4 |

这里的旧 trace success 只属于来源 full Policy，已在 JSON 中单独标记为 `source_trace_success/steps`，不能转称冻结臂的成败。action-head-only 在所有候选短窗口第 1 步都输出 close；其余冻结臂和 full Policy 前 20 步都没有 close。这反映固定输入上的命令映射不同，不能证明冻结臂在环境里会失败，也不能把提前 close 当作可用的策略修复。

## 结论与限制

本项将“不是只缺一个重新拟合的 action head”进一步收窄为：在当前架构和预算中，训练 visual encoder 与 fusion/recurrent 路径都能改善 source-support 拟合；task 7 对冻结 fusion 的 candidate 输出变化尤其明显。由于 freeze-scope 对照同时改变参数数量和优化自由度，不能把误差差值解读成视觉或 fusion 的纯因果贡献。当前只有 task 4/7、一个训练 seed 和 source-support 拟合指标，不能推广到全部任务、query 泛化或正式 SPI。

## 下一步

先在 task 4/7 和两 observation arm 上用另外两个初始化 seed 重复相同四臂 source-support 诊断，检查 MSE 层级是否稳定。这个复核只估计训练初始化敏感性；若排序复现，再另行设计能保持 HDF5/runtime-native 公平对齐的独立 query 评价，不能把 support 拟合提升写成泛化提升。所有阶段继续不访问 qualification、不更新 SPT/Knowledge、不注册 Module。
